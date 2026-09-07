import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import xgboost as xgb
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem   # 标准 Morgan 位指纹 ECFP4

rdBase.DisableLog("rdApp.warning")
np.random.seed(42)

# ============================ 配置 ============================
REPORT_DIR = Path(r"D:\GitHub\RepoRT\processed_data")
TOP_N = 20
N_SPLITS = 10
RT_COL = "rt"
N_TOP_FEATURES = 15
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

RDKIT_NAMES = [name for name, _ in Descriptors._descList]


# ======================== 文件 I/O =========================
def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="utf-8", encoding_errors="strict")


def get_rt_file(folder: Path):
    ds = folder.name
    iso = folder / f"{ds}_rtdata_isomeric_success.tsv"
    canon = folder / f"{ds}_rtdata_canonical_success.tsv"
    if iso.exists():
        return iso, "isomeric"
    if canon.exists():
        return canon, "canonical"
    return None, None


def count_datasets(root: Path) -> pd.DataFrame:
    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        path, source = get_rt_file(folder)
        if path is None:
            continue
        df = read_tsv(path).dropna(subset=[RT_COL])
        rows.append({"dataset_id": folder.name, "source": source, "n_records": len(df)})
    return pd.DataFrame(rows)


def select_top20(counts: pd.DataFrame) -> pd.DataFrame:
    return (counts.dropna(subset=["source"])
                 .sort_values(["n_records", "dataset_id"], ascending=[False, True])
                 .head(TOP_N).reset_index(drop=True))


# ======================== 特征工程 =========================
def mol_features(smiles):
    """
    [RDKit_2D, Morgan_ECFP4 (MORGAN_BITS, 0/1)] 定长向量。
    Morgan 用标准 ECFP4 (AllChem, radius=2, 2048 bits)，独立 RNG 不污染全局。
    """
    mol = Chem.MolFromSmiles(str(smiles))
    n_desc = len(Descriptors._descList)
    if mol is None:
        return np.zeros(n_desc + MORGAN_BITS, dtype=np.float32)

    rdkit = [func(mol, avg=True) if name == "Ipc" else func(mol)
             for name, func in Descriptors._descList]

    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS)
    morgan = (np.frombuffer(bitvect.ToBitString().encode("ascii"), dtype=np.uint8) - 48).astype(np.float32)

    return np.concatenate([np.array(rdkit, dtype=np.float32), morgan])


def build_Xy(df: pd.DataFrame):
    """返回 (X DataFrame 带真实列名, y ndarray)，X/y 必然等长。"""
    df = df.dropna(subset=[RT_COL, "smiles.std"]).copy()
    df[RT_COL] = pd.to_numeric(df[RT_COL], errors="coerce")
    df = df.dropna(subset=[RT_COL]).reset_index(drop=True)

    feats = [np.nan_to_num(mol_features(s), nan=0.0)
             for s in df["smiles.std"].tolist()]
    X = pd.DataFrame(np.vstack(feats))

    col_names = [f"RDKit_{n}" for n in RDKIT_NAMES] + \
                [f"Morgan_{i}" for i in range(MORGAN_BITS)]
    X.columns = col_names[:len(X.columns)]
    X = X.dropna(axis=1, how="all").loc[:, lambda d: d.var() > 0].reset_index(drop=True)

    y = df[RT_COL].to_numpy()
    return X, y


# =================== 单数据集 10 折 OOF CV ==================
def evaluate_dataset(df: pd.DataFrame):
    """返回 (metrics, y_true, oof_pred, top_feats)"""
    if len(df) < N_SPLITS:
        raise ValueError(f"样本数不足 {N_SPLITS}，无法 10 折 CV")

    X, y = build_Xy(df)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof = np.zeros(len(df))
    importances = {}

    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X.iloc[tr])
        Xtr = pd.DataFrame(sc.transform(X.iloc[tr]), columns=X.columns)
        Xte = pd.DataFrame(sc.transform(X.iloc[te]), columns=X.columns)

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, tree_method="hist", eval_metric="rmse"
        )
        model.fit(Xtr, y[tr])
        oof[te] = model.predict(Xte)

        for name, gain in model.get_booster().get_score(importance_type="gain").items():
            importances[name] = importances.get(name, 0.0) + gain

    total = sum(importances.values()) or 1.0
    importances = {k: v / total for k, v in importances.items()}
    top_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:N_TOP_FEATURES]

    metrics = {
        "n": len(df),
        "Q2": r2_score(y, oof),
        "MAE": mean_absolute_error(y, oof),
        "RMSE": np.sqrt(mean_squared_error(y, oof)),
    }
    return metrics, y, oof, top_feats


# ============================ Main ==========================
def main():
    print("=== 实验A：同条件基线 (每个 dataset 内部 10 折 OOF CV) ===\n")
    if not REPORT_DIR.exists():
        raise FileNotFoundError(f"REPORT_DIR 不存在: {REPORT_DIR}")

    counts = count_datasets(REPORT_DIR)
    top20 = select_top20(counts)
    print("前 20 个数据集 (RT success 记录数降序):")
    for _, r in top20.iterrows():
        print(f"  {r['dataset_id']}\t{r['source']}\t{r['n_records']}")

    results = []
    all_oof = []
    all_top_features = []

    for _, r in top20.iterrows():
        ds = str(r["dataset_id"])
        folder = REPORT_DIR / ds
        path, source = get_rt_file(folder)
        if path is None:
            continue

        df = read_tsv(path)
        metrics, y_true, oof, top_feats = evaluate_dataset(df)

        metrics["dataset_id"] = ds
        metrics["source"] = source
        results.append(metrics)
        all_top_features.append(dict(top_feats))

        all_oof.append(pd.DataFrame({
            "dataset_id": ds,
            "y_true": y_true,
            "y_pred": oof,
        }))

        print(f"  {ds}: n={metrics['n']}, Q²={metrics['Q2']:.4f}, "
              f"MAE={metrics['MAE']:.4f} min, RMSE={metrics['RMSE']:.4f} min")

    # ================= 保存 3 个 CSV =================
    results_df = pd.DataFrame(results)
    oof_df_all = pd.concat(all_oof, ignore_index=True)
    results_df.to_csv("experiment_A_results.csv", index=False)
    oof_df_all.to_csv("experiment_A_oof.csv", index=False)

    feat_counter = Counter()
    for feat_dict in all_top_features:
        for name, val in feat_dict.items():
            feat_counter[name] += val
    pd.DataFrame(feat_counter.items(), columns=["feature", "gain"]) \
        .sort_values("gain", ascending=False) \
        .to_csv("experiment_A_feature_importance.csv", index=False)

    print(f"\n[已保存] experiment_A_results.csv ({len(results_df)} datasets)")
    print(f"[已保存] experiment_A_oof.csv ({len(oof_df_all)} rows)")
    print(f"[已保存] experiment_A_feature_importance.csv ({len(feat_counter)} features)")


if __name__ == "__main__":
    main()