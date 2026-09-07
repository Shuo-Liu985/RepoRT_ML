import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')
import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import xgboost as xgb
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, AllChem

from condition_encoder import (
    load_dataset_metadata, build_condition_features, get_condition_column_names,
)

rdBase.DisableLog("rdApp.warning")
np.random.seed(42)

# ============================ 配置 (只改 REPORT_DIR) ============================
REPORT_DIR = Path(r"D:\GitHub\RepoRT\processed_data")
TOP_N = 20
N_GROUPS = 5
RT_COL = "rt"
N_TOP_FEATURES = 15
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
DEDUP = "smiles"
Q2_OUTLIER = -1.0
MIN_SAMPLES = 50
RDKIT_NAMES = [name for name, _ in Descriptors._descList]
BASE = Path(__file__).parent / "output"

# ★ 可选：排除化学空间离群的数据集 (例如 0186)，单独分析
EXCLUDE_DATASETS = set()   # 例如 {"0186"}，默认为空


def read_tsv(path):
    return pd.read_csv(path, sep="\t", encoding="utf-8", encoding_errors="strict")


def get_rt_file(folder):
    ds = folder.name
    iso = folder / f"{ds}_rtdata_isomeric_success.tsv"
    canon = folder / f"{ds}_rtdata_canonical_success.tsv"
    if iso.exists():
        return iso, "isomeric"
    if canon.exists():
        return canon, "canonical"
    return None, None


def select_top20(root):
    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        path, source = get_rt_file(folder)
        if path is None:
            continue
        n = read_tsv(path).dropna(subset=[RT_COL]).shape[0]
        rows.append({"dataset_id": folder.name, "source": source, "n_records": n})
    df = pd.DataFrame(rows).dropna(subset=["source"])
    return df.sort_values(["n_records", "dataset_id"], ascending=[False, True]).head(TOP_N).reset_index(drop=True)


def canon(smiles):
    try:
        return Chem.CanonSmiles(str(smiles))
    except Exception:
        return str(smiles)


def mol_features(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    n_desc = len(Descriptors._descList)
    if mol is None:
        return np.zeros(n_desc + MORGAN_BITS, dtype=np.float32)
    rdkit = [func(mol, avg=True) if name == "Ipc" else func(mol)
             for name, func in Descriptors._descList]
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS)
    morgan = (np.frombuffer(bitvect.ToBitString().encode("ascii"), dtype=np.uint8) - 48).astype(np.float32)
    return np.concatenate([np.array(rdkit, dtype=np.float32), morgan])


def build_molecular_features(df):
    feats = [np.nan_to_num(mol_features(s), nan=0.0) for s in df["smiles.std"].tolist()]
    X = pd.DataFrame(np.vstack(feats))
    cols = [f"RDKit_{n}" for n in RDKIT_NAMES] + [f"Morgan_{i}" for i in range(MORGAN_BITS)]
    X.columns = cols[:len(X.columns)]
    X = X.dropna(axis=1, how="all").loc[:, lambda d: d.var() > 0].reset_index(drop=True)
    return X


# ======================== 训练 (防 NaN + 早停 + 退化) ========================
def _build_params():
    return {
        "objective": "reg:squarederror",
        "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.5, "reg_lambda": 1.0, "min_child_weight": 5,
        "tree_method": "hist", "eval_metric": "rmse", "seed": 42,
    }


def _train_fixed(Xtr, Xte, ytr):
    """固定 300 棵, 返回 (booster, pred), 绝不返回 NaN"""
    dtrain = xgb.DMatrix(Xtr.values, label=ytr, feature_names=list(Xtr.columns))
    dtest = xgb.DMatrix(Xte.values, feature_names=list(Xte.columns))
    booster = xgb.train(_build_params(), dtrain, num_boost_round=300)
    return booster, booster.predict(dtest)


def train_and_predict(X, y, tr, te, early_eval=True):
    Xtr = X.iloc[tr].copy(); Xte = X.iloc[te].copy()
    sc = RobustScaler(with_centering=False, quantile_range=(10, 90)).fit(Xtr)
    Xtr = pd.DataFrame(sc.transform(Xtr), columns=X.columns)
    Xte = pd.DataFrame(sc.transform(Xte), columns=X.columns)
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xte = Xte.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    booster, pred = _train_fixed(Xtr, Xte, y[tr])   # 兜底 (必为干净数值)
    used_es = False

    if early_eval and len(te) >= 20:
        try:
            dtrain = xgb.DMatrix(Xtr.values, label=y[tr], feature_names=list(X.columns))
            dtest = xgb.DMatrix(Xte.values, label=y[te], feature_names=list(X.columns))
            booster = xgb.train(
                _build_params(), dtrain, num_boost_round=1000,
                evals=[(dtest, "eval")], early_stopping_rounds=30, verbose_eval=False)
            # 兼容不同 xgboost 版本的 best_iteration 获取
            best = getattr(booster, "best_iteration", None)
            if best is None and hasattr(booster, "attr") and "best_iteration" in booster.attr():
                best = int(booster.attr("best_iteration"))
            best = int(best) if best else 300
            pred_es = booster.predict(dtest, iteration_range=(0, best + 1))
            if not np.isnan(pred_es).any():
                pred = pred_es
                used_es = True
                print(f"    [early_stopping] 成功, best_iteration={best}")
            else:
                print("    [early_stopping] 结果含NaN, 用固定树数")
        except Exception as e:
            print(f"    [early_stopping] 失败, 用固定树数: {type(e).__name__}")

    if not used_es:
        print("    [退化] 固定 300 棵")

    pred = np.asarray(pred, dtype=np.float64)
    if np.isnan(pred).any():
        print("    [WARN] 预测仍含NaN, 用训练集中位数填充")
        pred = np.nan_to_num(pred, nan=np.nanmedian(y[tr]))
    return pred, booster


def accum_gain(importance_accum, model_or_booster):
    try:
        score = model_or_booster.get_score(importance_type="gain")
    except AttributeError:
        score = model_or_booster.get_booster().get_score(importance_type="gain")
    for name, gain in score.items():
        importance_accum[name] = importance_accum.get(name, 0.0) + gain


def metric_dict(y_true, y_pred, n):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if np.isnan(y_pred).all() or np.std(y_pred) == 0 or len(y_true) < 2:
        return {"n": n, "Q2": np.nan, "MAE": np.nan, "RMSE": np.nan}
    return {"n": n, "Q2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))}


def robust_aggregate(results_df, col):
    """median [IQR], 剔除 Q2 离群折; 全 NaN / 单元素安全"""
    df = results_df.copy().dropna(subset=[col])
    if col == "Q2" and Q2_OUTLIER is not None:
        n_before = len(df)
        df = df[df[col] > Q2_OUTLIER]
        n_removed = n_before - len(df)
        if n_removed:
            print(f"    [Winsorize] 剔除 {n_removed} 个 Q2<={Q2_OUTLIER:.0f} 的离群折")

    # ★ 先强制转数值再 dropna，杜绝残留 NaN
    vals = pd.to_numeric(df[col], errors='coerce').dropna().values.astype(float)
    if len(vals) == 0:
        return np.nan, np.nan, df
    if len(vals) == 1:
        return float(vals[0]), 0.0, df

    # ★ 安全计算分位数（双重保障）
    with np.errstate(invalid='ignore'):
        med = float(np.nanmedian(vals))
        q1 = float(np.nanquantile(vals, 0.25))
        q3 = float(np.nanquantile(vals, 0.75))
    return med, q3 - q1, df

# ============================ Main ==========================
def main():
    os.makedirs(BASE, exist_ok=True)
    print(f"[输出目录] {BASE.resolve()}\n")

    print("=== Global/Universal Model: 跨条件泛化 (Top-20 + 精简条件编码) ===\n")
    if not REPORT_DIR.exists():
        raise FileNotFoundError(f"REPORT_DIR 不存在: {REPORT_DIR}")

    top20 = select_top20(REPORT_DIR)
    print(f"Top-{TOP_N} 数据集 (按 n_records 降序):")
    for _, r in top20.iterrows():
        print(f"  {r['dataset_id']}\t{r['source']}\t{r['n_records']}")

    seen = set()
    parts = []
    meta_rows = []
    for _, r in top20.iterrows():
        ds = str(r["dataset_id"])
        if ds in EXCLUDE_DATASETS:
            print(f"  [EXCLUDE] {ds}: 移出训练集 (单独外推测试)")
            continue
        folder = REPORT_DIR / ds
        path, source = get_rt_file(folder)
        if path is None:
            continue
        df = read_tsv(path); df["source"] = source
        before = len(df)
        if DEDUP == "smiles":
            df = df.dropna(subset=["smiles.std", RT_COL])
            df = df[~df["smiles.std"].apply(canon).duplicated()]
        elif DEDUP == "smiles_global":
            df = df.dropna(subset=["smiles.std", RT_COL])
            df = df[~df["smiles.std"].apply(lambda s: (c := canon(s)) in seen)]
            seen.update(df["smiles.std"].apply(canon))
        meta_row = load_dataset_metadata(folder, ds)
        if meta_row is not None:
            meta_rows.append(meta_row)
        print(f"  {ds}: {before} -> {len(df)} (去重:{DEDUP}) [{'OK' if meta_row is not None else 'MISSING'}]")
        parts.append((ds, folder, df))

    cond_cols = get_condition_column_names(meta_rows)
    print(f"\n[色谱条件] 共 {len(cond_cols)} 维 (精简后)")

    all_X, all_y, groups = [], [], []
    for ds, folder, df in parts:
        df = df.dropna(subset=[RT_COL, "smiles.std"]).copy()
        df[RT_COL] = pd.to_numeric(df[RT_COL], errors="coerce")
        df = df.dropna(subset=[RT_COL]).reset_index(drop=True)
        if len(df) < MIN_SAMPLES:
            print(f"  [SKIP] {ds}: 仅 {len(df)} 样本 (<{MIN_SAMPLES}, 排除)")
            continue
        X_mol = build_molecular_features(df)
        meta_row = load_dataset_metadata(folder, ds)
        X_cond = build_condition_features(df, ds, metadata=meta_row, all_columns=cond_cols)
        all_X.append(pd.concat([X_mol, X_cond], axis=1))
        all_y.append(df[RT_COL].to_numpy())
        groups.extend([ds] * len(df))
    if not all_X:
        raise RuntimeError("无可用数据集 (试试 DEDUP='smiles')")

    X = pd.concat(all_X, ignore_index=True)
    y = np.concatenate(all_y)
    groups = np.array(groups)
    print(f"\n[联合数据] 样本={len(X)}, 特征={X.shape[1]} (分子 + 条件 {len(cond_cols)}维)")

    print(f"\n[GroupKFold-{N_GROUPS}] 按 dataset_id 分组...")
    gkf = GroupKFold(n_splits=N_GROUPS)
    importance_accum = {}
    fold_records = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        pred, model = train_and_predict(X, y, tr, te)
        accum_gain(importance_accum, model)
        m = metric_dict(y[te], pred, len(te))
        fold_records.append({"fold": fold + 1, **m})
        print(f"  Fold {fold+1}: Q2={m['Q2']:.4f}, MAE={m['MAE']:.4f} min")
    fold_df = pd.DataFrame(fold_records)
    print("  [GroupKFold 聚合]")
    for col in ["Q2", "MAE", "RMSE"]:
        med, iqr, _ = robust_aggregate(fold_df, col)
        print(f"    {col}: {med:.4f} [{iqr:.4f}]")

    print(f"\n[LODO] Leave-One-Dataset-Out...")
    logo = LeaveOneGroupOut()
    lodo_records = []
    lodo_oof = np.zeros(len(X))
    for tr, te in logo.split(X, y, groups=groups):
        pred, _ = train_and_predict(X, y, tr, te)
        lodo_oof[te] = pred
        m = metric_dict(y[te], pred, len(te))
        lodo_records.append({"dataset_id": groups[te][0], **m})
        print(f"  留 {groups[te][0]}: Q2={m['Q2']:.4f}, MAE={m['MAE']:.4f} min")
    lodo_df = pd.DataFrame(lodo_records)
    print("  [LODO 聚合]")
    for col in ["Q2", "MAE", "RMSE"]:
        med, iqr, _ = robust_aggregate(lodo_df, col)
        print(f"    {col}: {med:.4f} [{iqr:.4f}]")

    total = sum(importance_accum.values()) or 1.0
    importance_accum = {k: v / total for k, v in importance_accum.items()}
    top_feats = sorted(importance_accum.items(), key=lambda x: x[1], reverse=True)[:N_TOP_FEATURES]

    lodo_df.to_csv(BASE / "global_results.csv", index=False)
    pd.DataFrame({"dataset_id": groups, "y_true": y, "y_pred": lodo_oof}) \
        .to_csv(BASE / "global_oof.csv", index=False)
    pd.DataFrame(top_feats, columns=["feature", "gain"]) \
        .to_csv(BASE / "global_feature_importance.csv", index=False)
    print(f"\n[已保存] -> {BASE.resolve()}")
    print(f"  global_results.csv ({len(lodo_df)} datasets)")
    print(f"  global_oof.csv ({len(y)} rows)")
    print(f"  global_feature_importance.csv ({len(top_feats)} features)")

    print("\n========== Global Model Summary ==========")
    print(f"Top-{TOP_N}, 样本={len(X)} (dedup={DEDUP}), 特征={X.shape[1]}")
    print(f"\nTop-5 特征:")
    for name, gain in top_feats[:5]:
        print(f"  {name}: {gain:.4f}")


if __name__ == "__main__":
    main()