import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, AllChem
from mordred import Calculator, descriptors as mordred_descriptors

rdBase.DisableLog("rdApp.warning")
np.random.seed(42)

# ============================ 配置 ============================
REPORT_DIR = Path(r"D:\GitHub\RepoRT\processed_data\0390")  # 修改为0390具体路径
BASE = Path(__file__).parent / "output"
N_SPLITS = 10
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
RDKIT_NAMES = [name for name, _ in Descriptors._descList]

# 初始化 Mordred 计算器 (仅2D)
mordred_calc = Calculator(mordred_descriptors, ignore_3D=True)

MODELS = {
    "RF": RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                                random_state=42, n_jobs=-1, tree_method="hist", eval_metric="rmse"),
    "LightGBM": lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                                  random_state=42, n_jobs=-1, verbose=-1)
}


# ======================== 特征工程 =========================
def get_rdkit_features(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return np.zeros(len(RDKIT_NAMES), dtype=np.float32)
    rdkit = [func(mol, avg=True) if name == "Ipc" else func(mol) for name, func in Descriptors._descList]
    return np.array(rdkit, dtype=np.float32)


def get_morgan_features(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return np.zeros(MORGAN_BITS, dtype=np.float32)
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS)
    return (np.frombuffer(bitvect.ToBitString().encode("ascii"), dtype=np.uint8) - 48).astype(np.float32)


def build_features(df):
    df = df.dropna(subset=["rt", "smiles.std"]).copy()
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    df = df.dropna(subset=["rt"]).reset_index(drop=True)

    mols = [Chem.MolFromSmiles(s) for s in df["smiles.std"].tolist()]

    # 1. 基础特征 (RDKit)
    feats_rdkit = np.vstack([get_rdkit_features(s) for s in df["smiles.std"].tolist()])
    X_basic = pd.DataFrame(feats_rdkit, columns=[f"RDKit_{n}" for n in RDKIT_NAMES])

    # 2. 扩展特征 (RDKit + Mordred + Morgan)
    feats_mordred = mordred_calc.pandas(mols, nproc=1)
    feats_mordred = feats_mordred.apply(pd.to_numeric, errors='coerce')

    feats_morgan = np.vstack([get_morgan_features(s) for s in df["smiles.std"].tolist()])
    X_morgan = pd.DataFrame(feats_morgan, columns=[f"Morgan_{i}" for i in range(MORGAN_BITS)])

    X_extended = pd.concat([X_basic, feats_mordred.add_prefix("Mordred_"), X_morgan], axis=1)

    # 清理无效特征
    for df_x in [X_basic, X_extended]:
        df_x.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_x.fillna(0.0, inplace=True)
        df_x.dropna(axis=1, how="all", inplace=True)
        df_x = df_x.loc[:, lambda d: d.var() > 0]

    return X_basic, X_extended, df["rt"].to_numpy()


# =================== 10 折 OOF CV ===================
def run_cv(model, X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof = np.zeros(len(y))

    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X.iloc[tr])
        Xtr = pd.DataFrame(sc.transform(X.iloc[tr]), columns=X.columns)
        Xte = pd.DataFrame(sc.transform(X.iloc[te]), columns=X.columns)

        model_clone = MODELS[model]  # 重新实例化以获取干净模型
        model_clone.fit(Xtr, y[tr])
        oof[te] = model_clone.predict(Xte)

    metrics = {
        "Q2": r2_score(y, oof),
        "MAE": mean_absolute_error(y, oof),
        "RMSE": np.sqrt(mean_squared_error(y, oof))
    }
    return metrics, oof


# ============================ Main ==========================
def main():
    BASE.mkdir(exist_ok=True)
    rt_path = REPORT_DIR / "0390_rtdata_isomeric_success.tsv"
    if not rt_path.exists():
        rt_path = REPORT_DIR / "0390_rtdata_canonical_success.tsv"

    df_raw = pd.read_csv(rt_path, sep="\t")
    print(f"Loaded 0390 dataset: {len(df_raw)} records")

    X_basic, X_extended, y = build_features(df_raw)
    print(f"Features built. Basic: {X_basic.shape[1]}, Extended: {X_extended.shape[1]}")

    results = []
    oof_list = []

    for feat_type, X in [("Basic", X_basic), ("Extended", X_extended)]:
        for name in MODELS.keys():
            print(f"Training {name} on {feat_type} features...")
            metrics, oof_preds = run_cv(name, X, y)
            res = {"dataset": "0390", "feature_space": feat_type, "model": name, **metrics}
            results.append(res)
            oof_list.append(pd.DataFrame({
                "dataset": "0390", "model": name, "feature_space": feat_type,
                "y_true": y, "y_pred": oof_preds
            }))
            print(f"  -> Q2: {metrics['Q2']:.4f}, MAE: {metrics['MAE']:.4f}")

    # 保存 CSV
    results_df = pd.DataFrame(results)
    oof_df = pd.concat(oof_list, ignore_index=True)

    results_df.to_csv(BASE / "experiment_results.csv", index=False)
    oof_df.to_csv(BASE / "experiment_oof.csv", index=False)

    # 敏感性分析汇总
    med_q2_all = np.nanmedian(results_df["Q2"])
    med_q2_filt = np.nanmedian(results_df[results_df["Q2"] > -1.0]["Q2"])
    print(f"\n[Summary] Median Q2 (All): {med_q2_all:.4f}")
    print(f"[Summary] Median Q2 (Winsorize Q2>-1): {med_q2_filt:.4f}")
    print(f"[Saved] CSV files to {BASE.resolve()}")


if __name__ == "__main__":
    main()