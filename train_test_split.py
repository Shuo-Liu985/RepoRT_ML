import os
import pandas as pd
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator  # 引入新的指纹生成器
from mordred import Calculator, descriptors
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import warnings

# 1. 屏蔽 Python 和 RDKit 的底层警告
warnings.filterwarnings('ignore')
rdBase.DisableLog('rdApp.warning')  # 彻底屏蔽 RDKit C++ 弃用警告 [1](@ref)
np.random.seed(42)

REPORT_DIR = r"D:\GitHub\RepoRT\processed_data"

NUMERIC_SYSTEM_FEATURES = [
    "column.length", "column.id", "column.particle.size",
    "column.temperature", "column.flowrate",
    "eluent.A.acn", "eluent.A.meoh", "eluent.A.h2o",
    "eluent.B.acn", "eluent.B.meoh", "eluent.B.h2o",
    "gradient.start.A", "gradient.end.A"
]
CATEGORICAL_SYSTEM_FEATURES = ["column.usp.code"]

# 初始化 Mordred 和新版 Morgan 指纹 API
mordred_calc = Calculator(descriptors, ignore_3D=True)
MORDRED_NAMES = mordred_calc.descriptors
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)  # 新API替代旧接口


# ---------- 数据加载 ----------
def clean_columns(df):
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_data(report_dir, max_datasets=5):
    frames = []
    subdirs = sorted([d for d in os.listdir(report_dir)
                      if os.path.isdir(os.path.join(report_dir, d)) and d.isdigit()])[:max_datasets]

    for ds_id in subdirs:
        ds_path = os.path.join(report_dir, ds_id)
        rt_file = None
        for fname in os.listdir(ds_path):
            if fname.startswith(ds_id + "_rtdata_canonical_success"):
                rt_file = os.path.join(ds_path, fname)
                break
        meta_file = os.path.join(ds_path, ds_id + "_metadata.tsv")
        if rt_file is None or not os.path.exists(meta_file):
            continue

        rt_df = pd.read_csv(rt_file, sep='\t', dtype=str, on_bad_lines='skip')
        meta_df = pd.read_csv(meta_file, sep='\t', dtype=str, on_bad_lines='skip')
        rt_df = clean_columns(rt_df)
        meta_df = clean_columns(meta_df)

        meta_row = meta_df.iloc[0]
        for col in NUMERIC_SYSTEM_FEATURES + CATEGORICAL_SYSTEM_FEATURES:
            if col in meta_row:
                rt_df[col] = meta_row[col]
        rt_df['dataset_id'] = ds_id
        frames.append(rt_df)

    if not frames:
        raise ValueError("未加载到任何数据")
    return pd.concat(frames, ignore_index=True).dropna(subset=['rt', 'smiles.std'])


# ---------- 分子特征计算 ----------
def compute_features(smiles_list, batch_size=200):
    all_data = []
    rdkit_2d_names = [desc_name for desc_name, _ in Descriptors._descList]

    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]
        mols = [Chem.MolFromSmiles(s) for s in batch_smiles]

        # RDKit 2D
        rdkit_rows = []
        for mol in mols:
            if mol is None:
                rdkit_rows.append([np.nan] * len(rdkit_2d_names))
            else:
                row = []
                for desc_name, func in Descriptors._descList:
                    try:
                        row.append(func(mol, avg=True) if desc_name == 'Ipc' else func(mol))
                    except:
                        row.append(np.nan)
                rdkit_rows.append(row)
        rdkit_df = pd.DataFrame(rdkit_rows, columns=[f"RDKit_{n}" for n in rdkit_2d_names])

        # Mordred
        mordred_rows = []
        for mol in mols:
            if mol is None:
                mordred_rows.append([np.nan] * len(MORDRED_NAMES))
            else:
                try:
                    vals = mordred_calc(mol)
                    mordred_rows.append([float(v) if v is not None else np.nan for v in vals])
                except:
                    mordred_rows.append([np.nan] * len(MORDRED_NAMES))
        mordred_df = pd.DataFrame(mordred_rows, columns=[f"Mordred_{n}" for n in MORDRED_NAMES])

        # Morgan 指纹 (使用新 API) [1](@ref)
        morgan_rows = []
        for mol in mols:
            if mol is None:
                morgan_rows.append([0.0] * 2048)
            else:
                fp = morgan_gen.GetFingerprintAsNumPy(mol)
                morgan_rows.append(fp)
        morgan_df = pd.DataFrame(morgan_rows, columns=[f"Morgan_{i}" for i in range(2048)])

        all_data.append(pd.concat([rdkit_df, mordred_df, morgan_df], axis=1))

    feat_df = pd.concat(all_data, ignore_index=True)
    feat_df = feat_df.apply(pd.to_numeric, errors='coerce')
    feat_df = feat_df.dropna(axis=1, how='all')
    feat_df = feat_df.fillna(0.0)
    feat_df = feat_df.loc[:, feat_df.var() > 0]
    return feat_df.astype(np.float32)


# ---------- 主流程 ----------
def main():
    print("Step 1: 加载数据")
    df = load_data(REPORT_DIR, max_datasets=5)
    print(f"总样本数: {len(df)}")

    print("Step 2: 计算分子特征")
    desc_df = compute_features(df['smiles.std'].tolist(), batch_size=200)
    print(f"分子特征维度: {desc_df.shape[1]}")

    # 构建系统特征
    sys_rows = []
    for _, row in df.iterrows():
        feat = {}
        for col in NUMERIC_SYSTEM_FEATURES:
            val = row.get(col, 0)
            feat[col] = float(val) if not pd.isna(val) else 0.0
        for col in CATEGORICAL_SYSTEM_FEATURES:
            val = row.get(col, "unknown")
            feat[col] = str(val) if not pd.isna(val) else "unknown"
        sys_rows.append(feat)
    sys_df = pd.DataFrame(sys_rows)

    # 独热编码
    cat_cols = [c for c in CATEGORICAL_SYSTEM_FEATURES if c in sys_df.columns]
    if cat_cols:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        cat_encoded = ohe.fit_transform(sys_df[cat_cols])
        cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(cat_cols))
        sys_df = pd.concat([sys_df[NUMERIC_SYSTEM_FEATURES].reset_index(drop=True), cat_df], axis=1)
    else:
        sys_df = sys_df[NUMERIC_SYSTEM_FEATURES]

    # 合并特征（保持 DataFrame 格式以保留列名）
    X = pd.concat([desc_df.reset_index(drop=True), sys_df.reset_index(drop=True)], axis=1)

    # 目标变量：对数变换缓解偏态
    y = np.log1p(df['rt'].astype(float).values)

    print("Step 3: 普通随机拆分 (修复分组不平衡)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

    print("Step 4: 特征标准化")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Step 5: 训练 XGBoost")
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
        eval_metric='rmse', tree_method='hist'
    )
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=50)

    # 逆变换回原空间
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    y_test_original = np.expm1(y_test)

    r2 = r2_score(y_test_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    print(f"\nR² = {r2:.4f}")
    print(f"RMSE = {rmse:.4f}")

    # 特征重要性（修复 f数字 映射问题）
    importance = model.get_booster().get_score(importance_type='gain')
    # 将 f0, f1 映射回原始列名
    mapped_importance = {X.columns[int(k[1:])]: v for k, v in importance.items()}
    imp_df = pd.DataFrame({
        'Feature': list(mapped_importance.keys()),
        'Importance': list(mapped_importance.values())
    }).sort_values('Importance', ascending=False)

    print("\nTop 15 特征重要性:")
    for _, row in imp_df.head(15).iterrows():
        print(f"  {row['Feature']} : {row['Importance']:.1f}")

    # 绘图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_original, y_pred, alpha=0.5, s=10, edgecolors='k')
    min_val = min(y_test_original.min(), y_pred.min())
    max_val = max(y_test_original.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual RT (min)')
    plt.ylabel('Predicted RT (min)')
    plt.title(f'XGBoost Fixed R²={r2:.3f} RMSE={rmse:.3f}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('rt_prediction_fixed.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()