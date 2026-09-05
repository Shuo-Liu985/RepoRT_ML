import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import xgboost as xgb
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

REPORT_DIR = r"D:\GitHub\RepoRT\processed_data"
DATASET_IDS = ["0260", "0261"]

DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors",
    "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings",
]

NUMERIC_SYSTEM_FEATURES = [
    "column.length", "column.id", "column.particle.size",
    "column.temperature", "column.flowrate",
    "gradient.start.A", "gradient.end.A",
    "eluent.A.acn", "eluent.A.meoh", "eluent.A.h2o",
    "eluent.B.acn", "eluent.B.meoh", "eluent.B.h2o",
]

CATEGORICAL_SYSTEM_FEATURES = ["column.usp.code"]


def load_dataset(report_dir, ds_id):
    """加载单个数据集，返回 (rt_df, meta_series)"""
    ds_path = os.path.join(report_dir, ds_id)
    if not os.path.isdir(ds_path):
        return None, None

    # 找 rtdata 文件
    rt_file = None
    for fname in os.listdir(ds_path):
        if fname.endswith("_rtdata_canonical_success.tsv"):
            rt_file = os.path.join(ds_path, fname)
            break
    meta_file = os.path.join(ds_path, ds_id + "_metadata.tsv")

    if rt_file is None or not os.path.exists(meta_file):
        return None, None

    # 健壮读取：跳过异常行，强制字符串
    rt_df = pd.read_csv(
        rt_file, sep='\t',
        dtype=str,                  # 先全部读成字符串，避免类型推断出错
        on_bad_lines='skip'         # 跳过解析错误的行
    )
    # 去重列名
    rt_df = rt_df.loc[:, ~rt_df.columns.duplicated()]
    # 强制重置索引
    rt_df = rt_df.reset_index(drop=True)

    # 清理关键列
    rt_df['rt'] = pd.to_numeric(rt_df['rt'], errors='coerce')
    rt_df = rt_df.dropna(subset=['rt', 'smiles.std'])

    # 读取元数据（单行）
    meta_df = pd.read_csv(meta_file, sep='\t', dtype=str)
    meta_series = meta_df.iloc[0]
    meta_series = meta_series[~meta_series.index.duplicated()]

    print(f"loaded {ds_id}: {len(rt_df)} compounds")
    return rt_df, meta_series


def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
        ]
    except Exception:
        return None


def main():
    # ---------- Step 1: 加载数据 ----------
    print("step 1: load data")
    all_rows = []          # 每行一个化合物
    for ds_id in DATASET_IDS:
        rt_df, meta = load_dataset(REPORT_DIR, ds_id)
        if rt_df is None:
            continue
        for _, row in rt_df.iterrows():
            desc = compute_descriptors(row['smiles.std'])
            if desc is None:
                continue
            # 分子描述符
            record = dict(zip(DESCRIPTOR_NAMES, desc))
            # 数值系统特征
            for col in NUMERIC_SYSTEM_FEATURES:
                val = meta.get(col, 0)
                record[col] = float(val) if val not in (None, '') else 0.0
            # 类别系统特征
            for col in CATEGORICAL_SYSTEM_FEATURES:
                val = meta.get(col, "unknown")
                record[col] = str(val) if val not in (None, '') else "unknown"
            # 标签与分组
            record['rt'] = float(row['rt'])
            record['dataset_id'] = ds_id
            all_rows.append(record)

    df = pd.DataFrame(all_rows)
    print(f"total valid samples: {len(df)}")

    # ---------- Step 2: 构建特征矩阵 ----------
    print("step 2: build features")
    feature_cols = DESCRIPTOR_NAMES + NUMERIC_SYSTEM_FEATURES + CATEGORICAL_SYSTEM_FEATURES
    X = df[feature_cols].copy()
    y = df['rt']
    groups = df['dataset_id']

    # 对类别特征做独热编码
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_feat = ohe.fit_transform(X[CATEGORICAL_SYSTEM_FEATURES])
    cat_names = ohe.get_feature_names_out(CATEGORICAL_SYSTEM_FEATURES)

    X_numeric = X[DESCRIPTOR_NAMES + NUMERIC_SYSTEM_FEATURES].reset_index(drop=True)
    X_cat = pd.DataFrame(cat_feat, columns=cat_names)
    X_final = pd.concat([X_numeric, X_cat], axis=1)

    print(f"total features: {X_final.shape[1]}")

    # ---------- Step 3: 按数据集分组划分（防止数据泄漏）----------
    print("step 3: grouped split")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_final, y, groups))
    X_train, X_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"train: {len(X_train)}, test: {len(X_test)}")

    # ---------- Step 4: 训练 ----------
    print("step 4: train xgboost")
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, eval_metric='rmse'
    )
    model.fit(X_train, y_train)
    print("training done")

    # ---------- Step 5: 评估 ----------
    print("step 5: evaluate")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # 特征重要性
    importance = model.get_booster().get_score(importance_type='weight')
    imp_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values('Importance', ascending=False)

    print("\ntop 15 feature importance:")
    for _, row in imp_df.head(15).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.1f}")

    # 散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual RT (min)')
    plt.ylabel('Predicted RT (min)')
    plt.title(f'XGBoost R2={r2:.3f} RMSE={rmse:.3f}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('rt_prediction_with_system.png', dpi=150)
    plt.show()

    print("done")


if __name__ == "__main__":
    main()
