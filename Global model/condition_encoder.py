import pandas as pd
import numpy as np
from pathlib import Path

# ================= 配置区 =================
# 根据你的代码上下文保留这些配置
NUMERIC_COND = ["column.length", "column.innerdiam", "column.particle.size", 
                "column.temperature", "flowrate", "gradient.duration"]
CATEGORIC_COND = ["column.name", "column.usp.code"]
SOLVENTS = ["h2o", "acn", "meoh"]
ADDITIVES = ["formic", "acetic", "trifluoroacetic", "nh4ac", "nh4form"]
PH_COL = "pH"
MIN_CAT_FREQ = 5

def load_dataset_metadata(folder, dataset_id):
    """
    按 RepoRT 官方单行宽表格式读取元数据
    """
    path = Path(folder) / f"{dataset_id}_metadata.tsv"
    if not path.exists():
        print(f"  [WARN] 元数据文件不存在: {path}")
        return None
    
    try:
        # RepoRT 元数据是单行宽表，列名即为参数名
        df = pd.read_csv(path, sep="\t", encoding="utf-8")
        
        # 将第一行数据转换为字典 (列名: 值)
        meta = {}
        for col in df.columns:
            val = df[col].iloc[0]
            # 处理可能的空值
            if pd.isna(val):
                meta[col] = None
            else:
                meta[col] = val
                
        return meta
    except Exception as e:
        print(f"  [ERROR] 读取元数据失败 {path}: {e}")
        return None

# ================= 下面是原有的辅助函数和特征构建逻辑 =================
def parse_pct(val):
    if val is None or pd.isna(val):
        return np.nan
    try:
        return float(re.sub(r"[^\d.eE\-+]", "", str(val)))
    except:
        return np.nan

def build_condition_features(df, dataset_id, metadata=None, all_columns=None):
    n = len(df)
    cols = {}

    # 数值型
    for key in NUMERIC_COND:
        val = parse_pct(metadata.get(key, np.nan)) if metadata else np.nan
        cols[key] = [val] * n

    # 衍生特征：梯度斜率 & 柱体积
    start_b = parse_pct(metadata.get("gradient.start.B", np.nan))
    end_b   = parse_pct(metadata.get("gradient.end.B", np.nan))
    dur     = parse_pct(metadata.get("gradient.duration", np.nan)) or 30.0
    slope = ((end_b - start_b) / dur) if (pd.notna(start_b) and pd.notna(end_b) and dur > 0) else 0.0
    cols["cond_gradient_slope_B"] = [slope] * n

    length = parse_pct(metadata.get("column.length", np.nan)) or 0.0
    diam   = parse_pct(metadata.get("column.innerdiam", np.nan)) or 2.1
    vol = np.pi * (diam/2)**2 * length * 1e-3
    cols["cond_column_volume"] = [vol] * n

    # 溶剂/添加剂 binary (简化版)
    for sol in SOLVENTS:
        for add in ADDITIVES:
            key = f"cond_solvent_{sol}_{add}"
            val = 0
            if metadata:
                eluent_a = str(metadata.get("eluent.A.composition", "")).lower()
                eluent_b = str(metadata.get("eluent.B.composition", "")).lower()
                adds = str(metadata.get("eluent.additives", "")).lower()
                if sol in eluent_a or sol in eluent_b:
                    if add in adds:
                        val = 1
            cols[key] = [val] * n

    # pH
    ph_val = parse_pct(metadata.get(PH_COL, np.nan)) if metadata else np.nan
    cols[PH_COL] = [ph_val] * n

    # 分类变量 one-hot
    for c in CATEGORIC_COND:
        raw_val = str(metadata.get(c, "")) if metadata else ""
        if raw_val:
            col_name = f"{c}__{raw_val}"
            if all_columns and col_name in all_columns:
                cols[col_name] = [1] * n

    # 对齐输出
    X = pd.DataFrame(cols)
    if all_columns:
        for col in all_columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X[all_columns]
    return X

def get_condition_column_names(meta_rows):
    """生成条件特征列名 (保持与原逻辑兼容)"""
    cols = list(NUMERIC_COND) + ["cond_gradient_slope_B", "cond_column_volume", PH_COL]
    # 动态添加分类变量
    cat_vals = set()
    for meta in meta_rows:
        if meta is None: continue
        for c in CATEGORIC_COND:
            val = meta.get(c, "")
            if val: cat_vals.add(f"{c}__{val}")
            
    # 频次过滤 (简化)
    from collections import Counter
    cnt = Counter()
    for meta in meta_rows:
        if meta is None: continue
        for c in CATEGORIC_COND:
            val = meta.get(c, "")
            if val: cnt[f"{c}__{val}"] += 1
            
    for key, freq in cnt.items():
        if freq >= MIN_CAT_FREQ:
            cols.append(key)
            
    # 溶剂组合
    for sol in SOLVENTS:
        for add in ADDITIVES:
            cols.append(f"cond_solvent_{sol}_{add}")
            
    return cols