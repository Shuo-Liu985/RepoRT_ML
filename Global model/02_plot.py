"""
02_plot.py - 可视化 Global Model 结果
★ 读取 output/ 下的 3 个 CSV 并生成图表
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------- 配置 ----------
BASE = Path(__file__).parent / "output"
WINSORIZE_Q2_THRESHOLD = -1.0          # 剔除 Q² ≤ 此阈值的折
PLOT_RT_LIMIT = None                   # 设为 20 则强制 x/y 轴范围 [0,20]，None 则自动
os.makedirs(BASE, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 200})

def main():
    # 1. 读取数据
    lodo = pd.read_csv(BASE / "global_results.csv")
    oof = pd.read_csv(BASE / "global_oof.csv")
    imp = pd.read_csv(BASE / "global_feature_importance.csv").sort_values("gain", ascending=True)

    # 2. 打印完整 LODO 表
    print("=== LODO Results (all datasets) ===")
    print(lodo.round(4).to_string(index=False))

    # 3. Winsorize 过滤（用于聚合统计和后续绘图）
    mask_keep = lodo["Q2"] > WINSORIZE_Q2_THRESHOLD
    n_excluded = (~mask_keep).sum()
    lodo_filt = lodo[mask_keep].copy()
    print(f"\nWinsorize: excluded {n_excluded} datasets with Q² <= {WINSORIZE_Q2_THRESHOLD}")

    # 4. 聚合统计（中位数 ± IQR）
    stats = {}
    for col in ["Q2", "MAE", "RMSE"]:
        vals = lodo_filt[col].dropna()
        med = np.nanmedian(vals)
        q25, q75 = np.nanquantile(vals, [0.25, 0.75])
        stats[col] = (med, q75 - q25)
        print(f"{col}: median={med:.4f}, IQR={stats[col][1]:.4f}")

    # ---------- 图1: OOF 散点图 ----------
    fig1, ax1 = plt.subplots(figsize=(6.5, 6.5))
    ax1.scatter(oof["y_true"], oof["y_pred"], alpha=0.25, s=3, c="#2196F3", edgecolors="none")
    # 对角线
    lo, hi = oof["y_true"].min(), oof["y_true"].max()
    if PLOT_RT_LIMIT:
        lo, hi = 0, PLOT_RT_LIMIT
    ax1.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_xlabel("True RT (min)")
    ax1.set_ylabel("Predicted RT (min)")
    ax1.set_title("LODO Out-of-Fold Predictions")
    ax1.set_aspect("equal")
    plt.tight_layout()
    fig1.savefig(BASE / "01_lodo_scatter.png", dpi=300)
    print("\n[Saved] 01_lodo_scatter.png")

    # ---------- 图2: 按 dataset 分色散点图 ----------
    unique_ds = oof["dataset_id"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ds)))
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for ds, color in zip(unique_ds, colors):
        sub = oof[oof["dataset_id"] == ds]
        ax2.scatter(sub["y_true"], sub["y_pred"], alpha=0.4, s=4, label=ds, color=color, edgecolors="none")
    ax2.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_xlabel("True RT (min)")
    ax2.set_ylabel("Predicted RT (min)")
    ax2.set_title("LODO Predictions (colored by dataset)")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, markerscale=2)
    plt.tight_layout()
    fig2.savefig(BASE / "02_lodo_by_dataset.png", dpi=150, bbox_inches="tight")
    print("[Saved] 02_lodo_by_dataset.png")

    # ---------- 图3: Q² 柱状图（含剔除标记） ----------
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    # 按 Q² 排序
    lodo_sorted = lodo.sort_values("Q2")
    ds_names = lodo_sorted["dataset_id"].astype(str)
    q2_vals = lodo_sorted["Q2"]
    colors_bar = ["#e74c3c" if v <= WINSORIZE_Q2_THRESHOLD else "#3498db" for v in q2_vals]
    bars = ax3.bar(range(len(q2_vals)), q2_vals, color=colors_bar, width=0.7)
    ax3.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_xticks(range(len(ds_names)))
    ax3.set_xticklabels(ds_names, rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Q²")
    ax3.set_title("LODO Q² per Dataset (red = excluded by Winsorize)")
    # 标注中位数线
    med_q2 = stats["Q2"][0]
    ax3.axhline(y=med_q2, color="green", linestyle=":", linewidth=1.2, label=f"Median Q² = {med_q2:.3f}")
    ax3.legend(fontsize=9)
    plt.tight_layout()
    fig3.savefig(BASE / "03_q2_barchart.png", dpi=150)
    print("[Saved] 03_q2_barchart.png")

    # ---------- 图4: 特征重要性（Top-15） ----------
    imp_top = imp.tail(15)
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    ax4.barh(range(len(imp_top)), imp_top["gain"], color="#4CAF50")
    ax4.set_yticks(range(len(imp_top)))
    ax4.set_yticklabels(imp_top["feature"], fontsize=8)
    ax4.set_xlabel("Gain (normalized)")
    ax4.set_title("Top-15 Feature Importance (LODO)")
    plt.tight_layout()
    fig4.savefig(BASE / "04_feature_importance.png", dpi=150, bbox_inches="tight")
    print("[Saved] 04_feature_importance.png")

    # ---------- 汇总文本 ----------
    lines = [
        "=== Global Model Summary ===",
        f"Samples: {len(oof)}",
        f"Datasets total: {len(lodo)}, after Winsorize: {len(lodo_filt)}",
        "",
        "LODO Metrics (after Winsorize, median [IQR]):",
    ]
    for col in ["Q2", "MAE", "RMSE"]:
        med, iqr = stats[col]
        lines.append(f"  {col}: {med:.4f} [{iqr:.4f}]")
    lines.append("")
    lines.append("Top-5 Features:")
    for _, row in imp_top.tail(5).iloc[::-1].iterrows():
        lines.append(f"  {row['feature']}: {row['gain']:.4f}")
    text = "\n".join(lines)
    print(text)
    with open(BASE / "summary.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n[Saved] summary.txt")

    plt.show()

if __name__ == "__main__":
    main()