import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).parent / "output"
plt.rcParams.update({"font.size": 12, "figure.dpi": 300})


def main():
    res_path = BASE / "experiment_results.csv"
    oof_path = BASE / "experiment_oof.csv"

    if not res_path.exists():
        raise FileNotFoundError("请先运行 01_generate_csv.py")

    results = pd.read_csv(res_path)
    oof = pd.read_csv(oof_path)

    # 图1：性能指标分组柱状图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["Q2", "MAE", "RMSE"]
    for ax, m in zip(axes, metrics):
        sns.barplot(x="model", y=m, hue="feature_space", data=results, ax=ax)
        ax.set_title(f"{m} Comparison")
        ax.set_ylim(min(0, results[m].min() - 0.1), results[m].max() + 0.1)
    plt.tight_layout()
    fig.savefig(BASE / "fig1_metrics_comparison.png", dpi=300)
    print("[Saved] fig1_metrics_comparison.png")

    # 图2：OOF 预测 vs 真实散点图 (以最优组合为例)
    best_row = results.loc[results["Q2"].idxmax()]
    print(f"Best combo: {best_row['model']} - {best_row['feature_space']}")

    best_oof = oof[(oof["model"] == best_row["model"]) & (oof["feature_space"] == best_row["feature_space"])]
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.scatter(best_oof["y_true"], best_oof["y_pred"], alpha=0.5, s=10)

    # 修正：只对数值列计算范围
    y_true = best_oof["y_true"]
    y_pred = best_oof["y_pred"]
    all_vals = pd.concat([y_true, y_pred])
    lo = max(0, all_vals.min())          # 保留时间一般 ≥ 0
    hi = all_vals.max()
    lims = [lo, hi]

    ax2.plot(lims, lims, "k--", lw=1)
    ax2.set_xlabel("True RT (min)")
    ax2.set_ylabel("Predicted RT (min)")
    ax2.set_title(f"OOF Predictions ({best_row['model']} | {best_row['feature_space']})")
    ax2.set_aspect("equal")
    plt.tight_layout()
    fig2.savefig(BASE / "fig2_best_scatter.png", dpi=300)
    print("[Saved] fig2_best_scatter.png")

    # 图3：特征空间效果对比 (Q2)
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    sns.boxplot(x="feature_space", y="Q2", data=results, ax=ax3)
    ax3.set_title("Feature Space Impact on Q2")
    plt.tight_layout()
    fig3.savefig(BASE / "fig3_feature_space_impact.png", dpi=300)
    print("[Saved] fig3_feature_space_impact.png")


if __name__ == "__main__":
    main()