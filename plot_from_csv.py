"""
根据 3 个 CSV 生成全部图表 —— 纯可视化，不训练、零等待（最终版）
================================================================
依赖文件（已生成）:
  experiment_A_results.csv        : dataset_id, source, n, Q2, MAE, RMSE
  experiment_A_oof.csv            : dataset_id, y_true, y_pred
  experiment_A_feature_importance.csv : feature, gain   (真实分子特征)

用法: 放在上述 CSV 同目录下运行
  python plot_all.py

产出: figures_experiment_a/
  01_parity_plot.png
  02_residual_vs_predicted.png
  03_residual_histogram.png
  04_williams_plot.png
  05_feature_importance.png     <- 真实化学描述符名 (RDKit_MolLogP 等)
  06_per_dataset_metrics.png
  06b_metrics_boxplot.png
  summary.txt
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

np.random.seed(42)

# ================== 配置 ==================
RESULTS_CSV = Path("experiment_A_results.csv")
OOF_CSV = Path("experiment_A_oof.csv")
FEAT_CSV = Path("experiment_A_feature_importance.csv")
OUT_DIR = Path("figures_experiment_a")
N_TOP = 15
# ==========================================


def _safe(s):
    """去掉 Windows GBK 无法编码的特殊字符"""
    return (s.replace("\u00b2", "2").replace("\u00b1", "+/-")
             .replace("\u00d7", "x").replace("\u2192", "->")
             .replace("\u2264", "<=").replace("\u2265", ">="))


def _hline(ax, y=0.0):
    ax.axhline(y=y, color="black", linewidth=1.0)


def _vline(ax, x=0.0):
    ax.axvline(x=x, color="red", linewidth=1.5)


def load_data():
    for p in (RESULTS_CSV, OOF_CSV):
        if not p.exists():
            raise FileNotFoundError(f"找不到 {p}，请先运行表格计算脚本")
    results = pd.read_csv(RESULTS_CSV)
    oof = pd.read_csv(OOF_CSV)
    results["dataset_id"] = results["dataset_id"].astype(str)
    oof["dataset_id"] = oof["dataset_id"].astype(str)
    for col in ["Q2", "MAE", "RMSE"]:
        if col not in results.columns:
            raise ValueError(f"results CSV 缺少列: {col}")
    for col in ["y_true", "y_pred"]:
        if col not in oof.columns:
            raise ValueError(f"oof CSV 缺少列: {col}")
    print(f"[OK] {RESULTS_CSV}: {results.shape}  ({results['dataset_id'].nunique()} datasets)")
    print(f"[OK] {OOF_CSV}: {oof.shape}")
    if FEAT_CSV.exists():
        print(f"[OK] {FEAT_CSV}: 存在 -> 将绘制真实分子特征重要性图")
    else:
        print(f"[WARN] {FEAT_CSV}: 不存在 -> 图5降级为 per-dataset Q2 proxy")
    return results, oof


# ============ 1. Parity Plot ============
def plot_parity(oof, out):
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, (ds, sub) in enumerate(oof.groupby("dataset_id")):
        ax.scatter(sub["y_true"], sub["y_pred"], s=12, alpha=0.5,
                   color=colors[i % len(colors)], label=str(ds), edgecolors="none")
    lims = [oof[["y_true", "y_pred"]].min().min(), oof[["y_true", "y_pred"]].max().max()]
    ax.plot(lims, lims, linestyle="--", color="red", linewidth=1.5, label="Identity")
    r2 = r2_score(oof["y_true"], oof["y_pred"])
    ax.text(0.05, 0.95, "Overall R2 = {:.4f}".format(r2), transform=ax.transAxes,
            fontsize=11, va="top", bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    ax.set_xlabel("Experimental RT (min)")
    ax.set_ylabel("Predicted RT (min)")
    ax.set_title("Parity Plot - Experiment A (10-fold OOF, per-dataset)")
    ax.legend(ncol=2, fontsize=6, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "01_parity_plot.png", dpi=150)
    plt.close(fig)


# ============ 2&3. Residuals ============
def plot_residuals(oof, out):
    d = oof.copy()
    d["residual"] = d["y_pred"] - d["y_true"]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, (ds, sub) in enumerate(d.groupby("dataset_id")):
        ax.scatter(sub["y_pred"], sub["residual"], s=12, alpha=0.5,
                   color=colors[i % len(colors)], label=str(ds), edgecolors="none")
    _hline(ax, 0.0)
    ax.set_xlabel("Predicted RT (min)")
    ax.set_ylabel("Residual (Pred - True, min)")
    ax.set_title("Residual vs Predicted - systematic bias diagnosis")
    ax.legend(ncol=2, fontsize=6, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "02_residual_vs_predicted.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(d["residual"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    _vline(ax, 0.0)
    ax.set_xlabel("Residual (min)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution (OOF errors)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "03_residual_histogram.png", dpi=150)
    plt.close(fig)


# ============ 4. Williams Plot ============
def plot_williams(oof, out):
    residual = oof["y_pred"].to_numpy() - oof["y_true"].to_numpy()
    sigma = np.sqrt(np.mean(residual ** 2))
    std_resid = residual / (sigma + 1e-9)
    n = len(residual)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(np.zeros(n), std_resid, s=15, alpha=0.5, color="#C44E52", edgecolors="none")
    _hline(ax, 3.0)
    _hline(ax, -3.0)
    n_out = int(np.sum(np.abs(std_resid) > 3))
    pct = 100.0 * n_out / n
    ax.text(0.05, 0.95, "Outside +/-3sigma: {} ({:.1f}%)".format(n_out, pct),
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("Index (strict AD requires PLS hat matrix for leverage)")
    ax.set_ylabel("Standardized Residual")
    ax.set_title("Williams Plot - Applicability Domain (OECD): +/-3sigma threshold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "04_williams_plot.png", dpi=150)
    plt.close(fig)


# ============ 5. Feature Importance ============
def plot_feature_importance(out):
    """优先真实分子特征 (feature, gain)，否则 per-dataset Q2 proxy"""
    if FEAT_CSV.exists():
        feat = pd.read_csv(FEAT_CSV)
        if "feature" in feat.columns and "gain" in feat.columns:
            top = feat.sort_values("gain", ascending=False).head(N_TOP)
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.barh(range(len(top))[::-1], top["gain"].values[::-1], color="#8172B3")
            ax.set_yticks(range(len(top))[::-1])
            ax.set_yticklabels(top["feature"].values[::-1], fontsize=9)
            ax.set_xlabel("Aggregated Gain Importance (sum over datasets)")
            ax.set_title("Top {} Feature Importance - Experiment A".format(N_TOP))
            fig.tight_layout()
            fig.savefig(out / "05_feature_importance.png", dpi=150)
            plt.close(fig)
            print("[OK] 05: 真实分子特征 Top-{} (来自 {})".format(len(top), FEAT_CSV))
            print("  Top5:", ", ".join(f"{n} ({v:.3f})" for n, v in
                                        zip(top["feature"].head().values, top["gain"].head().values)))
            return
    # proxy
    results = pd.read_csv(RESULTS_CSV)
    df = results.sort_values("Q2").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(df)), df["Q2"], color=plt.cm.viridis(np.linspace(0.2, 0.9, len(df))))
    _hline(ax, np.mean(df["Q2"]))
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(["{}".format(o) for o in df["dataset_id"]], fontsize=8)
    ax.set_xlabel("Q2 (per-dataset fit)")
    ax.set_title("Per-dataset Fit (Q2) - proxy (no feature gain CSV)")
    fig.tight_layout()
    fig.savefig(out / "05_feature_importance.png", dpi=150)
    plt.close(fig)
    print("[INFO] 05: proxy 图 (per-dataset Q2)")


# ============ 6 & 6b. Per-dataset Metrics ============
def plot_per_dataset_summary(results, out):
    df = results.sort_values("Q2").reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, title, color) in zip(axes, [
        ("Q2", "Q2 by dataset", "#55A868"),
        ("MAE", "MAE (min) by dataset", "#4C72B0"),
        ("RMSE", "RMSE (min) by dataset", "#C44E52"),
    ]):
        ax.bar(range(len(df)), df[col], color=color)
        ax.axhline(df[col].median(), color="red", linestyle="--",
                   label="median={:.3f}".format(df[col].median()))
        ax.set_xlabel("Dataset (sorted by Q2)")
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "06_per_dataset_metrics.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([results["Q2"], results["MAE"], results["RMSE"]])
    ax.set_xticklabels(["Q2", "MAE (min)", "RMSE (min)"])
    ax.set_title("Distribution across datasets (Experiment A)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "06b_metrics_boxplot.png", dpi=150)
    plt.close(fig)


# ============ Summary ============
def save_summary(results, oof, out):
    L = []
    L.append("=" * 50)
    L.append("Experiment A - Summary")
    L.append("=" * 50)
    L.append("Datasets: {}".format(len(results)))
    L.append("Total OOF samples: {}".format(len(oof)))
    L.append("")
    for col in ["Q2", "MAE", "RMSE"]:
        q1, med, q3 = results[col].quantile([0.25, 0.5, 0.75])
        L.append("{0}: mean={1:.4f}, median={2:.4f}, IQR=[{3:.4f}, {4:.4f}]".format(
            col, results[col].mean(), med, q1, q3))
    r2 = r2_score(oof["y_true"], oof["y_pred"])
    mae = mean_absolute_error(oof["y_true"], oof["y_pred"])
    rmse = np.sqrt(mean_squared_error(oof["y_true"], oof["y_pred"]))
    L.append("")
    L.append("Global (all OOF pooled):")
    L.append("  R2  = {:.4f}".format(r2))
    L.append("  MAE = {:.4f} min ({:.1f} s)".format(mae, mae * 60))
    L.append("  RMSE= {:.4f} min ({:.1f} s)".format(rmse, rmse * 60))
    L.append("")
    L.append("Per-dataset detail (sorted by Q2 desc):")
    for _, r in results.sort_values("Q2", ascending=False).iterrows():
        L.append("  {0}: n={1}, Q2={2:.4f}, MAE={3:.4f}, RMSE={4:.4f}".format(
            r["dataset_id"], int(r["n"]), r["Q2"], r["MAE"], r["RMSE"]))
    txt = _safe("\n".join(L))
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    results, oof = load_data()

    print("\n[1/6] Parity Plot ..."); plot_parity(oof, OUT_DIR)
    print("[2/6] Residual vs Predicted ..."); plot_residuals(oof, OUT_DIR)
    print("[3/6] Williams Plot ..."); plot_williams(oof, OUT_DIR)
    print("[4/6] Feature Importance ..."); plot_feature_importance(OUT_DIR)
    print("[5/6] Per-dataset Metrics ..."); plot_per_dataset_summary(results, OUT_DIR)
    print("[6/6] Summary ..."); save_summary(results, oof, OUT_DIR)

    print("\n[完成] 所有图表 -> " + str(OUT_DIR.resolve()))


if __name__ == "__main__":
    main()
