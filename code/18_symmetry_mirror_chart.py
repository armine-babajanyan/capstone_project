"""XGBoost SHAP symmetry mirror-bar chart.

Produces a single publication-quality butterfly bar chart showing
mean |SHAP| for the top-10 crisis ∪ top-10 growth XGBoost features,
with Spearman ρ annotations quantifying directional symmetry.

Reads:
    results/shap/shap_values_xgb_crisis.npz
    results/shap/shap_values_xgb_growth.npz

Writes:
    results/figures/xgb_shap_symmetry_mirror.png
    results/figures/xgb_shap_symmetry_mirror.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr

from src import config

TOP_K = 10
N_BINS = 20


# ── helpers ──────────────────────────────────────────────────────────

def load_xgb_shap(label: str):
    path = config.RESULTS_DIR / "shap" / f"shap_values_xgb_{label}.npz"
    with np.load(path, allow_pickle=False) as z:
        values = z["values"]
        feature_cols = z["feature_cols"].tolist()
        X = pd.DataFrame(z["X"], columns=feature_cols)
    return values, X, feature_cols


def mean_abs_ranking(values: np.ndarray, feature_cols: list[str]) -> pd.Series:
    return (
        pd.Series(np.abs(values).mean(axis=0), index=feature_cols)
        .sort_values(ascending=False)
    )


def bin_curve(x: np.ndarray, s: np.ndarray, n_bins: int = N_BINS):
    """Quantile-bin feature values, return (centers, mean_shap).
    Adapts bin count for low-cardinality features (e.g. macro vars)."""
    finite = np.isfinite(x) & np.isfinite(s)
    x, s = x[finite], s[finite]
    n_unique = len(np.unique(x))
    if n_unique < 3:
        return None
    # Adapt bins for low-cardinality features
    actual_bins = min(n_bins, n_unique)

    quantiles = np.linspace(0, 1, actual_bins + 1)
    edges = np.unique(np.quantile(x, quantiles))
    if len(edges) < 3:
        return None

    idx = np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, len(edges) - 2)
    n_actual = len(edges) - 1

    mean_s = np.full(n_actual, np.nan)
    for b in range(n_actual):
        m = idx == b
        if m.any():
            mean_s[b] = s[m].mean()

    centers = (quantiles[:-1] + np.diff(quantiles) / 2)[:n_actual]
    valid = np.isfinite(mean_s)
    return centers[valid], mean_s[valid]


def compute_spearman(feat, sv_c, X_c, feats_c, sv_g, X_g, feats_g):
    """Spearman ρ between crisis SHAP curve and growth SHAP curve."""
    if feat not in feats_c or feat not in feats_g:
        return np.nan, np.nan

    ci = feats_c.index(feat)
    gi = feats_g.index(feat)

    cd = bin_curve(X_c[feat].to_numpy(), sv_c[:, ci])
    cg = bin_curve(X_g[feat].to_numpy(), sv_g[:, gi])
    if cd is None or cg is None:
        return np.nan, np.nan

    pct_c, ms_c = cd
    pct_g, ms_g = cg
    common = np.intersect1d(np.round(pct_c, 4), np.round(pct_g, 4))
    if len(common) < 3:
        return np.nan, np.nan

    m_c = np.isin(np.round(pct_c, 4), common)
    m_g = np.isin(np.round(pct_g, 4), common)
    rho, pval = spearmanr(ms_c[m_c], ms_g[m_g])
    return rho, pval


# ── pretty feature names ─────────────────────────────────────────────

def prettify(name: str) -> str:
    return config.pretty(name)


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    sv_c, X_c, feats_c = load_xgb_shap("crisis")
    sv_g, X_g, feats_g = load_xgb_shap("growth")

    rank_c = mean_abs_ranking(sv_c, feats_c)
    rank_g = mean_abs_ranking(sv_g, feats_g)

    # Top-5 union (preserving order by total importance)
    top_union = list(dict.fromkeys(
        list(rank_c.head(TOP_K).index) + list(rank_g.head(TOP_K).index)
    ))

    # Build data for chart
    rows = []
    for feat in top_union:
        crisis_val  = float(rank_c[feat]) if feat in rank_c.index else 0.0
        growth_val  = float(rank_g[feat]) if feat in rank_g.index else 0.0
        rho, pval   = compute_spearman(feat, sv_c, X_c, feats_c, sv_g, X_g, feats_g)
        rows.append({
            "feature":    feat,
            "pretty":     prettify(feat),
            "crisis":     crisis_val,
            "growth":     growth_val,
            "total":      crisis_val + growth_val,
            "rho":        rho,
            "rho_pval":   pval,
        })

    df = pd.DataFrame(rows).sort_values("total", ascending=True).reset_index(drop=True)
    print(df[["feature", "crisis", "growth", "rho"]].to_string(index=False))

    # ── plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    y = np.arange(len(df))
    bar_h = 0.62

    # Crisis bars (extend LEFT = negative direction)
    ax.barh(y, -df["crisis"], height=bar_h, color="#e63946", label="Crisis",
            edgecolor="white", linewidth=0.4, zorder=3)
    # Growth bars (extend RIGHT)
    ax.barh(y, df["growth"], height=bar_h, color="#5DB453", label="Growth",
            edgecolor="white", linewidth=0.4, zorder=3)

    # Feature labels in the centre
    ax.set_yticks(y)
    ax.set_yticklabels(df["pretty"], fontsize=9, fontweight="medium")

    # Spearman ρ annotations at the end of the longer bar
    for i, row in df.iterrows():
        x_pos = row["growth"] + 0.012
        if np.isfinite(row["rho"]):
            rho_str = f"ρ = {row['rho']:+.2f}"
            color = "#5DB453" if row["rho"] > 0.3 else ("#e63946" if row["rho"] < -0.3 else "#666666")
            ax.text(x_pos, i, rho_str, va="center", ha="left",
                    fontsize=8.5, color=color, fontstyle="italic")
        else:
            # Feature only in one model or insufficient variation
            shared = (row["crisis"] > 0) and (row["growth"] > 0)
            note = "ρ: n/a" if not shared else "ρ: n/a (low var.)"
            ax.text(x_pos, i, note, va="center", ha="left",
                    fontsize=8.5, color="#999999", fontstyle="italic")

    # Symmetric x-axis
    x_max = max(df["crisis"].max(), df["growth"].max()) * 1.45
    ax.set_xlim(-x_max, x_max)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{abs(v):.2f}"))
    ax.set_xlabel("Mean |SHAP value|", fontsize=11, labelpad=8)

    # Centre spine
    ax.axvline(0, color="#333333", linewidth=0.8, zorder=4)

    # Grid & spines
    ax.grid(axis="x", alpha=0.15, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Column headers — stacked above the plot area
    ax.text(-x_max * 0.50, len(df) + 0.35, "Crisis",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#e63946")
    ax.text( x_max * 0.50, len(df) + 0.35, "Growth",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#5DB453")

    # Footnote explaining ρ
    fig.text(0.5, -0.02,
             "ρ = Spearman correlation between binned SHAP(crisis) and SHAP(growth)",
             ha="center", va="top", fontsize=8.5, color="#555555", fontstyle="italic")

    fig.tight_layout()

    out_dir = config.ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "xgb_shap_symmetry_mirror.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "xgb_shap_symmetry_mirror.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_dir / 'xgb_shap_symmetry_mirror.png'}")
    print(f"Wrote {out_dir / 'xgb_shap_symmetry_mirror.pdf'}")


if __name__ == "__main__":
    main()
