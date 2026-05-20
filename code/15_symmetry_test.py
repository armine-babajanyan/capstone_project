"""Symmetry test — deterioration vs growth SHAP drivers.

Per Model_Plan.md sec 10 item 3:
  1. Load cached SHAP values from 14_shap_analysis.py for both labels.
  2. Rank features by mean |SHAP| and align the top-15 features appearing
     in either label.
  3. For each shared feature, bin observations by percentile of the
     feature value, take the mean SHAP per bin, and compare the
     deterioration curve against the *negated* growth curve.
     A perfect mirror would make the two curves coincide.
  4. Quantify similarity with the Spearman rank correlation between the
     deterioration curve and the negated growth curve across the bins.

A high positive Spearman corr => the feature drives deterioration and
growth symmetrically (low capital ratio predicts deterioration <=> high
capital ratio predicts growth).
A correlation near zero or negative => asymmetric driver, worth
highlighting in the paper.

Writes:
    results/metrics/symmetry_spearman.csv
    results/figures/symmetry_curves.png                  (grid of all shared features)
    results/figures/symmetry_<feature>.png               (one panel per shared feature)
    results/figures/symmetry_spearman_bar.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src import config


TOP_K = 15           # top-K by mean |SHAP| per label; union goes into the test
N_BINS = 20          # number of percentile bins for the SHAP-vs-feature curves
MIN_UNIQUE = N_BINS  # skip features with fewer unique values than bins
MODEL_FAMILY = "xgb" # model to use (xgb, lgbm, or rf)


# ---------- load cached SHAP ----------

def load_shap(family: str, label: str):
    path = config.RESULTS_DIR / "shap" / f"shap_values_{family}_{label}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run 14_shap_analysis.py first."
        )
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


# ---------- binning ----------

def bin_curve(x: np.ndarray, s: np.ndarray, n_bins: int = N_BINS):
    """Return (bin_center_percentiles, mean_feature_val, mean_shap) with NaNs
    dropped. Uses quantile bins so each bin has ~equal count; bins with
    identical edges (e.g. sparse integer features) are collapsed.
    """
    finite = np.isfinite(x) & np.isfinite(s)
    x, s = x[finite], s[finite]
    if len(np.unique(x)) < n_bins:
        return None

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, quantiles))
    if len(edges) < 3:
        return None

    idx = np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, len(edges) - 2)
    n_actual = len(edges) - 1

    mean_x = np.full(n_actual, np.nan)
    mean_s = np.full(n_actual, np.nan)
    for b in range(n_actual):
        m = idx == b
        if m.any():
            mean_x[b] = x[m].mean()
            mean_s[b] = s[m].mean()

    centers_pct = (quantiles[:-1] + np.diff(quantiles) / 2)[:n_actual]
    valid = np.isfinite(mean_s)
    return centers_pct[valid], mean_x[valid], mean_s[valid]


# ---------- main ----------

def main() -> None:
    label_deter = "crisis"
    label_growth = "growth"
    if not set([label_deter, label_growth]).issubset(config.LABELS):
        raise RuntimeError(
            f"Symmetry test expects both {label_deter} and {label_growth} "
            f"in config.LABELS; got {config.LABELS}"
        )

    print(f"Loading SHAP caches for {MODEL_FAMILY}...")
    sv_d, X_d, feats_d = load_shap(MODEL_FAMILY, label_deter)
    sv_g, X_g, feats_g = load_shap(MODEL_FAMILY, label_growth)

    shared_features = [c for c in feats_d if c in feats_g]
    print(f"  shared feature columns: {len(shared_features)}")

    rank_d = mean_abs_ranking(sv_d, feats_d)
    rank_g = mean_abs_ranking(sv_g, feats_g)
    top_union = list(
        dict.fromkeys(
            list(rank_d.head(TOP_K).index) + list(rank_g.head(TOP_K).index)
        )
    )
    top_union = [f for f in top_union if f in shared_features]
    print(f"  top-{TOP_K}-union (shared): {len(top_union)} features")

    sv_d_df = pd.DataFrame(sv_d, columns=feats_d)
    sv_g_df = pd.DataFrame(sv_g, columns=feats_g)

    # Per-feature curves + Spearman
    records = []
    curves = {}
    for feat in top_union:
        xd = X_d[feat].to_numpy()
        sd = sv_d_df[feat].to_numpy()
        xg = X_g[feat].to_numpy()
        sg = sv_g_df[feat].to_numpy()

        cd = bin_curve(xd, sd)
        cg = bin_curve(xg, sg)
        if cd is None or cg is None:
            print(f"  skip {feat} (insufficient variation)")
            continue

        # Align curves on the intersection of percentile centers (they should
        # coincide when feature distributions on the test set are identical
        # across labels, which they are -- same rows -- so this is exact).
        pct_d, mx_d, ms_d = cd
        pct_g, mx_g, ms_g = cg
        common = np.intersect1d(np.round(pct_d, 4), np.round(pct_g, 4))
        if len(common) < 5:
            print(f"  skip {feat} (few common bins)")
            continue
        m_d = np.isin(np.round(pct_d, 4), common)
        m_g = np.isin(np.round(pct_g, 4), common)
        ms_d = ms_d[m_d]; mx_d = mx_d[m_d]
        ms_g = ms_g[m_g]
        neg_ms_g = -ms_g

        rho, pval = spearmanr(ms_d, neg_ms_g)
        records.append({
            "feature": feat,
            "rank_deter": int(rank_d.index.get_loc(feat)) + 1,
            "rank_growth": int(rank_g.index.get_loc(feat)) + 1,
            "mean_abs_shap_deter": float(rank_d[feat]),
            "mean_abs_shap_growth": float(rank_g[feat]),
            "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
            "spearman_p": float(pval) if np.isfinite(pval) else np.nan,
            "n_bins": int(len(common)),
        })
        curves[feat] = (mx_d, ms_d, ms_g, neg_ms_g)

    sym_df = (pd.DataFrame(records)
              .sort_values("spearman_rho", ascending=False, na_position="last")
              .reset_index(drop=True))
    metrics_dir = config.RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    sym_df.to_csv(metrics_dir / "symmetry_spearman.csv", index=False)
    print(f"\nWrote {metrics_dir / 'symmetry_spearman.csv'}")
    print(sym_df.to_string(index=False))

    figures_dir = config.ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- per-feature panels ---
    for feat, (mx, ms_d, ms_g, neg_ms_g) in curves.items():
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.plot(mx, ms_d, marker="o", color="#d62728",
                label=f"SHAP for {label_deter}")
        ax.plot(mx, ms_g, marker="s", linestyle="--", color="#2ca02c",
                label=f"SHAP for {label_growth}")
        ax.plot(mx, neg_ms_g, marker="x", linestyle=":", color="#1f77b4",
                label=f"-SHAP for {label_growth}")
        ax.axhline(0, color="gray", linewidth=0.5)
        rho = sym_df.loc[sym_df["feature"] == feat, "spearman_rho"].iloc[0]
        ax.set_title(f"{feat} (Spearman rho = {rho:.2f})")
        ax.set_xlabel("Feature value (bin mean)")
        ax.set_ylabel("Mean SHAP in bin")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        safe = feat.replace("/", "_").replace(" ", "_")
        fig.savefig(figures_dir / f"symmetry_{safe}.png", dpi=200)
        plt.close(fig)

    # --- grid of all panels ---
    n = len(curves)
    if n:
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                                 squeeze=False)
        for ax, (feat, (mx, ms_d, ms_g, neg_ms_g)) in zip(axes.ravel(), curves.items()):
            ax.plot(mx, ms_d, marker="o", color="#d62728", label="deter")
            ax.plot(mx, neg_ms_g, marker="x", linestyle=":", color="#1f77b4",
                    label="-growth")
            ax.axhline(0, color="gray", linewidth=0.5)
            rho = sym_df.loc[sym_df["feature"] == feat, "spearman_rho"].iloc[0]
            ax.set_title(f"{feat}\nrho={rho:.2f}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=8, loc="best")
        fig.suptitle("Symmetry: SHAP(deter) vs -SHAP(growth)", y=1.0)
        fig.tight_layout()
        fig.savefig(figures_dir / "symmetry_curves.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {figures_dir / 'symmetry_curves.png'}")

    # --- Spearman summary bar ---
    if len(sym_df):
        fig, ax = plt.subplots(figsize=(6.5, max(4.5, 0.3 * len(sym_df))))
        data = sym_df.iloc[::-1]
        colors = ["#2ca02c" if r > 0.5 else ("#d62728" if r < 0 else "#7f7f7f")
                  for r in data["spearman_rho"]]
        ax.barh(data["feature"], data["spearman_rho"], color=colors)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Spearman rho: SHAP(deter) vs -SHAP(growth)")
        ax.set_title("Driver symmetry across top features")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "symmetry_spearman_bar.png", dpi=200)
        plt.close(fig)
        print(f"Wrote {figures_dir / 'symmetry_spearman_bar.png'}")


if __name__ == "__main__":
    main()
