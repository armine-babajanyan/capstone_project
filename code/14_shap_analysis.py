"""SHAP analysis for the best tree model per label.

Per Model_Plan.md sec 10:
  * Compute TreeSHAP values on the test set for the best XGBoost model
    for each label (crisis, growth).
  * Produce beeswarm summary plot, bar plot of mean |SHAP|, and
    dependence plots for the top-10 features.
  * Cache SHAP arrays to results/shap/ so the symmetry test in
    15_symmetry_test.py does not recompute them.

"Best" here defaults to XGBoost per the plan, but the module exposes
MODEL_FAMILY so you can rerun on LightGBM instead by setting it to "lgbm".

Writes:
    results/shap/shap_values_<label>.npz          (values, base value, X)
    results/figures/shap_beeswarm_<label>.png
    results/figures/shap_bar_<label>.png
    results/figures/shap_dependence_<label>_<feature>.png   (top 10 per label)
    results/metrics/shap_top_features.csv          (mean |SHAP| table per label)
"""
import json
import os
import sys
import joblib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import lightgbm as lgb

from src import config


MODEL_FAMILIES = ["xgb", "lgbm", "rf"]
TOP_K_DEPENDENCE = 10
TOP_K_BAR = 15


# ---------- model loading ----------

def load_tree_model(family: str, label: str):
    """Return (predict-compatible model, feature_cols)."""
    if family == "xgb":
        feat_path = config.MODELS_DIR / f"xgb_{label}.features.json"
        model_path = config.MODELS_DIR / f"xgb_{label}.json"
        feature_cols = json.loads(feat_path.read_text())
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        return booster, feature_cols
    elif family == "lgbm":
        feat_path = config.MODELS_DIR / f"lgbm_{label}.features.json"
        model_path = config.MODELS_DIR / f"lgbm_{label}.txt"
        feature_cols = json.loads(feat_path.read_text())
        booster = lgb.Booster(model_file=str(model_path))
        return booster, feature_cols
    elif family == "rf":
        model_path = config.MODELS_DIR / f"rf_{label}.joblib"
        bundle = joblib.load(model_path)
        return bundle["model"], bundle["features"]
    else:
        raise ValueError(f"Family must be 'xgb', 'lgbm', or 'rf', got {family!r}")


import shap.explainers._tree

# Monkey-patch float in shap._tree to fix XGBoost 3.x base_score parsing bug
original_float = float
def patched_float(x):
    if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
        x = x[1:-1]
    return original_float(x)
shap.explainers._tree.float = patched_float

def compute_shap(model, X: pd.DataFrame):
    """Compute TreeSHAP values for a binary classifier.

    Returns (shap_values: ndarray shape (n, p), base_value: float).
    Handles the API quirks where different versions / backends return
    either a raw array or an Explanation object.
    """
    explainer = shap.TreeExplainer(model)
    result = explainer(X)
    # New SHAP (>=0.40): returns an Explanation; .values is (n, p) for binary
    if hasattr(result, "values"):
        values = result.values
        base = result.base_values
        if values.ndim == 3:         # (n, p, classes) -> take positive class
            values = values[:, :, 1]
            base = base[:, 1] if base.ndim == 2 else base
        base_value = float(np.mean(base))
        return values, base_value
    # Very old SHAP: returns a list for classifiers
    if isinstance(result, list):
        values = result[1] if len(result) == 2 else result[0]
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)) and len(np.ravel(base)) > 1:
            base = np.ravel(base)[1]
        return values, float(base)
    return result, float(explainer.expected_value)


# ---------- plots ----------

def plot_beeswarm(shap_values: np.ndarray, X: pd.DataFrame, out_path: Path,
                  title: str) -> None:
    plt.figure(figsize=(7.5, 7.5))
    shap.summary_plot(shap_values, X, show=False, max_display=TOP_K_BAR)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_bar(mean_abs: pd.Series, out_path: Path, title: str) -> None:
    top = mean_abs.head(TOP_K_BAR).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, max(4.5, 0.3 * len(top))))
    ax.barh(top.index, top.values, color="#1f77b4")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dependence(feature: str, shap_values: np.ndarray, X: pd.DataFrame,
                    out_path: Path, title: str) -> None:
    plt.figure(figsize=(5.5, 4.5))
    shap.dependence_plot(
        feature, shap_values, X,
        interaction_index="auto", show=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------- main ----------

def main() -> None:
    shap_dir = config.RESULTS_DIR / "shap"
    figures_dir = config.ROOT / "results" / "figures"
    metrics_dir = config.RESULTS_DIR / "metrics"
    shap_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    top_rows = []

    for family in MODEL_FAMILIES:
        print(f"\n--- Model Family: {family} ---")
        for label in config.LABELS:
            print(f"\n=== {label} ===")

            try:
                model, feature_cols = load_tree_model(family, label)
            except FileNotFoundError:
                print(f"  Model files for {family}/{label} not found, skipping.")
                continue

            clean_csv = config.get_clean_csv(label)
            splits_csv = config.get_splits_csv(label)
            print(f"Reading {clean_csv} and {splits_csv}")
            df = pd.read_csv(clean_csv, dtype={"id_rssd": "int64", "year": "int16"})
            splits = pd.read_csv(splits_csv, dtype={"id_rssd": "int64", "year": "int16"})
            df = df.merge(splits, on=config.ID_COLS, how="inner")

            test = df.loc[df["split"] == "test"].reset_index(drop=True)
            print(f"  test rows: {len(test):,}")

            # Performance optimization: sample for SHAP if dataset is large or using RF
            SHAP_SAMPLE_SIZE = 2000
            if len(test) > SHAP_SAMPLE_SIZE:
                print(f"  sampling {SHAP_SAMPLE_SIZE:,} rows for SHAP stability and speed...")
                X_full = test[feature_cols].copy()
                X = X_full.sample(n=SHAP_SAMPLE_SIZE, random_state=config.RANDOM_SEED).sort_index()
            else:
                X = test[feature_cols].copy()

            print(f"  computing TreeSHAP for {family} on {len(X):,} rows...")
            shap_values, base_value = compute_shap(model, X)
            print(f"  base value: {base_value:.4f}   "
                  f"shap shape: {shap_values.shape}")

            # Cache for symmetry test (usually for best model, here we cache all with family suffix)
            cache_path = shap_dir / f"shap_values_{family}_{label}.npz"
            np.savez_compressed(
                cache_path,
                values=shap_values,
                base_value=np.array([base_value]),
                X=X.to_numpy(),
                feature_cols=np.array(feature_cols),
            )
            print(f"  wrote {cache_path}")

            # Pretty-name the features for plot display
            pretty_cols = [config.pretty(f) for f in feature_cols]
            X_pretty = X.copy()
            X_pretty.columns = pretty_cols
            mean_abs = (
                pd.Series(np.abs(shap_values).mean(axis=0), index=pretty_cols)
                .sort_values(ascending=False)
            )
            for rank, (feat, val) in enumerate(mean_abs.items(), start=1):
                top_rows.append({"family": family, "label": label, "rank": rank,
                                 "feature": feature_cols[pretty_cols.index(feat)],
                                 "mean_abs_shap": float(val)})

            # Plots
            plot_beeswarm(
                shap_values, X_pretty,
                figures_dir / f"shap_beeswarm_{family}_{label}.png",
                title=f"SHAP summary — {label} ({family.upper()})",
            )
            plot_bar(
                mean_abs,
                figures_dir / f"shap_bar_{family}_{label}.png",
                title=f"Mean |SHAP| — {label} ({family.upper()})",
            )
            top_k = mean_abs.head(TOP_K_DEPENDENCE).index.tolist()
            for pfeat in top_k:
                raw_feat = feature_cols[pretty_cols.index(pfeat)]
                safe = raw_feat.replace("/", "_").replace(" ", "_")
                plot_dependence(
                    pfeat, shap_values, X_pretty,
                    figures_dir / f"shap_dependence_{family}_{label}_{safe}.png",
                    title=f"{pfeat} — {label} ({family.upper()})",
                )
            print(f"  wrote beeswarm, bar, and {len(top_k)} dependence plots")

    pd.DataFrame(top_rows).to_csv(
        metrics_dir / "shap_top_features.csv", index=False
    )
    print(f"\nWrote {metrics_dir / 'shap_top_features.csv'}")


if __name__ == "__main__":
    main()
