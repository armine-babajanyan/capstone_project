"""Global interpretability and feature importance comparison across all models.

This script:
1. Loads coefficients from linear models (Logit, Lasso, Mundlak, GEE).
2. Loads SHAP importance from tree models (XGBoost, LightGBM, Random Forest).
3. Produces standardized coefficient bar plots for linear models.
4. Produces a consensus heatmap/table comparing top features across all models.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src import config

TOP_K = 15

def plot_coef_bar(df: pd.DataFrame, model_name: str, label: str, out_path: Path):
    """Plot horizontal bar chart of standardized coefficients."""
    df = df.copy()
    df["abs_coef"] = df["coef"].abs()
    top = df.sort_values("abs_coef", ascending=False).head(TOP_K).iloc[::-1]
    
    plt.figure(figsize=(8, 6))
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in top["coef"]]
    labels = [config.pretty(f) for f in top["feature"]]
    plt.barh(labels, top["coef"], color=colors)
    plt.xlabel("Standardized Coefficient (Magnitude = Importance, Sign = Direction)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    metrics_dir = config.RESULTS_DIR / "metrics"
    figures_dir = config.ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load linear coefficients
    coef_files = {
        "logit": metrics_dir / "baselines_coefficients.csv",
        "lasso": metrics_dir / "baselines_coefficients.csv",
        "gee": metrics_dir / "gee_coefficients.csv",
    }

    all_importances = []

    for model_key, path in coef_files.items():
        if not path.exists():
            print(f"Skipping {model_key}: {path} not found.")
            continue
        
        df = pd.read_csv(path)
        # Filter for the specific model if multiple are in one file
        if model_key in ["logit", "lasso"]:
            df = df[df["model"] == model_key]
        
        models = df["model"].unique() if "model" in df.columns else [model_key]
        for m in models:
            m_df = df[df["model"] == m] if "model" in df.columns else df
            for label in m_df["label"].unique():
                label_df = m_df[m_df["label"] == label]
                if label_df.empty: continue
                
                # Save individual plot
                mname = m
                plot_path = figures_dir / f"coef_bar_{mname}_{label}.png"
                plot_coef_bar(label_df, mname.upper(), label, plot_path)
                print(f"  Saved {plot_path}")
                
                # Record for consensus (using absolute coef for rank)
                label_df = label_df.copy()
                label_df["importance"] = label_df["coef"].abs()
                # Normalize importance to [0, 1] within model/label
                max_imp = label_df["importance"].max()
                if max_imp > 0:
                    label_df["norm_importance"] = label_df["importance"] / max_imp
                else:
                    label_df["norm_importance"] = 0
                
                for _, row in label_df.iterrows():
                    all_importances.append({
                        "model": mname,
                        "label": label,
                        "feature": row["feature"],
                        "importance": row["norm_importance"]
                    })

    shap_path = metrics_dir / "shap_top_features.csv"
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        # Handle old CSV format without 'family' column
        if "family" not in shap_df.columns:
            shap_df["family"] = "best_tree"
            
        for family in shap_df["family"].unique():
            for label in shap_df["label"].unique():
                subset = shap_df[(shap_df["family"] == family) & (shap_df["label"] == label)]
                if subset.empty: continue
                
                # Normalize SHAP
                max_shap = subset["mean_abs_shap"].max()
                for _, row in subset.iterrows():
                    all_importances.append({
                        "model": family,
                        "label": label,
                        "feature": row["feature"],
                        "importance": row["mean_abs_shap"] / max_shap if max_shap > 0 else 0
                    })
    else:
        print(f"Skipping SHAP: {shap_path} not found.")

    if not all_importances:
        print("No importance data found to create consensus.")
        return

    # 3. Consolidate and Rank
    imp_df = pd.DataFrame(all_importances)
    
    for label in imp_df["label"].unique():
        print(f"\n--- Consensus for {label} ---")
        pivot = imp_df[imp_df["label"] == label].pivot_table(
            index="feature", columns="model", values="importance"
        )
        
        # Calculate mean importance and number of models that use each feature
        pivot["mean_importance"] = pivot.mean(axis=1)
        pivot["n_models"] = pivot.drop(columns="mean_importance").notna().sum(axis=1)
        # Sort: multi-model features first, then by mean importance
        pivot = pivot.sort_values(["n_models", "mean_importance"],
                                  ascending=[False, False])
        
        # Plot Consensus Heatmap
        top_pivot = pivot.head(20).drop(columns=["mean_importance", "n_models"])
        top_pivot.index = [config.pretty(f) for f in top_pivot.index]
        # Reorder and rename model columns
        col_order = ["logit", "lasso",
                     "gee_independence", "xgb", "lgbm", "rf"]
        col_rename = {"logit": "Logit", "lasso": "Lasso",
                      "gee_independence": "GEE", "xgb": "XGB",
                      "lgbm": "LGBM", "rf": "RF"}
        col_order = [c for c in col_order if c in top_pivot.columns]
        top_pivot = top_pivot[col_order].rename(columns=col_rename)
        plt.figure(figsize=(12, 10))
        from matplotlib.colors import LinearSegmentedColormap
        cmap_red = LinearSegmentedColormap.from_list("light_to_red", ["#fde8e8", "#8b0000"])
        sns.heatmap(top_pivot, annot=True, cmap=cmap_red, fmt=".2f",
                    annot_kws={"fontsize": 12})
        plt.xticks(fontsize=14, rotation=45, ha="right")
        plt.yticks(fontsize=14)
        plt.xlabel("Model", fontsize=15)
        plt.ylabel("Feature", fontsize=15)
        plt.tight_layout()
        consensus_plot = figures_dir / f"consensus_importance_{label}.png"
        plt.savefig(consensus_plot, dpi=200)
        print(f"  Saved {consensus_plot}")
        
        # Save CSV (drop helper columns)
        pivot.drop(columns="n_models").to_csv(metrics_dir / f"feature_consensus_{label}.csv")
        print(f"  Saved {metrics_dir / f'feature_consensus_{label}.csv'}")

if __name__ == "__main__":
    main()
