"""Robustness pack per Model_Plan.md sec 11.

Runs four robustness analyses on the test set using the best tree model
family (XGBoost by default; override with ROBUST_MODEL=lgbm).

1. Time stability      : metrics computed year-by-year on the test window.
2. Size subgroups      : metrics split by asset-size quartile (quartiles
                         are fit on the training window only, then applied
                         to test to avoid leakage).
3. Permutation importance : shuffle each feature on the test set and
                         measure drop in PR-AUC. Cross-check against SHAP
                         ranking from script 11.
4. Feature ablation    : retrain the model with each of the top-5 feature
                         groups ablated, measure test PR-AUC change.
                         Group definitions are in FEATURE_GROUPS below;
                         edit if your column names differ.

Writes:
    results/metrics/robust_time_stability.csv
    results/metrics/robust_size_subgroups.csv
    results/metrics/robust_permutation.csv
    results/metrics/robust_ablation.csv
    results/figures/robust_time_stability.png
    results/figures/robust_size_subgroups.png
    results/figures/robust_permutation_top.png
"""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, confusion_matrix,
)

from src import config


ROBUST_MODEL = os.environ.get("ROBUST_MODEL", "xgb")   # "xgb" or "lgbm"
RNG = np.random.default_rng(config.RANDOM_SEED)
N_PERMUTATIONS = 5        # permutation importance repeats per feature
TOP_K_PERM = 20           # how many top features (by SHAP) to permute
SIZE_COL = "assets"       # column to use for size quartiles

# Edit to match your column names. Any group whose members don't exist
# in the feature set is silently skipped.
FEATURE_GROUPS: dict[str, list[str]] = {
    "profitability": ["roa", "nim", "cost_of_funding"],
    "capital":       ["capital_ratio", "capital_ratio_chg"],
    "asset_quality": ["npl_ratio", "npl_ratio_chg", "oreo"],
    "growth_levels": ["asset_growth", "loan_growth", "deposit_growth",
                      "liab_growth", "equity_growth"],
    "intrayear":     [],    # auto-populated: any feature containing these tokens
    "macro":         ["gdp_growth", "inflation", "unemployment",
                      "fed_funds_rate", "real_int_rate", "pop_growth"],
}
INTRAYEAR_TOKENS = ("intra", "q4_", "_sd_", "_max_decline")


# ---------- metrics ----------

def core_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                 threshold: float | None = None) -> dict:
    out = {
        "n_total": int(len(y_true)),
        "n_pos":   int(y_true.sum()),
    }
    if len(np.unique(y_true)) < 2:
        out.update(pr_auc=np.nan, roc_auc=np.nan, brier=np.nan)
        return out
    out["pr_auc"] = average_precision_score(y_true, y_prob)
    out["roc_auc"] = roc_auc_score(y_true, y_prob)
    out["brier"] = brier_score_loss(y_true, y_prob)
    if threshold is not None:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        out.update(
            sensitivity=sens, specificity=spec,
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        )
    return out


# ---------- model loading & prediction ----------

def load_model(label: str):
    if ROBUST_MODEL == "xgb":
        feats = json.loads(
            (config.MODELS_DIR / f"xgb_{label}.features.json").read_text())
        booster = xgb.Booster()
        booster.load_model(str(config.MODELS_DIR / f"xgb_{label}.json"))
        return booster, feats
    elif ROBUST_MODEL == "lgbm":
        feats = json.loads(
            (config.MODELS_DIR / f"lgbm_{label}.features.json").read_text())
        booster = lgb.Booster(
            model_file=str(config.MODELS_DIR / f"lgbm_{label}.txt"))
        return booster, feats
    raise ValueError(f"Unknown ROBUST_MODEL: {ROBUST_MODEL}")


def predict(booster, X: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if ROBUST_MODEL == "xgb":
        dmat = xgb.DMatrix(X[feature_cols].to_numpy(), feature_names=feature_cols)
        return booster.predict(dmat)
    return booster.predict(X[feature_cols].to_numpy())


def fit_model_like(best_params: dict, best_iter: int,
                   X_tr: pd.DataFrame, y_tr: np.ndarray):
    """Refit a fresh model using the tuned hyperparameters — for ablation."""
    if ROBUST_MODEL == "xgb":
        params = dict(
            objective="binary:logistic", eval_metric="aucpr",
            tree_method="hist", n_estimators=best_iter,
            random_state=config.RANDOM_SEED, verbosity=0,
            **best_params,
        )
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, verbose=False)
        return model
    else:
        params = dict(
            objective="binary", metric="average_precision",
            boosting_type="gbdt", n_estimators=best_iter,
            subsample_freq=1, random_state=config.RANDOM_SEED,
            verbosity=-1, n_jobs=-1, **best_params,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        return model


def predict_fresh(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# ---------- main ----------

def main() -> None:
    print(f"Robustness pack on model family: {ROBUST_MODEL}")
    
    thresholds = pd.read_csv(config.RESULTS_DIR / "metrics" / "thresholds.csv")
    best_params_path = config.RESULTS_DIR / "metrics" / f"{'xgboost' if ROBUST_MODEL == 'xgb' else 'lightgbm'}_best_params.json"
    best_params_all = json.loads(best_params_path.read_text())

    metrics_dir = config.RESULTS_DIR / "metrics"
    figures_dir = config.ROOT / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    time_rows, size_rows, perm_rows, abl_rows = [], [], [], []

    for label in config.LABELS:
        print(f"\n=== {label} ===")

        clean_csv = config.get_clean_csv(label)
        splits_csv = config.get_splits_csv(label)
        print(f"Reading {clean_csv} and {splits_csv}")
        df = pd.read_csv(clean_csv, dtype={"id_rssd": "int64", "year": "int16"})
        splits = pd.read_csv(splits_csv, dtype={"id_rssd": "int64", "year": "int16"})
        df = df.merge(splits, on=config.ID_COLS, how="inner")

        train_mask = df["split"] == "train"
        test_mask = df["split"] == "test"
        print(f"  train rows: {train_mask.sum():,}   test rows: {test_mask.sum():,}")

        # Size quartiles fit on training
        if SIZE_COL not in df.columns:
            print(f"  WARNING: '{SIZE_COL}' not in data; size-subgroup analysis skipped")
            size_edges = None
        else:
            size_edges = np.quantile(
                df.loc[train_mask, SIZE_COL].dropna(), [0, 0.25, 0.5, 0.75, 1.0])
            size_edges[0] = -np.inf; size_edges[-1] = np.inf

        # Feature-group token expansion
        feat_pool = [c for c in df.columns
                     if c not in set(config.ID_COLS + config.LABELS
                                     + config.LABEL_COMPONENT_COLS
                                     + ["split", "cv_fold"])
                     and pd.api.types.is_numeric_dtype(df[c])]
        FEATURE_GROUPS["intrayear"] = [
            c for c in feat_pool if any(tok in c for tok in INTRAYEAR_TOKENS)
        ]

        booster, feature_cols = load_model(label)
        best = best_params_all[label]
        y_test = df.loc[test_mask, label].astype(int).to_numpy()
        X_test = df.loc[test_mask, feature_cols].reset_index(drop=True)
        X_train = df.loc[train_mask, feature_cols].reset_index(drop=True)
        y_train = df.loc[train_mask, label].astype(int).to_numpy()

        thr = float(
            thresholds.loc[(thresholds["model"] == ROBUST_MODEL)
                           & (thresholds["label"] == label), "threshold"].iloc[0]
        )
        p_test = predict(booster, X_test, feature_cols)
        baseline_pr = average_precision_score(y_test, p_test)
        print(f"  baseline test PR-AUC: {baseline_pr:.4f}   threshold: {thr:.4f}")

        # --- 1. Time stability ---
        test_years = df.loc[test_mask, "year"].to_numpy()
        for yr in sorted(np.unique(test_years)):
            m = test_years == yr
            row = core_metrics(y_test[m], p_test[m], threshold=thr)
            row.update(label=label, year=int(yr))
            time_rows.append(row)

        # --- 2. Size subgroups ---
        if size_edges is not None and SIZE_COL in df.columns:
            sizes = df.loc[test_mask, SIZE_COL].to_numpy()
            q_idx = np.digitize(sizes, size_edges[1:-1], right=False)
            for q in range(4):
                m = q_idx == q
                if m.sum() < 10:
                    continue
                row = core_metrics(y_test[m], p_test[m], threshold=thr)
                row.update(label=label, size_quartile=f"Q{q+1}")
                size_rows.append(row)

        # --- 3. Permutation importance ---
        # rank features by SHAP top-K if available, otherwise by model gain
        shap_path = config.RESULTS_DIR / "shap" / f"shap_values_{label}.npz"
        if shap_path.exists():
            with np.load(shap_path, allow_pickle=False) as z:
                shap_values = z["values"]
                shap_feats = z["feature_cols"].tolist()
            mean_abs = pd.Series(
                np.abs(shap_values).mean(axis=0), index=shap_feats
            ).sort_values(ascending=False)
            perm_feats = [f for f in mean_abs.index if f in feature_cols][:TOP_K_PERM]
        else:
            perm_feats = feature_cols[:TOP_K_PERM]

        print(f"  permuting {len(perm_feats)} features x {N_PERMUTATIONS} repeats...")
        for feat in perm_feats:
            drops = []
            for _ in range(N_PERMUTATIONS):
                X_perm = X_test.copy()
                X_perm[feat] = RNG.permutation(X_perm[feat].to_numpy())
                p_perm = predict(booster, X_perm, feature_cols)
                drops.append(baseline_pr - average_precision_score(y_test, p_perm))
            perm_rows.append({
                "label": label, "feature": feat,
                "baseline_pr_auc": float(baseline_pr),
                "mean_pr_auc_drop": float(np.mean(drops)),
                "std_pr_auc_drop": float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0,
            })

        # --- 4. Feature-group ablation (retrain) ---
        print(f"  ablating {len(FEATURE_GROUPS)} feature groups...")
        for gname, gcols in FEATURE_GROUPS.items():
            drop_cols = [c for c in gcols if c in feature_cols]
            if not drop_cols:
                continue
            keep = [c for c in feature_cols if c not in drop_cols]
            model = fit_model_like(best["params"], int(best["best_iter"]),
                                   X_train[keep], y_train)
            p_abl = predict_fresh(model, X_test[keep])
            pr_abl = average_precision_score(y_test, p_abl)
            abl_rows.append({
                "label": label, "group": gname, "n_dropped": len(drop_cols),
                "baseline_pr_auc": float(baseline_pr),
                "ablated_pr_auc": float(pr_abl),
                "delta_pr_auc": float(pr_abl - baseline_pr),
            })

    # Persist
    time_df = pd.DataFrame(time_rows)
    size_df = pd.DataFrame(size_rows)
    perm_df = pd.DataFrame(perm_rows)
    abl_df = pd.DataFrame(abl_rows)

    time_df.to_csv(metrics_dir / "robust_time_stability.csv", index=False)
    size_df.to_csv(metrics_dir / "robust_size_subgroups.csv", index=False)
    perm_df.to_csv(metrics_dir / "robust_permutation.csv", index=False)
    abl_df.to_csv(metrics_dir / "robust_ablation.csv", index=False)
    print("\nWrote 4 robustness CSVs to results/metrics/")

    # Figures
    if not time_df.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for label in config.LABELS:
            d = time_df[time_df["label"] == label].sort_values("year")
            ax.plot(d["year"], d["pr_auc"], marker="o", label=label)
        ax.set_xlabel("Test year")
        ax.set_ylabel("PR-AUC")
        ax.set_title("Time stability on test window")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "robust_time_stability.png", dpi=200)
        plt.close(fig)

    if not size_df.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        width = 0.35
        quarts = ["Q1", "Q2", "Q3", "Q4"]
        x = np.arange(len(quarts))
        for i, label in enumerate(config.LABELS):
            d = size_df[size_df["label"] == label].set_index("size_quartile")
            vals = [d.loc[q, "pr_auc"] if q in d.index else np.nan for q in quarts]
            ax.bar(x + (i - 0.5) * width, vals, width, label=label)
        ax.set_xticks(x); ax.set_xticklabels(quarts)
        ax.set_ylabel("PR-AUC")
        ax.set_title(f"Size-quartile stability ({SIZE_COL})")
        ax.grid(axis="y", alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "robust_size_subgroups.png", dpi=200)
        plt.close(fig)

    if not perm_df.empty:
        n_per = 15
        fig, axes = plt.subplots(
            1, len(config.LABELS),
            figsize=(5 * len(config.LABELS), max(4.5, 0.3 * n_per)),
            squeeze=False,
        )
        for ax, label in zip(axes.ravel(), config.LABELS):
            d = (perm_df[perm_df["label"] == label]
                 .sort_values("mean_pr_auc_drop", ascending=False)
                 .head(n_per).iloc[::-1])
            pretty_labels = [config.pretty(f) for f in d["feature"]]
            ax.barh(pretty_labels, d["mean_pr_auc_drop"],
                    xerr=d["std_pr_auc_drop"], color="#1f77b4")
            ax.set_xlabel("PR-AUC drop when permuted")
            ax.set_title(label)
            ax.grid(axis="x", alpha=0.3)
        fig.suptitle("Permutation importance (top features)", y=1.02)
        fig.tight_layout()
        fig.savefig(figures_dir / "robust_permutation_top.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)

    print("\nSummary (ablation):")
    print(abl_df.to_string(index=False))


if __name__ == "__main__":
    main()
