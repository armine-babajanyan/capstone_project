"""Mixed-effects logit with Mundlak/Chamberlain correlated random effects.

A true GLMM (e.g. R's glmer) is computationally prohibitive on 350k rows
with 24k groups. The Mundlak (1978) trick gives the same key insight —
separating within-bank from between-bank variation — at pooled-logit cost.

For each time-varying feature x_it, we add the bank-level mean x_bar_i as
an extra regressor. The coefficient on x_it then captures within-bank
changes (a bank's capital ratio dropping relative to its own average),
while the coefficient on x_bar_i captures between-bank differences
(a chronically low-capital bank vs a chronically high-capital one).

This is equivalent to a correlated random effects model under normality
of the random intercept (Wooldridge, 2019, ch. 15). For prediction on
unseen banks the random effect averages to zero, so predictive power is
similar to pooled logit — but the coefficient decomposition is directly
useful for the paper's interpretability story.

Writes:
    models/mundlak_<label>.joblib
    results/metrics/mundlak_val_metrics.csv
    results/metrics/mundlak_coefficients.csv
    results/metrics/val_predictions_mundlak.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, confusion_matrix,
)

from src import config
from src.metrics import val_metrics
from src.preprocessing import log_transform_dollars, make_recency_weights


# ---------- Mundlak feature engineering ----------

def add_mundlak_means(df: pd.DataFrame, feature_cols: list[str],
                      train_mask: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Add bank-level means of time-varying features. Means are computed
    on training rows only; for validation/test banks not seen in training,
    the global training mean is used (= random effect averaged to zero)."""
    mean_cols = []
    # Compute bank-level means on training window
    train_means = (
        df.loc[train_mask]
        .groupby("id_rssd")[feature_cols]
        .mean()
    )
    # Global fallback for banks not in training
    global_means = train_means.mean()

    for col in feature_cols:
        mcol = f"{col}_bankmean"
        # Map bank means to all rows
        bank_mean_map = train_means[col]
        df[mcol] = df["id_rssd"].map(bank_mean_map)
        # Fill unseen banks with global mean
        df[mcol] = df[mcol].fillna(global_means[col])
        mean_cols.append(mcol)

    return df, mean_cols




# ---------- main ----------

def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = config.RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results_metrics = []
    results_coefs = []
    val_preds = None

    for label in config.LABELS:
        print(f"\n=== {label} ===")

        clean_csv = config.get_clean_csv(label)
        splits_csv = config.get_splits_csv(label)
        print(f"Reading {clean_csv} and {splits_csv}")
        df = pd.read_csv(clean_csv, dtype={"id_rssd": "int64", "year": "int16"})
        splits = pd.read_csv(splits_csv, dtype={"id_rssd": "int64", "year": "int16"})
        df = df.merge(splits, on=config.ID_COLS, how="inner")

        train_mask = df["split"] == "train"
        val_mask = df["split"] == "val"

        if val_preds is None:
            val_preds = df.loc[val_mask, config.ID_COLS].copy()

        # Base feature set
        excluded = set(config.ID_COLS + config.LABELS + config.LABEL_COMPONENT_COLS
                       + ["split", "cv_fold"])
        feature_cols = [c for c in df.columns
                        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

        X_probe = df.loc[train_mask, feature_cols]
        feature_cols = [c for c in feature_cols
                        if X_probe[c].notna().sum() > 0
                        and X_probe[c].nunique(dropna=True) > 1]
        print(f"  base features: {len(feature_cols)}")

        # Add Mundlak means (bank-level averages of each feature)
        df, mean_cols = add_mundlak_means(df, feature_cols, train_mask)
        all_feat_cols = feature_cols + mean_cols
        print(f"  + {len(mean_cols)} Mundlak means = {len(all_feat_cols)} total features")

        X_train_raw = log_transform_dollars(df.loc[train_mask, all_feat_cols])
        X_val_raw = log_transform_dollars(df.loc[val_mask, all_feat_cols])

        # Impute + standardize
        X_train_raw = X_train_raw.replace([np.inf, -np.inf], np.nan)
        X_val_raw = X_val_raw.replace([np.inf, -np.inf], np.nan)
        median = X_train_raw.median()
        X_train_imp = X_train_raw.fillna(median)
        X_val_imp = X_val_raw.fillna(median)

        scaler = StandardScaler().fit(X_train_imp)
        X_train_s = pd.DataFrame(scaler.transform(X_train_imp),
                                 index=X_train_imp.index, columns=all_feat_cols)
        X_val_s = pd.DataFrame(scaler.transform(X_val_imp),
                               index=X_val_imp.index, columns=all_feat_cols)

        print(f"  train: {train_mask.sum():,}   val: {val_mask.sum():,}")

        prep_bundle = {
            "feature_cols": feature_cols, "mean_cols": mean_cols,
            "all_feat_cols": all_feat_cols, "median": median, "scaler": scaler,
        }

        y_train = df.loc[train_mask, label].astype(int).to_numpy()
        y_val = df.loc[val_mask, label].astype(int).to_numpy()
        print(f"  train pos rate: {y_train.mean():.4f}   "
              f"val pos rate: {y_val.mean():.4f}")

        model = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=config.RANDOM_SEED,
        )
        sw = make_recency_weights(df.loc[train_mask, "year"])
        model.fit(X_train_s, y_train, sample_weight=sw)

        p_val = model.predict_proba(X_val_s)[:, 1]
        m = val_metrics(y_val, p_val)
        m.update(model="mundlak", label=label, n_features=len(all_feat_cols))
        results_metrics.append(m)

        joblib.dump({"model": model, **prep_bundle},
                    config.MODELS_DIR / f"mundlak_{label}.joblib")

        # Coefficients: split into within-bank (base) and between-bank (mean)
        coefs = model.coef_.ravel()
        for feat, coef in zip(all_feat_cols, coefs):
            kind = "between" if feat.endswith("_bankmean") else "within"
            results_coefs.append({
                "model": "mundlak", "label": label,
                "feature": feat, "coef": float(coef), "effect_type": kind,
            })
        val_preds[f"mundlak_{label}"] = p_val

        # Print top within and between effects
        coef_df = pd.DataFrame(results_coefs)
        coef_df = coef_df[coef_df["label"] == label]
        print("  Top within-bank effects (time-varying):")
        within = (coef_df[coef_df["effect_type"] == "within"]
                  .reindex(coef_df["coef"].abs().sort_values(ascending=False).index)
                  .dropna(subset=["feature"]).head(5))
        for _, r in within.iterrows():
            print(f"    {r['feature']:<30s} {r['coef']:>8.4f}")
        print("  Top between-bank effects (Mundlak means):")
        between = (coef_df[coef_df["effect_type"] == "between"]
                   .reindex(coef_df["coef"].abs().sort_values(ascending=False).index)
                   .dropna(subset=["feature"]).head(5))
        for _, r in between.iterrows():
            print(f"    {r['feature']:<30s} {r['coef']:>8.4f}")

    pd.DataFrame(results_metrics).to_csv(
        metrics_dir / "mundlak_val_metrics.csv", index=False
    )
    pd.DataFrame(results_coefs).to_csv(
        metrics_dir / "mundlak_coefficients.csv", index=False
    )
    val_preds.to_csv(metrics_dir / "val_predictions_mundlak.csv", index=False)

    print("\nValidation metrics:")
    print(pd.DataFrame(results_metrics).to_string(index=False))


if __name__ == "__main__":
    main()
