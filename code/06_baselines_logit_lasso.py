"""Baseline logistic regression and L1-regularized logit (LASSO).

For each label (deteriorate_t1, growth_t1):
  * Preprocess features (median-impute, winsorize 1/99th pct, standardize),
    with all statistics computed on the *training* window only.
  * Plain logit.
  * LASSO with C chosen by temporal CV (maximizing PR-AUC), using the gapped
    expanding-window folds produced by 05_splits.py.
  * Evaluate on the validation window, save models, coefficients, val preds.

Does not touch the test set — that is script 11's job.

Writes:
    models/logit_<label>.joblib
    models/lasso_<label>.joblib
    results/metrics/baselines_val_metrics.csv
    results/metrics/baselines_coefficients.csv
    results/metrics/val_predictions_baselines.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, accuracy_score, f1_score,
)

from src import config
from src.metrics import val_metrics
from src.splits import iter_cv_folds
from src.preprocessing import log_transform_dollars, make_recency_weights


# ---------- preprocessing ----------

def compute_winsor_bounds(X: pd.DataFrame, p_low=0.01, p_high=0.99):
    return X.quantile(p_low), X.quantile(p_high)


def apply_winsor(X: pd.DataFrame, lo: pd.Series, hi: pd.Series) -> pd.DataFrame:
    return X.clip(lower=lo, upper=hi, axis=1)


def drop_degenerate_columns(X: pd.DataFrame) -> list[str]:
    """Drop all-NaN and zero-variance columns (training-set based)."""
    keep = []
    for c in X.columns:
        col = X[c]
        if col.notna().sum() == 0:
            continue
        if col.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return keep







# ---------- CV fold translation ----------

def build_cv_indexer(splits: pd.DataFrame, train_mask: pd.Series):
    """LogisticRegressionCV wants positional indices into the training array.
    iter_cv_folds yields indices into the full splits frame; remap them.
    """
    train_positions = pd.Series(
        np.arange(train_mask.sum()), index=splits.index[train_mask]
    )
    folds = []
    for _k, tr_idx, va_idx in iter_cv_folds(splits):
        tr_pos = train_positions.loc[tr_idx].to_numpy()
        va_pos = train_positions.loc[va_idx].to_numpy()
        folds.append((tr_pos, va_pos))
    return folds


# ---------- main ----------

def main() -> None:
    # Output locations
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
        print(f"  merged: {len(df):,} rows")

        # Feature set: numeric, excluding ids / labels / label-components / split cols
        excluded = set(config.ID_COLS + config.LABELS + config.LABEL_COMPONENT_COLS
                       + ["split", "cv_fold"])
        feature_cols = [c for c in df.columns
                        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

        train_mask = df["split"] == "train"
        val_mask = df["split"] == "val"
        print(f"  train rows: {train_mask.sum():,}   val rows: {val_mask.sum():,}")
        print(f"  candidate features: {len(feature_cols)}")

        # Drop degenerate columns using training data only
        X_train_raw = df.loc[train_mask, feature_cols]
        feature_cols = drop_degenerate_columns(X_train_raw)
        X_train_raw = X_train_raw[feature_cols]
        X_val_raw = df.loc[val_mask, feature_cols]
        print(f"  features after degeneracy filter: {len(feature_cols)}")

        # Log-transform dollar-amount features (compresses 25,000× skew)
        X_train_raw = log_transform_dollars(X_train_raw)
        X_val_raw = log_transform_dollars(X_val_raw)

        # Preprocessing statistics on TRAIN only
        median = X_train_raw.median()
        lo, hi = compute_winsor_bounds(X_train_raw.fillna(median))

        def preprocess(X: pd.DataFrame) -> pd.DataFrame:
            return apply_winsor(X.fillna(median), lo, hi)

        X_train_p = preprocess(X_train_raw)
        X_val_p = preprocess(X_val_raw)

        scaler = StandardScaler().fit(X_train_p)
        X_train_s = pd.DataFrame(scaler.transform(X_train_p),
                                 index=X_train_p.index, columns=feature_cols)
        X_val_s = pd.DataFrame(scaler.transform(X_val_p),
                               index=X_val_p.index, columns=feature_cols)

        # Recency weights: 2× weight for post-2000 observations
        sample_weight = make_recency_weights(df.loc[train_mask, "year"])

        cv_folds = build_cv_indexer(df, train_mask)
        print(f"  temporal CV folds: {len(cv_folds)}")

        if val_preds is None:
            val_preds = df.loc[val_mask, config.ID_COLS].copy()

        # Preprocessing artifacts to bundle with each model (needed at test time)
        prep_bundle = {
            "feature_cols": feature_cols, "median": median,
            "winsor_lo": lo, "winsor_hi": hi, "scaler": scaler,
        }

        y_train = df.loc[train_mask, label].astype(int).to_numpy()
        y_val = df.loc[val_mask, label].astype(int).to_numpy()
        print(f"  train pos rate: {y_train.mean():.4f}   "
              f"val pos rate: {y_val.mean():.4f}")

        # ---- Plain logit ----
        print("  Plain logit...")
        logit = LogisticRegression(
            C=1e10, solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=config.RANDOM_SEED,
        ).fit(X_train_s, y_train, sample_weight=sample_weight)

        p_val = logit.predict_proba(X_val_s)[:, 1]
        m = val_metrics(y_val, p_val)
        m.update(model="logit", label=label, n_features=len(feature_cols))
        results_metrics.append(m)

        joblib.dump({"model": logit, "features": feature_cols, **prep_bundle},
                    config.MODELS_DIR / f"logit_{label}.joblib")
        for feat, coef in zip(feature_cols, logit.coef_.ravel()):
            results_coefs.append(
                {"model": "logit", "label": label, "feature": feat, "coef": coef}
            )
        val_preds[f"logit_{label}"] = p_val

        # ---- LASSO with temporal-CV C selection ----
        print("  LASSO (L1) with temporal CV (scoring=PR-AUC)...")
        lasso = LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 10),
            penalty="elasticnet", solver="saga",
            l1_ratios=[1],
            scoring="average_precision",
            cv=cv_folds,
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
            n_jobs=-1,
            random_state=config.RANDOM_SEED,
        ).fit(X_train_s, y_train, sample_weight=sample_weight)
        selected_C = float(lasso.C_[0])
        coefs = lasso.coef_.ravel()
        nonzero = int((coefs != 0).sum())
        print(f"    selected C = {selected_C:.4g}   "
              f"non-zero coefs: {nonzero}/{len(feature_cols)}")

        p_val = lasso.predict_proba(X_val_s)[:, 1]
        m = val_metrics(y_val, p_val)
        m.update(model="lasso", label=label, n_features=nonzero, C=selected_C)
        results_metrics.append(m)

        joblib.dump({"model": lasso, "features": feature_cols, **prep_bundle},
                    config.MODELS_DIR / f"lasso_{label}.joblib")
        for feat, coef in zip(feature_cols, coefs):
            if coef != 0:
                results_coefs.append(
                    {"model": "lasso", "label": label, "feature": feat, "coef": coef}
                )
        val_preds[f"lasso_{label}"] = p_val
        # (Removed label-specific CSV save)

    # Persist
    pd.DataFrame(results_metrics).to_csv(
        metrics_dir / "baselines_val_metrics.csv", index=False
    )
    pd.DataFrame(results_coefs).to_csv(
        metrics_dir / "baselines_coefficients.csv", index=False
    )
    val_preds.to_csv(metrics_dir / "val_predictions_baselines.csv", index=False)
    print(f"Wrote {metrics_dir / 'val_predictions_baselines.csv'}")

    print("\nValidation metrics:")
    print(pd.DataFrame(results_metrics).to_string(index=False))


if __name__ == "__main__":
    main()
