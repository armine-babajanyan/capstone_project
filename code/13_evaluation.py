"""Test-set evaluation — the only script that touches the held-out test window.

For each of the 5 models (logit, lasso, xgb, lgbm, rf) x 2 labels
(deteriorate_t1, growth_t1):
  * Load the saved model from models/.
  * Rebuild the exact feature matrix on the test window (with the same
    preprocessing: median imputation for logit/lasso/rf using the bundled
    training medians, winsorization+standardization for logit/lasso).
  * Apply the F2-optimal threshold from results/metrics/thresholds.csv.
  * Compute the full metric suite from Model_Plan.md sec 9:
        PR-AUC, ROC-AUC, Brier, Type I / Type II error,
        Sensitivity, Specificity, G-mean, Recall@Precision=0.5,
        plus TP/FP/FN/TN at the chosen threshold.
  * Persist predictions and a single headline metrics CSV for the paper.

Writes:
    results/metrics/test_metrics.csv            (headline table)
    results/metrics/test_predictions.csv        (id, year, label, every prob column)
    results/tables/test_metrics.tex               (LaTeX fragment)
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    confusion_matrix, precision_recall_curve,
)

from src import config
from src.preprocessing import log_transform_dollars
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


MODEL_NAMES = ["logit", "lasso", "hazard_logit",
               "mundlak", "xgb", "lgbm", "rf", "gee_independence"]


# ---------- metrics ----------

def recall_at_precision(y_true: np.ndarray, y_prob: np.ndarray,
                        target_precision: float = 0.5) -> float:
    """Highest recall achievable while keeping precision >= target.

    Returns 0 if the target precision is never achieved.
    """
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    feasible = prec >= target_precision
    if not feasible.any():
        return 0.0
    return float(rec[feasible].max())


def full_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                 threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan      # recall / TPR
    spec = tn / (tn + fp) if (tn + fp) else np.nan      # TNR
    type1 = fp / (fp + tn) if (fp + tn) else np.nan     # false alarm rate
    type2 = fn / (fn + tp) if (fn + tp) else np.nan     # missed-distress rate
    gmean = float(np.sqrt(sens * spec)) if np.isfinite(sens * spec) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else np.nan
    return {
        "pr_auc":        average_precision_score(y_true, y_prob),
        "roc_auc":       roc_auc_score(y_true, y_prob),
        "accuracy":      acc,
        "recall":        sens,
        "precision":     prec,
        "brier":         brier_score_loss(y_true, y_prob),
        "threshold":     threshold,
        "sensitivity":   sens,
        "specificity":   spec,
        "type1_error":   type1,
        "type2_error":   type2,
        "g_mean":        gmean,
        "recall_at_p50": recall_at_precision(y_true, y_prob, 0.5),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_pos":   int(y_true.sum()),
        "n_total": int(len(y_true)),
    }


# ---------- preprocessing replay per model family ----------

def prep_linear(X: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Apply the winsor + standardize pipeline saved by script 05."""
    cols = bundle["feature_cols"]
    X = X[cols].copy()
    X = X.fillna(bundle["median"])
    X = X.clip(lower=bundle["winsor_lo"], upper=bundle["winsor_hi"], axis=1)
    Xs = bundle["scaler"].transform(X)
    return pd.DataFrame(Xs, index=X.index, columns=cols)


def predict_logit_like(bundle: dict, X_test_full: pd.DataFrame) -> np.ndarray:
    """Logit uses a VIF-reduced feature subset stored as bundle['features']."""
    Xp = prep_linear(X_test_full, bundle)
    return bundle["model"].predict_proba(Xp[bundle["features"]])[:, 1]


def predict_lasso(bundle: dict, X_test_full: pd.DataFrame) -> np.ndarray:
    Xp = prep_linear(X_test_full, bundle)
    return bundle["model"].predict_proba(Xp)[:, 1]


def predict_rf(bundle: dict, X_test_full: pd.DataFrame) -> np.ndarray:
    cols = bundle["features"]
    X = X_test_full[cols].fillna(bundle["median"])
    return bundle["model"].predict_proba(X)[:, 1]


def predict_xgb(model_path: Path, features_path: Path,
                X_test_full: pd.DataFrame) -> np.ndarray:
    feature_cols = json.loads(features_path.read_text())
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    # XGBClassifier saves as Booster-compatible JSON; use DMatrix for inference
    dmat = xgb.DMatrix(X_test_full[feature_cols].to_numpy(),
                       feature_names=feature_cols)
    return booster.predict(dmat)


def predict_lgbm(model_path: Path, features_path: Path,
                 X_test_full: pd.DataFrame) -> np.ndarray:
    feature_cols = json.loads(features_path.read_text())
    booster = lgb.Booster(model_file=str(model_path))
    return booster.predict(X_test_full[feature_cols].to_numpy())

def predict_hazard(bundle: dict, test_df: pd.DataFrame) -> np.ndarray:
    """Hazard model needs age/size splines + year dummies rebuilt on test."""
    import statsmodels.api as sm
    result = bundle["model"]
    feature_cols = bundle["feature_cols"]
    scaler = bundle["scaler"]
    median = bundle["median"]

    X_base = log_transform_dollars(test_df[feature_cols].copy())

    # Age splines
    age_col = "age"
    age_arr = test_df[age_col].to_numpy().astype(float)
    bad = ~np.isfinite(age_arr)
    age_arr[bad] = bundle["age_median"]
    age_s = bundle["age_spline"].transform(age_arr.reshape(-1, 1))
    age_names = [f"age_spline_{i}" for i in range(age_s.shape[1])]

    # Size splines
    size_col = "num_employees"
    size_arr = test_df[size_col].to_numpy().astype(float)
    bad = ~np.isfinite(size_arr)
    size_arr[bad] = bundle["size_median"]
    size_s = bundle["size_spline"].transform(size_arr.reshape(-1, 1))
    size_names = [f"size_spline_{i}" for i in range(size_s.shape[1])]

    # Year dummies (unseen test years get all zeros)
    yr_cols = [f"yr_{yr}" for yr in bundle["dummy_years"]]
    yr_df = pd.DataFrame(0, index=X_base.index, columns=yr_cols, dtype="int8")
    for yr in bundle["dummy_years"]:
        mask = test_df["year"] == yr
        if mask.any():
            yr_df.loc[mask, f"yr_{yr}"] = 1

    X = pd.concat([
        X_base,
        pd.DataFrame(age_s, index=X_base.index, columns=age_names),
        pd.DataFrame(size_s, index=X_base.index, columns=size_names),
        yr_df,
    ], axis=1)

    X = X.fillna(median).replace([np.inf, -np.inf], np.nan).fillna(median)
    X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    X_const = sm.add_constant(X_scaled)
    return np.asarray(result.predict(X_const))


def predict_mundlak(bundle: dict, X_test_full: pd.DataFrame, test_df: pd.DataFrame, full_df: pd.DataFrame, train_mask: pd.Series) -> np.ndarray:
    """Mundlak model uses base features + bank-level means."""
    feature_cols = bundle["feature_cols"]
    mean_cols = bundle["mean_cols"]
    all_cols = bundle["all_feat_cols"]

    # Compute bank-level means on training window
    train_means = (
        full_df.loc[train_mask]
        .groupby("id_rssd")[feature_cols]
        .mean()
    )
    # Global fallback for banks not in training
    global_means = train_means.mean()

    # Re-build the bank means for the test set
    X_test_means = pd.DataFrame(index=test_df.index)
    for col in feature_cols:
        mcol = f"{col}_bankmean"
        bank_mean_map = train_means[col]
        X_test_means[mcol] = test_df["id_rssd"].map(bank_mean_map)
        X_test_means[mcol] = X_test_means[mcol].fillna(global_means[col])

    # Join base features (already log-transformed in X_test_full) and bank means (raw, like in training)
    X = pd.concat([X_test_full[feature_cols], X_test_means], axis=1)
    X = X.reindex(columns=all_cols)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(bundle["median"])
    X_scaled = bundle["scaler"].transform(X)
    return bundle["model"].predict_proba(X_scaled)[:, 1]





# ---------- main ----------

def main() -> None:
    thresholds_path = config.RESULTS_DIR / "metrics" / "thresholds.csv"
    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"{thresholds_path} not found. Run 12_calibration.py first."
        )
    thresholds = pd.read_csv(thresholds_path)

    rows = []
    preds_out = None

    for label in config.LABELS:
        print(f"\n=== {label} ===")

        clean_csv = config.get_clean_csv(label)
        splits_csv = config.get_splits_csv(label)
        print(f"Reading {clean_csv} and {splits_csv}")
        df = pd.read_csv(clean_csv, dtype={"id_rssd": "int64", "year": "int16"})
        splits = pd.read_csv(splits_csv, dtype={"id_rssd": "int64", "year": "int16"})
        df = df.merge(splits, on=config.ID_COLS, how="inner")

        test_mask = df["split"] == "test"
        test = df.loc[test_mask].reset_index(drop=True)
        
        if preds_out is None:
            preds_out = test[config.ID_COLS].copy()
            
        preds_out[label] = test[label].to_numpy()

        print(f"  test rows: {len(test):,} "
              f"(years {test['year'].min()}-{test['year'].max()})")
        print(f"    {label} positives: {int(test[label].sum())} "
              f"({test[label].mean():.4f})")

        # Feature pool: numeric, non-id, non-label, non-split. Individual models
        # take the subset they were trained on via their saved feature list.
        excluded = set(config.ID_COLS + config.LABELS + config.LABEL_COMPONENT_COLS
                       + ["split", "cv_fold"])
        feat_pool = [c for c in test.columns
                     if c not in excluded and pd.api.types.is_numeric_dtype(test[c])]
        X_test_full = log_transform_dollars(test[feat_pool])

        y_true = test[label].astype(int).to_numpy()

        for model_name in MODEL_NAMES:
            thr_row = thresholds[(thresholds["model"] == model_name)
                                 & (thresholds["label"] == label)]
            if thr_row.empty:
                print(f"  skip {model_name}: no threshold in thresholds.csv")
                continue
            threshold = float(thr_row["threshold"].iloc[0])

            try:
                if model_name == "logit":
                    bundle = joblib.load(config.MODELS_DIR / f"logit_{label}.joblib")
                    y_prob = predict_logit_like(bundle, X_test_full)
                elif model_name == "lasso":
                    bundle = joblib.load(config.MODELS_DIR / f"lasso_{label}.joblib")
                    y_prob = predict_lasso(bundle, X_test_full)
                elif model_name == "hazard_logit":
                    bundle = joblib.load(config.MODELS_DIR / f"{model_name}_{label}.joblib")
                    y_prob = predict_hazard(bundle, test)
                elif model_name == "mundlak":
                    bundle = joblib.load(config.MODELS_DIR / f"mundlak_{label}.joblib")
                    y_prob = predict_mundlak(bundle, X_test_full, test, df, df["split"] == "train")
                elif model_name == "rf":
                    bundle = joblib.load(config.MODELS_DIR / f"rf_{label}.joblib")
                    y_prob = predict_rf(bundle, X_test_full)
                elif model_name == "xgb":
                    y_prob = predict_xgb(
                        config.MODELS_DIR / f"xgb_{label}.json",
                        config.MODELS_DIR / f"xgb_{label}.features.json",
                        X_test_full,
                    )
                elif model_name == "lgbm":
                    y_prob = predict_lgbm(
                        config.MODELS_DIR / f"lgbm_{label}.txt",
                        config.MODELS_DIR / f"lgbm_{label}.features.json",
                        X_test_full,
                    )
                elif model_name == "gee_independence":
                    bundle = joblib.load(config.MODELS_DIR / f"gee_independence_{label}.joblib")
                    # GEE predict: rebuild scaled features + constant
                    import statsmodels.api as sm
                    from sklearn.preprocessing import StandardScaler
                    cols = bundle["feature_cols"]
                    X = X_test_full[cols].replace([np.inf, -np.inf], np.nan)
                    X = X.fillna(bundle["median"])
                    X_scaled = pd.DataFrame(bundle["scaler"].transform(X),
                                            index=X.index, columns=cols)
                    X_const = sm.add_constant(X_scaled)
                    y_prob = bundle["result"].predict(X_const).to_numpy()
                else:
                    continue

            except FileNotFoundError as e:
                print(f"  skip {model_name}: missing artifact ({e.filename})")
                continue

            m = full_metrics(y_true, y_prob, threshold)
            m.update(model=model_name, label=label)
            rows.append(m)
            preds_out[f"{model_name}_{label}"] = y_prob

            print(f"  {model_name:<6s}  "
                  f"PR-AUC={m['pr_auc']:.3f}  ROC-AUC={m['roc_auc']:.3f}  "
                  f"Brier={m['brier']:.3f}  "
                  f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}  "
                  f"G-mean={m['g_mean']:.3f}  R@P50={m['recall_at_p50']:.3f}")

    if not rows:
        raise RuntimeError("No models evaluated. Check models/ and thresholds.csv.")

    metrics_dir = config.RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    col_order = [
        "model", "label", "pr_auc", "roc_auc", "accuracy", "recall", "precision",
        "brier", "threshold", "sensitivity", "specificity",
        "type1_error", "type2_error", "g_mean", "recall_at_p50",
        "tp", "fp", "fn", "tn", "n_pos", "n_total",
    ]
    metrics_df = pd.DataFrame(rows)[col_order]
    metrics_df.to_csv(metrics_dir / "test_metrics.csv", index=False)
    print(f"\nWrote {metrics_dir / 'test_metrics.csv'}")

    preds_out.to_csv(metrics_dir / "test_predictions.csv", index=False)
    print(f"Wrote {metrics_dir / 'test_predictions.csv'}")

    # LaTeX fragment for the paper
    tables_dir = config.ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    latex_cols = ["model", "accuracy", "recall", "precision", "pr_auc", "roc_auc"]
    
    for label in config.LABELS:
        label_df = metrics_df[metrics_df["label"] == label][latex_cols]
        out_path = tables_dir / f"test_metrics_{label}.tex"
        out_path.write_text(
            label_df.to_latex(
                index=False, float_format="%.3f",
                caption=f"Test-set performance: {label.capitalize()} target.",
                label=f"tab:test_metrics_{label}",
            )
        )
        print(f"Wrote {out_path}")

    print("\nHeadline test metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
