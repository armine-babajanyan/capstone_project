"""Random Forest training with Optuna temporal-CV tuning.

Same structure as 09_train_xgboost.py and 10_train_lightgbm.py. RF has no
early stopping, so n_estimators is a tuned hyperparameter. sklearn's
RandomForestClassifier does not handle NaN natively, so features are
median-imputed using training-set medians only.

Override trial budget with the N_TRIALS env var (default 100). RF Optuna
trials are slow because each trial fits the full forest per fold; consider
N_TRIALS=30 for a first pass.

Writes:
    models/rf_<label>.joblib           (model + imputer + feature list)
    results/metrics/rf_val_metrics.csv
    results/metrics/rf_best_params.json
    results/metrics/val_predictions_rf.csv
    results/logs/optuna_rf.db
"""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, accuracy_score, f1_score,
)

from src import config
from src.splits import iter_cv_folds
from src.metrics import val_metrics
from src.preprocessing import log_transform_dollars



N_TRIALS = int(os.environ.get("N_TRIALS", 20))




# ---------- CV fold translation ----------

def positional_cv_folds(df_full: pd.DataFrame, train_mask: pd.Series):
    pos = pd.Series(np.arange(train_mask.sum()), index=df_full.index[train_mask])
    out = []
    for _k, tr_idx, va_idx in iter_cv_folds(df_full):
        out.append((pos.loc[tr_idx].to_numpy(), pos.loc[va_idx].to_numpy()))
    return out


# ---------- Optuna objective ----------

def make_objective(X_train: pd.DataFrame, y_train: np.ndarray, cv_folds):
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("max_depth", 5, 25),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 50),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5]
            ),
            class_weight=trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]
            ),
            bootstrap=True,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
        fold_scores = []
        for tr_pos, va_pos in cv_folds:
            Xtr, Xva = X_train.iloc[tr_pos], X_train.iloc[va_pos]
            ytr, yva = y_train[tr_pos], y_train[va_pos]
            model = RandomForestClassifier(**params)
            model.fit(Xtr, ytr)
            p = model.predict_proba(Xva)[:, 1]
            fold_scores.append(average_precision_score(yva, p))
        return float(np.mean(fold_scores))
    return objective


# ---------- main ----------

def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = config.RESULTS_DIR / "metrics"
    logs_dir = config.RESULTS_DIR / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(logs_dir / 'optuna_rf.db').as_posix()}"

    all_metrics = []
    all_best_params: dict = {}
    val_preds = None

    for label in config.LABELS:
        print(f"\n=== {label} ===")

        clean_csv = config.get_clean_csv(label)
        splits_csv = config.get_splits_csv(label)
        print(f"Reading {clean_csv} and {splits_csv}")
        df = pd.read_csv(clean_csv, dtype={"id_rssd": "int64", "year": "int16"})
        splits = pd.read_csv(splits_csv, dtype={"id_rssd": "int64", "year": "int16"})
        df = df.merge(splits, on=config.ID_COLS, how="inner")

        excluded = set(config.ID_COLS + config.LABELS + config.LABEL_COMPONENT_COLS
                       + ["split", "cv_fold"])
        feature_cols = [c for c in df.columns
                        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

        train_mask = df["split"] == "train"
        val_mask = df["split"] == "val"
        
        if val_preds is None:
            val_preds = df.loc[val_mask, config.ID_COLS].reset_index(drop=True).copy()

        X_probe = df.loc[train_mask, feature_cols]
        feature_cols = [c for c in feature_cols
                        if X_probe[c].notna().sum() > 0
                        and X_probe[c].nunique(dropna=True) > 1]

        X_train_raw = log_transform_dollars(df.loc[train_mask, feature_cols].reset_index(drop=True))
        X_val_raw = log_transform_dollars(df.loc[val_mask, feature_cols].reset_index(drop=True))

        # RF doesn't handle NaN or Inf; convert Inf to NaN, then median-impute
        # using training-set medians only
        X_train_raw = X_train_raw.replace([np.inf, -np.inf], np.nan)
        X_val_raw = X_val_raw.replace([np.inf, -np.inf], np.nan)
        median = X_train_raw.median()
        X_train_full = X_train_raw.fillna(median)
        X_val = X_val_raw.fillna(median)

        print(f"  train: {train_mask.sum():,}   val: {val_mask.sum():,}   "
              f"features: {len(feature_cols)}")

        cv_folds = positional_cv_folds(df, train_mask)
        print(f"  temporal CV folds: {len(cv_folds)}")

        y_train = df.loc[train_mask, label].astype(int).to_numpy()
        y_val = df.loc[val_mask, label].astype(int).to_numpy()
        print(f"  train pos rate: {y_train.mean():.4f}")

        study = optuna.create_study(
            direction="maximize",
            study_name=f"rf_{label}",
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
        )
        existing = len(study.trials)
        to_run = max(N_TRIALS - existing, 0)
        print(f"  existing trials: {existing}   running: {to_run}")
        if to_run > 0:
            objective = make_objective(X_train_full, y_train, cv_folds)
            study.optimize(objective, n_trials=to_run, show_progress_bar=False)

        best_params = dict(study.best_params)
        print(f"  best CV PR-AUC: {study.best_value:.4f}")

        model = RandomForestClassifier(
            bootstrap=True,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            **best_params,
        )
        model.fit(X_train_full, y_train)

        p_val = model.predict_proba(X_val)[:, 1]
        m = val_metrics(y_val, p_val)
        m.update(model="rf", label=label, cv_pr_auc=float(study.best_value))
        all_metrics.append(m)
        all_best_params[label] = {
            "params": best_params,
            "cv_pr_auc": float(study.best_value),
        }

        model_path = config.MODELS_DIR / f"rf_{label}.joblib"
        joblib.dump(
            {"model": model, "features": feature_cols, "median": median},
            model_path,
        )
        val_preds[f"rf_{label}"] = p_val
        print(f"  saved {model_path}")

    pd.DataFrame(all_metrics).to_csv(
        metrics_dir / "rf_val_metrics.csv", index=False
    )
    (metrics_dir / "rf_best_params.json").write_text(
        json.dumps(all_best_params, indent=2)
    )
    val_preds.to_csv(metrics_dir / "val_predictions_rf.csv", index=False)

    print("\nValidation metrics:")
    print(pd.DataFrame(all_metrics).to_string(index=False))


if __name__ == "__main__":
    main()
