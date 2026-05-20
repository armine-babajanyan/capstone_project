"""LightGBM training with Optuna temporal-CV tuning.

Same structure as 09_train_xgboost.py, with LightGBM's idiomatic
hyperparameters (num_leaves instead of max_depth, min_child_samples instead
of min_child_weight) and LightGBM's native text format for model storage.

Override trial budget with the N_TRIALS env var (default 100).

Writes:
    models/lgbm_<label>.txt
    models/lgbm_<label>.features.json
    results/metrics/lightgbm_val_metrics.csv
    results/metrics/lightgbm_best_params.json
    results/metrics/val_predictions_lightgbm.csv
    results/logs/optuna_lgbm.db
"""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, accuracy_score, f1_score,
)

from src import config
from src.splits import iter_cv_folds
from src.metrics import val_metrics
from src.preprocessing import log_transform_dollars


N_TRIALS = int(os.environ.get("N_TRIALS", 100))
EARLY_STOPPING_ROUNDS = 50
MAX_N_ESTIMATORS = 2000


# ---------- CV fold translation to positional indices ----------

def positional_cv_folds(df_full: pd.DataFrame, train_mask: pd.Series):
    pos = pd.Series(np.arange(train_mask.sum()), index=df_full.index[train_mask])
    out = []
    for _k, tr_idx, va_idx in iter_cv_folds(df_full):
        out.append((pos.loc[tr_idx].to_numpy(), pos.loc[va_idx].to_numpy()))
    return out


# ---------- Optuna objective ----------

def make_objective(X_train: pd.DataFrame, y_train: np.ndarray,
                   cv_folds, scale_pos_weight_base: float):
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            objective="binary",
            metric="average_precision",
            boosting_type="gbdt",
            n_estimators=MAX_N_ESTIMATORS,
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            learning_rate=trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=1,            # required for subsample to take effect
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            scale_pos_weight=trial.suggest_float(
                "scale_pos_weight",
                scale_pos_weight_base * 0.5,
                scale_pos_weight_base * 1.5,
            ),
            random_state=config.RANDOM_SEED,
            verbosity=-1,
            n_jobs=-1,
        )
        fold_scores, fold_best_iters = [], []
        for tr_pos, va_pos in cv_folds:
            Xtr, Xva = X_train.iloc[tr_pos], X_train.iloc[va_pos]
            ytr, yva = y_train[tr_pos], y_train[va_pos]
            model = lgb.LGBMClassifier(**params)
            model.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                eval_metric="average_precision",
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            p = model.predict_proba(Xva)[:, 1]
            fold_scores.append(average_precision_score(yva, p))
            fold_best_iters.append(int(model.best_iteration_))
        trial.set_user_attr("best_iter_mean", int(np.mean(fold_best_iters)))
        return float(np.mean(fold_scores))
    return objective


# ---------- main ----------

def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = config.RESULTS_DIR / "metrics"
    logs_dir = config.RESULTS_DIR / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(logs_dir / 'optuna_lgbm.db').as_posix()}"

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

        X_train_full = log_transform_dollars(df.loc[train_mask, feature_cols].reset_index(drop=True))
        X_val = log_transform_dollars(df.loc[val_mask, feature_cols].reset_index(drop=True))
        print(f"  train: {train_mask.sum():,}   val: {val_mask.sum():,}   "
              f"features: {len(feature_cols)}")

        cv_folds = positional_cv_folds(df, train_mask)
        print(f"  temporal CV folds: {len(cv_folds)}")

        y_train = df.loc[train_mask, label].astype(int).to_numpy()
        y_val = df.loc[val_mask, label].astype(int).to_numpy()
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        spw = n_neg / max(n_pos, 1)
        print(f"  train pos rate: {y_train.mean():.4f}   "
              f"base scale_pos_weight = {spw:.2f}")

        study = optuna.create_study(
            direction="maximize",
            study_name=f"lgbm_{label}",
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
        )
        existing = len(study.trials)
        to_run = max(N_TRIALS - existing, 0)
        print(f"  existing trials: {existing}   running: {to_run}")
        if to_run > 0:
            objective = make_objective(X_train_full, y_train, cv_folds, spw)
            study.optimize(objective, n_trials=to_run, show_progress_bar=False)

        best_params = dict(study.best_params)
        best_iter = int(study.best_trial.user_attrs.get("best_iter_mean", 200))
        print(f"  best CV PR-AUC: {study.best_value:.4f}   best_iter: {best_iter}")

        refit_params = dict(
            objective="binary",
            metric="average_precision",
            boosting_type="gbdt",
            n_estimators=best_iter,
            subsample_freq=1,
            random_state=config.RANDOM_SEED,
            verbosity=-1,
            n_jobs=-1,
            **best_params,
        )
        model = lgb.LGBMClassifier(**refit_params)
        model.fit(X_train_full, y_train)

        p_val = model.predict_proba(X_val)[:, 1]
        m = val_metrics(y_val, p_val)
        m.update(model="lightgbm", label=label, best_iter=best_iter,
                 cv_pr_auc=float(study.best_value))
        all_metrics.append(m)
        all_best_params[label] = {
            "params": best_params,
            "best_iter": best_iter,
            "cv_pr_auc": float(study.best_value),
        }

        model_path = config.MODELS_DIR / f"lgbm_{label}.txt"
        model.booster_.save_model(model_path)
        (config.MODELS_DIR / f"lgbm_{label}.features.json").write_text(
            json.dumps(feature_cols)
        )
        val_preds[f"lgbm_{label}"] = p_val
        print(f"  saved {model_path}")

    pd.DataFrame(all_metrics).to_csv(
        metrics_dir / "lightgbm_val_metrics.csv", index=False
    )
    (metrics_dir / "lightgbm_best_params.json").write_text(
        json.dumps(all_best_params, indent=2)
    )
    val_preds.to_csv(metrics_dir / "val_predictions_lightgbm.csv", index=False)

    print("\nValidation metrics:")
    print(pd.DataFrame(all_metrics).to_string(index=False))


if __name__ == "__main__":
    main()
