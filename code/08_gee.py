"""Random Effects logit via GEE with exchangeable within-bank correlation.

This is the textbook panel data classification model (Liang & Zeger, 1986).
GEE with an exchangeable working correlation structure estimates population-
averaged effects — the expected change in P(crisis) for the average bank
when a feature changes by one unit — while accounting for the fact that
repeated observations on the same bank are correlated.

Key properties:
  * Coefficients are consistent even if the correlation structure is wrong
    (though efficiency improves when it's correct).
  * Unlike GLMM/glmer, GEE doesn't estimate per-bank random intercepts,
    so it scales to 20k+ banks without memory issues.
  * Unlike fixed-effects logit, it can predict on unseen banks.

Two correlation structures are fitted for comparison:
  * Exchangeable: constant within-bank correlation (the RE assumption)
  * Independence: ignores within-bank correlation (equivalent to pooled logit
    with robust standard errors — useful as a sanity check)

Writes:
    models/gee_exchangeable_<label>.joblib
    models/gee_independence_<label>.joblib
    results/metrics/gee_val_metrics.csv
    results/metrics/gee_coefficients.csv
    results/metrics/val_predictions_gee.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm
import statsmodels.genmod.cov_struct as cov_struct
from sklearn.preprocessing import StandardScaler

from src import config
from src.metrics import val_metrics
from src.preprocessing import log_transform_dollars


# ---------- main ----------

def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = config.RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results_metrics = []
    results_coefs = []
    val_preds = None

    structures = [
        ("exchangeable", cov_struct.Exchangeable()),
        ("independence", cov_struct.Independence()),
    ]

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

        # Feature set
        excluded = set(config.ID_COLS + config.LABELS + config.LABEL_COMPONENT_COLS
                       + ["split", "cv_fold"])
        feature_cols = [c for c in df.columns
                        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

        X_probe = df.loc[train_mask, feature_cols]
        feature_cols = [c for c in feature_cols
                        if X_probe[c].notna().sum() > 0
                        and X_probe[c].nunique(dropna=True) > 1]

        X_raw = log_transform_dollars(df[feature_cols].replace([np.inf, -np.inf], np.nan))
        median = X_raw.loc[train_mask].median()
        X_imp = X_raw.fillna(median)

        scaler = StandardScaler().fit(X_imp.loc[train_mask])
        X_scaled = pd.DataFrame(
            scaler.transform(X_imp), index=X_imp.index, columns=feature_cols
        )

        print(f"  train: {train_mask.sum():,}   val: {val_mask.sum():,}   "
              f"features: {len(feature_cols)}")

        # GEE needs the group variable (bank id) as an array aligned with the data
        groups_train = df.loc[train_mask, "id_rssd"].to_numpy()
        groups_val = df.loc[val_mask, "id_rssd"].to_numpy()

        n_banks_train = len(np.unique(groups_train))
        n_banks_val = len(np.unique(groups_val))
        print(f"  unique banks — train: {n_banks_train:,}   val: {n_banks_val:,}")

        # Sort training data by bank id (GEE requires observations grouped by cluster)
        train_order = df.loc[train_mask].sort_values("id_rssd").index
        val_order = df.loc[val_mask].sort_values("id_rssd").index

        prep_bundle = {
            "feature_cols": feature_cols,
            "median": median,
            "scaler": scaler,
        }

        y_train = df.loc[train_mask, label].astype(float)
        y_val = df.loc[val_mask, label].astype(int).to_numpy()
        print(f"  train pos rate: {y_train.mean():.4f}   "
              f"val pos rate: {y_val.mean():.4f}")

        X_tr = sm.add_constant(X_scaled.loc[train_mask])
        X_va = sm.add_constant(X_scaled.loc[val_mask])

        for struct_name, corr_struct in structures:
            print(f"  Fitting GEE ({struct_name})...")

            # GEE requires data sorted by group
            X_tr_sorted = X_tr.loc[train_order]
            y_tr_sorted = y_train.loc[train_order]
            groups_sorted = df.loc[train_order, "id_rssd"].to_numpy()

            model = sm.GEE(
                endog=y_tr_sorted,
                exog=X_tr_sorted,
                groups=groups_sorted,
                family=sm.families.Binomial(sm.families.links.Logit()),
                cov_struct=corr_struct,
            )

            try:
                result = model.fit(maxiter=100)
            except Exception as e:
                print(f"    FAILED: {e}")
                continue

            # Predict on validation
            p_val = result.predict(X_va.loc[val_order])

            # Re-align predictions to original val order
            p_val_aligned = p_val.reindex(val_order).to_numpy()
            # Map back to df val_mask order
            p_val_final = pd.Series(p_val_aligned, index=val_order).reindex(
                df.loc[val_mask].index
            ).to_numpy()

            m = val_metrics(y_val, p_val_final)
            model_name = f"gee_{struct_name}"
            m.update(model=model_name, label=label, n_features=len(feature_cols))

            # Report estimated correlation if exchangeable
            if struct_name == "exchangeable":
                try:
                    dep_params = result.cov_struct.summary()
                    print(f"    within-bank correlation: {dep_params}")
                except Exception:
                    pass

            results_metrics.append(m)

            joblib.dump({"result": result, **prep_bundle},
                        config.MODELS_DIR / f"{model_name}_{label}.joblib")

            # Coefficients (skip intercept at index 0)
            params = result.params[1:]
            pvalues = result.pvalues[1:]
            for feat, coef, pv in zip(feature_cols, params, pvalues):
                if abs(coef) > 1e-10:
                    results_coefs.append({
                        "model": model_name, "label": label,
                        "feature": feat, "coef": float(coef),
                        "robust_se": float(result.bse[feature_cols.index(feat) + 1]),
                        "pvalue": float(pv),
                    })

            val_preds[f"{model_name}_{label}"] = p_val_final
            print(f"    PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}")

    pd.DataFrame(results_metrics).to_csv(
        metrics_dir / "gee_val_metrics.csv", index=False
    )
    pd.DataFrame(results_coefs).to_csv(
        metrics_dir / "gee_coefficients.csv", index=False
    )
    val_preds.to_csv(metrics_dir / "val_predictions_gee.csv", index=False)

    print("\nValidation metrics:")
    print(pd.DataFrame(results_metrics).to_string(index=False))


if __name__ == "__main__":
    main()
