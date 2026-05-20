"""Calibration and threshold selection on the validation set.

For each of the 5 models (logit, lasso, xgb, lgbm, rf) x 2 labels:
  * Calibration curve (predicted probability vs observed positive rate)
    with Brier score reported in the legend.
  * Operating threshold chosen by maximizing F2 on validation (weights
    recall 2x precision, per Model_Plan.md sec 9).

The thresholds written here are applied to the test set by 13_evaluation.py.

Reads the five val_predictions_*.csv files written by scripts 05-08.

Writes:
    results/metrics/thresholds.csv
    results/figures/calibration_<label>.png
    results/figures/f2_vs_threshold_<label>.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss, precision_recall_curve, precision_score, recall_score,
)

from src import config


MODEL_NAMES = ["logit", "lasso",
               "mundlak", "xgb", "lgbm", "rf", "gee_independence"]
SOURCE_CSVS = {
    "val_predictions_baselines.csv": ["logit", "lasso"],
    "val_predictions_mundlak.csv": ["mundlak"],
    "val_predictions_xgboost.csv": ["xgb"],
    "val_predictions_lightgbm.csv": ["lgbm"],
    "val_predictions_rf.csv": ["rf"],
    "val_predictions_gee.csv": ["gee_independence"],
}
COLORS = {
    "logit": "#7f7f7f",
    "lasso": "#bcbd22",
    "hazard_logit": "#8c564b",
    "mundlak": "#e377c2",
    "xgb":   "#1f77b4",
    "lgbm":  "#2ca02c",
    "rf":    "#d62728",
    "gee_independence": "#aec7e8",
}


# ---------- helpers ----------

def best_f2_threshold(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 2.0):
    """Return (threshold, f2, precision, recall) that maximize F-beta on val.

    Uses precision_recall_curve so the threshold grid is exactly the set of
    distinct probability values in y_prob, no arbitrary grid spacing.
    """
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    # prec, rec have length N+1 (last point is recall=0); thresh length N
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec)
        fbeta = np.where(np.isfinite(fbeta), fbeta, 0.0)
    # align: exclude the trailing PR point that has no associated threshold
    fbeta = fbeta[:-1]
    prec = prec[:-1]
    rec = rec[:-1]
    if len(thresh) == 0:
        return 0.5, 0.0, 0.0, 0.0
    k = int(np.argmax(fbeta))
    return float(thresh[k]), float(fbeta[k]), float(prec[k]), float(rec[k])


def load_val_frame(label: str) -> pd.DataFrame:
    """Build a single DataFrame with labels and all model predictions on val for a specific label."""
    clean_csv = config.get_clean_csv(label)
    splits_csv = config.get_splits_csv(label)
    
    df = pd.read_csv(clean_csv, dtype={"id_rssd": "int64", "year": "int16"})
    splits = pd.read_csv(splits_csv, dtype={"id_rssd": "int64", "year": "int16"})

    val_ids = splits.loc[splits["split"] == "val", config.ID_COLS]
    val = val_ids.merge(
        df[config.ID_COLS + [label]], on=config.ID_COLS, how="left"
    )

    metrics_dir = config.RESULTS_DIR / "metrics"
    for csv_name in SOURCE_CSVS:
        path = metrics_dir / csv_name
        if not path.exists():
            print(f"  WARNING: {path} missing; models from it will be skipped")
            continue
        preds = pd.read_csv(path, dtype={"id_rssd": "int64", "year": "int16"})
        
        # Only keep predictions for this label to avoid merge conflicts across different targets
        keep_cols = config.ID_COLS + [col for col in preds.columns if col.endswith(f"_{label}")]
        preds_label = preds[keep_cols]
        
        val = val.merge(preds_label, on=config.ID_COLS, how="left")
    return val


# ---------- plots ----------

def plot_calibration(val: pd.DataFrame, label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect")

    y_true = val[label].to_numpy()
    for model in MODEL_NAMES:
        col = f"{model}_{label}"
        if col not in val.columns:
            continue
        y_prob = val[col].to_numpy()
        mask = ~np.isnan(y_prob)
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_prob[mask]
        frac_pos, mean_pred = calibration_curve(yt, yp, n_bins=10, strategy="quantile")
        brier = brier_score_loss(yt, yp)
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.5,
                color=COLORS.get(model), label=f"{model} (Brier={brier:.3f})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(f"Calibration — {label} (validation)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_f2_vs_threshold(val: pd.DataFrame, label: str,
                         thresholds_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    y_true = val[label].to_numpy()
    grid = np.linspace(0.001, 0.999, 200)
    for model in MODEL_NAMES:
        col = f"{model}_{label}"
        if col not in val.columns:
            continue
        y_prob = val[col].to_numpy()
        mask = ~np.isnan(y_prob)
        yt, yp = y_true[mask], y_prob[mask]
        f2_vals = []
        for t in grid:
            pred = (yp >= t).astype(int)
            prec = precision_score(yt, pred, zero_division=0)
            rec = recall_score(yt, pred, zero_division=0)
            f2 = (5 * prec * rec / (4 * prec + rec)) if (4 * prec + rec) else 0.0
            f2_vals.append(f2)
        ax.plot(grid, f2_vals, color=COLORS.get(model), label=model, linewidth=1.5)
        # mark the chosen threshold
        row = thresholds_df[(thresholds_df["model"] == model)
                            & (thresholds_df["label"] == label)]
        if not row.empty:
            t = row["threshold"].iloc[0]
            f2 = row["f2"].iloc[0]
            ax.scatter([t], [f2], color=COLORS.get(model), zorder=5, s=40,
                       edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("F2")
    ax.set_title(f"F2 vs threshold — {label} (validation)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------- main ----------

def main() -> None:
    rows = []
    
    figures_dir = config.ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    for label in config.LABELS:
        print(f"\nLoading validation predictions for {label}...")
        val = load_val_frame(label)
        print(f"  val rows: {len(val):,}")

        for model in MODEL_NAMES:
            col = f"{model}_{label}"
            if col not in val.columns:
                print(f"  skip {col} (missing)")
                continue
            y_true = val[label].to_numpy()
            y_prob = val[col].to_numpy()
            mask = ~np.isnan(y_prob)
            yt, yp = y_true[mask], y_prob[mask]
            if mask.sum() == 0:
                continue
            t, f2, prec, rec = best_f2_threshold(yt, yp, beta=2.0)
            brier = brier_score_loss(yt, yp)
            rows.append({
                "model": model, "label": label,
                "threshold": t, "f2": f2,
                "precision": prec, "recall": rec,
                "brier": brier, "n_val": int(mask.sum()),
            })
            print(f"  {model:<6s} {label:<18s} threshold={t:.4f}  "
                  f"F2={f2:.3f}  P={prec:.3f}  R={rec:.3f}  Brier={brier:.3f}")
        
        # Get threshold df for this label to plot
        thresholds_df = pd.DataFrame([r for r in rows if r["label"] == label])
        if not thresholds_df.empty:
            cal_path = figures_dir / f"calibration_{label}.png"
            f2_path = figures_dir / f"f2_vs_threshold_{label}.png"
            plot_calibration(val, label, cal_path)
            plot_f2_vs_threshold(val, label, thresholds_df, f2_path)
            print(f"  wrote {cal_path}")
            print(f"  wrote {f2_path}")

    thresholds_all = pd.DataFrame(rows)
    metrics_dir = config.RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    thresholds_all.to_csv(metrics_dir / "thresholds.csv", index=False)
    print(f"\nWrote {metrics_dir / 'thresholds.csv'}")


if __name__ == "__main__":
    main()
