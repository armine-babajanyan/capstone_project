"""Shared metric functions used across the pipeline."""
import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, precision_recall_curve,
)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                           metric: str = "f1") -> tuple[float, float]:
    """Sweep thresholds to find the one maximizing *metric*.

    Parameters
    ----------
    metric : {"f1", "f2", "youden"}
        ``"f1"`` — standard F1;
        ``"f2"`` — F-beta with beta=2 (weights recall 2×);
        ``"youden"`` — Youden's J = sensitivity + specificity - 1.

    Returns
    -------
    (threshold, score) — the best threshold and its corresponding score.
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thresholds)+1 points; drop last
    prec = prec[:-1]
    rec = rec[:-1]

    if metric == "f1":
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = 2 * prec * rec / (prec + rec)
        scores = np.where(np.isfinite(scores), scores, 0.0)
    elif metric == "f2":
        beta = 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
        scores = np.where(np.isfinite(scores), scores, 0.0)
    elif metric == "youden":
        # Need to compute sensitivity and specificity from scratch
        scores = np.empty(len(thresholds))
        for i, t in enumerate(thresholds):
            y_pred = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) else 0
            spec = tn / (tn + fp) if (tn + fp) else 0
            scores[i] = sens + spec - 1
    else:
        raise ValueError(f"Unknown metric '{metric}'")

    if len(thresholds) == 0:
        return 0.5, 0.0
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def val_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Compute a comprehensive metric dict for validation reporting.

    Reports metrics at both the given threshold AND the optimal F1 threshold.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan

    # Find optimal F1 threshold
    opt_threshold, opt_f1 = find_optimal_threshold(y_true, y_prob, metric="f1")
    y_pred_opt = (y_prob >= opt_threshold).astype(int)
    opt_prec = precision_score(y_true, y_pred_opt, zero_division=0)
    opt_rec = recall_score(y_true, y_pred_opt, zero_division=0)

    return {
        "pr_auc":          average_precision_score(y_true, y_prob),
        "roc_auc":         roc_auc_score(y_true, y_prob),
        "brier":           brier_score_loss(y_true, y_prob),
        "accuracy":        accuracy_score(y_true, y_pred),
        "precision":       precision_score(y_true, y_pred, zero_division=0),
        "recall":          recall_score(y_true, y_pred, zero_division=0),
        "f1":              f1_score(y_true, y_pred, zero_division=0),
        "sensitivity@0.5": sens,
        "specificity@0.5": spec,
        # Optimal-threshold metrics
        "opt_threshold":   opt_threshold,
        "opt_f1":          opt_f1,
        "opt_precision":   opt_prec,
        "opt_recall":      opt_rec,
        "n_pos":           int(y_true.sum()),
        "n_total":         int(len(y_true)),
    }
