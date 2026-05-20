"""Shared preprocessing utilities used across all training scripts.

Centralises the log-transform for dollar-amount features and recency
sample-weighting so every model family applies the same pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def log_transform_dollars(df: pd.DataFrame,
                          cols: list[str] | None = None) -> pd.DataFrame:
    """Apply signed log1p transform to dollar-amount / count features.

    ``sign(x) * log1p(|x|)`` preserves the sign for features like
    ``ytdnonint_inc`` that can be negative while compressing the extreme
    right skew (~25,000× max/median) that otherwise dominates linear models.

    Only columns actually present in *df* are transformed (safe to call
    on feature subsets).
    """
    if cols is None:
        cols = config.DOLLAR_COLS
    df = df.copy()
    for c in cols:
        if c in df.columns:
            s = df[c].astype(float)
            df[c] = np.sign(s) * np.log1p(np.abs(s))
    return df


def make_recency_weights(years: pd.Series | np.ndarray,
                         cutoff_year: int = config.RECENCY_WEIGHT_YEAR,
                         boost: float = 2.0) -> np.ndarray:
    """Return per-sample weight array: ``boost`` for years >= cutoff, 1 otherwise.

    This gives more influence to the modern banking regime (post-2000)
    which is more representative of validation/test conditions.
    """
    years = np.asarray(years)
    weights = np.ones(len(years), dtype=float)
    weights[years >= cutoff_year] = boost
    return weights
