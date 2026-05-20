"""Temporal train/val/test split and gapped expanding-window CV folds.

The executable that materializes splits to disk is `code/05_splits.py`.
This module holds the *functions* that script (and downstream training
scripts) import.
"""
from __future__ import annotations
import pandas as pd

from . import config


def sanity_check(df: pd.DataFrame) -> None:
    """Assert the cleaned CSV is ready for splitting."""
    for c in config.ID_COLS:
        assert c in df.columns, f"missing id column '{c}'"
    for lbl in config.LABELS:
        assert lbl in df.columns, f"missing label column '{lbl}'"
        assert df[lbl].notna().all(), f"'{lbl}' contains NA rows; drop upstream"
        assert df[lbl].isin([0, 1]).all(), f"'{lbl}' is not binary 0/1"
    assert not df.duplicated(subset=config.ID_COLS).any(), \
        "duplicate (id_rssd, year) rows"


def assign_outer_split(
    df: pd.DataFrame,
    train_years: tuple[int, int] = config.TRAIN_YEARS,
    val_years: tuple[int, int] = config.VAL_YEARS,
    test_years: tuple[int, int] = config.TEST_YEARS,
) -> pd.Series:
    """Return a Series labeling each row 'train' / 'val' / 'test' / 'gap'.

    Rows in gap years (between train and val, between val and test) are
    labeled 'gap' and excluded from every stage.
    """
    y = df["year"]
    split = pd.Series("gap", index=df.index, dtype="object")
    split[y.between(*train_years)] = "train"
    split[y.between(*val_years)] = "val"
    split[y.between(*test_years)] = "test"
    return split


def make_temporal_cv_folds(
    train_df: pd.DataFrame,
    n_folds: int = config.N_CV_FOLDS,
) -> pd.Series:
    """Assign the last n_folds years of the training window as one-year
    validation folds (expanding-window CV).

    Returns a Series aligned with train_df.index: fold id (0..n_folds-1) for
    rows that serve as a fold's validation year, -1 otherwise. Downstream
    code uses `iter_cv_folds` to recover (train, val) index pairs with the
    appropriate gap applied.
    """
    years = sorted(train_df["year"].unique())
    if len(years) < n_folds + config.CV_GAP + 1:
        raise ValueError(
            f"Training window has {len(years)} years; need at least "
            f"{n_folds + config.CV_GAP + 1} for {n_folds} folds."
        )
    val_years = years[-n_folds:]
    fold = pd.Series(-1, index=train_df.index, dtype="int8")
    for k, vy in enumerate(val_years):
        fold[train_df["year"] == vy] = k
    return fold


def iter_cv_folds(splits: pd.DataFrame, gap: int = config.CV_GAP):
    """Yield (train_idx, val_idx) pairs for each CV fold.

    `splits` must have columns 'year', 'split', 'cv_fold'. For each fold k,
    train_idx = outer-train rows with year < (fold k's val year) - gap.
    """
    fold_ids = sorted(splits.loc[splits["cv_fold"] >= 0, "cv_fold"].unique())
    for k in fold_ids:
        val_idx = splits.index[splits["cv_fold"] == k]
        val_year = splits.loc[val_idx, "year"].min()
        train_idx = splits.index[
            (splits["split"] == "train") & (splits["year"] < val_year - gap)
        ]
        yield int(k), train_idx, val_idx
