"""Produce temporal train/val/test splits and gapped expanding-window CV folds.

Reads:  data/processed_data/df_clean.csv
Writes: data/processed_data/splits.csv  (columns: id_rssd, year, split, cv_fold)

Downstream scripts merge this file onto df_clean.csv on (id_rssd, year).
"""
import sys
from pathlib import Path

# Make `from src import ...` work when running `python code/05_splits.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src import config
from src.splits import (
    assign_outer_split,
    make_temporal_cv_folds,
    sanity_check,
)


def main() -> None:
    for label in config.LABELS:
        print(f"\n=======================================================")
        print(f"Generating splits for target: {label}")
        print(f"=======================================================")
        
        clean_csv = config.get_clean_csv(label)
        splits_csv = config.get_splits_csv(label)
        
        print(f"Reading {clean_csv}")
        df = pd.read_csv(
            clean_csv,
            dtype={"id_rssd": "int64", "year": "int16"},
        )
        print(f"  {len(df):,} rows, years {df['year'].min()}–{df['year'].max()}")

        # --- Deduplicate (id_rssd, year) rows ---
        _growth_cols = [c for c in [
            "asset_growth", "deposit_growth", "loan_growth",
            "liab_growth", "equity_growth",
            "capital_ratio_chg", "roa_chg", "roe_chg",
            "nim_chg", "cost_of_funding_chg",
        ] if c in df.columns]

        n_before = len(df)
        if df.duplicated(subset=config.ID_COLS).any():
            df["_dedup_score"] = df[_growth_cols].abs().sum(axis=1)
            df = (
                df.sort_values("_dedup_score", ascending=False)
                  .drop_duplicates(subset=config.ID_COLS, keep="first")
                  .drop(columns="_dedup_score")
                  .sort_index()
            )
            print(f"  Dropped {n_before - len(df):,} duplicate (id_rssd, year) rows "
                  f"→ {len(df):,} rows remain")

        sanity_check(df)

        leaked = [c for c in config.LABEL_COMPONENT_COLS
                  if c in df.columns and c != label]
        if leaked:
            print(f"  WARNING: label-component columns present in dataset; "
                  f"training scripts must drop them: {leaked}")

        # Outer split
        split = assign_outer_split(df)
        print("\nOuter split row counts:")
        print(split.value_counts().to_string())

        print("\nPositive rate per split:")
        rates = df.groupby(split)[label].mean().rename(f"{label}_rate")
        print(rates.to_string())

        # Temporal CV folds within the training window
        train_mask = split == "train"
        cv_fold = pd.Series(-1, index=df.index, dtype="int8")
        cv_fold.loc[train_mask] = make_temporal_cv_folds(df.loc[train_mask])

        # Persist assignments
        out = df[config.ID_COLS].copy()
        out["split"] = split.values
        out["cv_fold"] = cv_fold.values

        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out.to_csv(splits_csv, index=False)
        print(f"\nWrote {splits_csv} ({len(out):,} rows)")

        # Summary of CV folds
        print("\nCV folds (validation year per fold):")
        cv_summary = (
            out[out["cv_fold"] >= 0]
            .groupby("cv_fold")["year"]
            .agg(val_year="min", n_rows="count")
        )
        print(cv_summary.to_string())


if __name__ == "__main__":
    main()
