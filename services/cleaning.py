"""
services/cleaning.py — Autolysis v2
Optional automated data cleaning pipeline.
Activated only when the user explicitly enables the toggle in the UI.

Cleaning steps (in order):
1. Drop columns where >85% of values are missing
2. Impute numeric NaN with column median
3. Strip whitespace from all string columns
4. Detect and parse datetime-like string columns
5. Remove fully duplicate rows

Returns (cleaned_df, report_dict) so the LLM narrative can mention what was done.
"""

import pandas as pd
import numpy as np
import structlog
from typing import Tuple

logger = structlog.get_logger()

# Threshold: columns missing more than this fraction get dropped
MISSING_DROP_THRESHOLD = 0.85


def auto_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Run the automated cleaning pipeline on the dataframe.

    Returns:
        (cleaned_df, report)  where report is a dict with:
            - "actions": list[str] — human-readable list of what was done
            - "dropped_columns": list[str]
            - "imputed_columns": list[str]
            - "datetime_parsed": list[str]
            - "duplicates_removed": int
            - "rows_before": int
            - "rows_after": int
            - "cols_before": int
            - "cols_after": int
    """
    rows_before = len(df)
    cols_before = len(df.columns)
    actions = []
    dropped_columns = []
    imputed_columns = []
    datetime_parsed = []

    df = df.copy()

    # ── Step 1: Drop columns with too many missing values ──────────────────────
    missing_frac = df.isnull().mean()
    cols_to_drop = missing_frac[missing_frac > MISSING_DROP_THRESHOLD].index.tolist()
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        dropped_columns = cols_to_drop
        actions.append(
            f"Dropped {len(cols_to_drop)} column(s) with >{int(MISSING_DROP_THRESHOLD*100)}% missing: "
            f"{', '.join(cols_to_drop)}"
        )
        logger.info("cleaning_drop_columns", dropped=cols_to_drop)

    # ── Step 2: Strip whitespace from string columns ───────────────────────────
    str_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in str_cols:
        df[col] = df[col].str.strip()

    # ── Step 3: Parse datetime-like string columns ─────────────────────────────
    for col in str_cols:
        if col in df.columns:
            sample = df[col].dropna().head(20)
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() >= 0.8:  # 80%+ parseable → datetime column
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                    datetime_parsed.append(col)
                    actions.append(f"Parsed '{col}' as datetime")
                    logger.info("cleaning_datetime_parsed", col=col)
                except Exception:
                    pass

    # ── Step 4: Impute numeric NaN with column median ─────────────────────────
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            imputed_columns.append(col)
            actions.append(
                f"Imputed {null_count} missing value(s) in '{col}' with median ({median_val:.4g})"
            )
            logger.info("cleaning_imputed", col=col, null_count=int(null_count))

    # ── Step 5: Remove fully duplicate rows ───────────────────────────────────
    dupes = df.duplicated().sum()
    if dupes > 0:
        df.drop_duplicates(inplace=True)
        actions.append(f"Removed {dupes} fully duplicate row(s)")
        logger.info("cleaning_duplicates_removed", count=int(dupes))

    rows_after = len(df)
    cols_after = len(df.columns)

    report = {
        "actions": actions,
        "dropped_columns": dropped_columns,
        "imputed_columns": imputed_columns,
        "datetime_parsed": datetime_parsed,
        "duplicates_removed": int(dupes) if dupes > 0 else 0,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "cols_before": cols_before,
        "cols_after": cols_after,
    }

    if actions:
        logger.info(
            "cleaning_complete",
            actions_taken=len(actions),
            rows_removed=rows_before - rows_after,
            cols_removed=cols_before - cols_after,
        )
    else:
        logger.info("cleaning_no_changes_needed")

    return df, report
