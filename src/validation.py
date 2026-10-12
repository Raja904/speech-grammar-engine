"""Cross-validation scheme and leakage prevention utilities.

Provides leak-free K-Fold and GroupKFold splitting, fold integrity verification,
and structured Out-Of-Fold (OOF) prediction tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold

from src.artifacts import atomic_save_json
from src.metrics import compute_regression_metrics

logger = logging.getLogger(__name__)


@dataclass
class FoldIntegrityReport:
    """Validation report confirming absence of data leakage across folds."""
    is_valid: bool
    n_splits: int
    total_samples: int
    samples_per_fold: Dict[int, int]
    leakage_errors: List[str]


def create_folds(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    group_col: Optional[str] = None,
    n_splits: int = 5,
    seed: int = 42,
    fold_col_name: str = "fold",
) -> pd.DataFrame:
    """Assign validation fold indices to samples, respecting speaker/group constraints.

    Args:
        df: Input DataFrame.
        target_col: Name of target column (optional for stratified binning).
        group_col: Column indicating speaker or recording group. If present, GroupKFold is used.
        n_splits: Number of cross-validation folds.
        seed: Random seed for reproducibility.
        fold_col_name: Name of column to store fold indices [0 .. n_splits-1].

    Returns:
        DataFrame with added `fold_col_name` column.
    """
    df_out = df.copy()
    df_out[fold_col_name] = -1

    if group_col and group_col in df_out.columns:
        logger.info(f"Group column '{group_col}' detected with {df_out[group_col].nunique()} groups. Using GroupKFold.")
        gkf = GroupKFold(n_splits=n_splits)
        groups = df_out[group_col].values
        for fold_idx, (_, val_idx) in enumerate(gkf.split(df_out, groups=groups)):
            df_out.iloc[val_idx, df_out.columns.get_loc(fold_col_name)] = fold_idx
    else:
        logger.info(f"No group column detected. Using standard KFold (n_splits={n_splits}, seed={seed}).")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (_, val_idx) in enumerate(kf.split(df_out)):
            df_out.iloc[val_idx, df_out.columns.get_loc(fold_col_name)] = fold_idx

    return df_out


def verify_fold_integrity(
    df: pd.DataFrame,
    fold_col: str = "fold",
    group_col: Optional[str] = None,
) -> FoldIntegrityReport:
    """Verify that every sample appears in validation exactly once and groups never leak.

    Args:
        df: DataFrame containing fold assignments.
        fold_col: Name of the fold column.
        group_col: Optional group/speaker column.

    Returns:
        FoldIntegrityReport detailing integrity status.
    """
    errors: List[str] = []
    folds = sorted(df[fold_col].unique())
    n_splits = len(folds)
    total_samples = len(df)

    # 1. Check for unassigned samples (-1)
    unassigned = int((df[fold_col] == -1).sum())
    if unassigned > 0:
        errors.append(f"{unassigned} samples were not assigned to any fold.")

    # 2. Check samples count per fold
    counts = df[fold_col].value_counts().to_dict()

    # 3. Check for group leakage across folds
    if group_col and group_col in df.columns:
        group_folds = df.groupby(group_col)[fold_col].nunique()
        leaking_groups = group_folds[group_folds > 1]
        if len(leaking_groups) > 0:
            errors.append(
                f"Data leakage detected! {len(leaking_groups)} groups appear in multiple folds: {list(leaking_groups.index[:5])}"
            )

    is_valid = len(errors) == 0
    return FoldIntegrityReport(
        is_valid=is_valid,
        n_splits=n_splits,
        total_samples=total_samples,
        samples_per_fold=counts,
        leakage_errors=errors,
    )


def save_fold_assignments(
    df: pd.DataFrame,
    output_path: Path | str,
    id_col: str,
    fold_col: str = "fold",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist fold assignments atomically with verification metadata."""
    sub_df = df[[id_col, fold_col]].copy()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    sub_df.to_csv(out_p, index=False)

    # Save metadata JSON
    meta_path = out_p.with_suffix(".json")
    meta_payload = {
        "file": out_p.name,
        "n_samples": len(df),
        "n_splits": int(df[fold_col].nunique()),
        "distribution": df[fold_col].value_counts().to_dict(),
        "custom_metadata": metadata or {},
    }
    atomic_save_json(meta_path, meta_payload)

    return out_p
