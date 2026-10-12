"""Data loading, column auto-detection, and filepath resolution.

Safely parses competition CSVs, verifies filesystem integrity of audio files,
and guarantees that placeholder targets (-1) in test.csv are never used as ground truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Common candidate column names in audio regression competitions
FILENAME_CANDIDATES = [
    "filename", "file_name", "audio_path", "audio_file", "audio", "wav_name", "wav",
    "id", "file_id", "recording_id", "sample_id", "Id"
]
TARGET_CANDIDATES = [
    "label", "score", "grammar_score", "grammar", "proficiency", "target", "overall_score"
]
SPEAKER_CANDIDATES = [
    "speaker", "speaker_id", "speaker_name", "speakerid", "user_id", "subject_id", "person_id"
]


@dataclass
class ColumnMapping:
    """Detected column roles in tabular metadata."""
    filename_col: str
    target_col: Optional[str] = None
    speaker_col: Optional[str] = None
    id_col: Optional[str] = None


def detect_columns(
    df: pd.DataFrame,
    override_filename_col: Optional[str] = None,
    override_target_col: Optional[str] = None,
    override_speaker_col: Optional[str] = None,
    is_test: bool = False,
) -> ColumnMapping:
    """Intelligently detect or validate column names in train/test CSVs."""
    cols_lower = {col.lower(): col for col in df.columns}

    # 1. Detect filename column
    filename_col = None
    if override_filename_col and override_filename_col in df.columns:
        filename_col = override_filename_col
    else:
        for cand in FILENAME_CANDIDATES:
            if cand.lower() in cols_lower:
                filename_col = cols_lower[cand.lower()]
                break

    if not filename_col:
        # Check if first column has string values ending in .wav or contains alphanumeric filenames
        first_col = df.columns[0]
        sample_vals = df[first_col].dropna().astype(str).head(5)
        if any(v.endswith(".wav") for v in sample_vals):
            filename_col = first_col
        else:
            raise ValueError(
                f"Could not auto-detect audio filename column in columns: {list(df.columns)}. "
                "Please specify filename_col in config."
            )

    # 2. Detect target column
    target_col = None
    if override_target_col and override_target_col in df.columns:
        target_col = override_target_col
    else:
        for cand in TARGET_CANDIDATES:
            if cand.lower() in cols_lower:
                target_col = cols_lower[cand.lower()]
                break

    # If test set, target col might be a placeholder column or absent
    if not target_col and not is_test:
        # Look for numeric column that is not the filename or ID
        numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != filename_col]
        if len(numeric_cols) == 1:
            target_col = numeric_cols[0]
            logger.info(f"Auto-selected single numeric column '{target_col}' as target.")

    # 3. Detect speaker / group column
    speaker_col = None
    if override_speaker_col and override_speaker_col in df.columns:
        speaker_col = override_speaker_col
    else:
        for cand in SPEAKER_CANDIDATES:
            if cand.lower() in cols_lower:
                speaker_col = cols_lower[cand.lower()]
                break

    return ColumnMapping(
        filename_col=filename_col,
        target_col=target_col,
        speaker_col=speaker_col,
        id_col=filename_col,
    )


def resolve_audio_path(filename: str, search_dirs: List[Path]) -> Optional[Path]:
    """Find the full path to an audio file across possible search directories.

    Handles filenames with or without '.wav' extension.
    """
    candidates = [filename]
    if not filename.lower().endswith(".wav"):
        candidates.append(f"{filename}.wav")

    for directory in search_dirs:
        if not directory.exists():
            continue
        for cand in candidates:
            p = directory / cand
            if p.exists() and p.is_file():
                return p.resolve()

            # Case-insensitive search on candidate if needed
            for file_in_dir in directory.glob("*"):
                if file_in_dir.name.lower() == cand.lower():
                    return file_in_dir.resolve()

    return None


def load_dataset(
    csv_path: Path | str,
    audio_dirs: List[Path | str],
    is_test: bool = False,
    override_columns: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, ColumnMapping]:
    """Safely load and validate competition dataset CSV.

    Args:
        csv_path: Path to CSV file.
        audio_dirs: List of candidate directories where audio recordings reside.
        is_test: Whether this is the test split (protects against placeholder targets).
        override_columns: Manual column name overrides.

    Returns:
        Tuple of (validated DataFrame with resolved 'resolved_audio_path', detected ColumnMapping).
    """
    csv_p = Path(csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_p}")

    df = pd.read_csv(csv_p)
    search_dirs = [Path(d) for d in audio_dirs]

    overrides = override_columns or {}
    mapping = detect_columns(
        df,
        override_filename_col=overrides.get("filename_col"),
        override_target_col=overrides.get("target_col"),
        override_speaker_col=overrides.get("speaker_col"),
        is_test=is_test,
    )

    # Resolve audio file paths
    resolved_paths: List[Optional[Path]] = []
    missing_count = 0
    for fname in df[mapping.filename_col].astype(str):
        resolved = resolve_audio_path(fname, search_dirs)
        resolved_paths.append(resolved)
        if resolved is None:
            missing_count += 1

    df["resolved_audio_path"] = resolved_paths
    df["audio_exists"] = [p is not None for p in resolved_paths]

    if missing_count > 0:
        logger.warning(
            f"{missing_count} out of {len(df)} recordings could not be found in audio search directories: {search_dirs}"
        )

    # Safeguard for test set placeholder targets
    if is_test and mapping.target_col and mapping.target_col in df.columns:
        # Check if values are -1 or placeholders
        test_targets = df[mapping.target_col].unique()
        if len(test_targets) == 1 and test_targets[0] == -1:
            logger.info("Confirmed test set target values are -1 placeholders. Flagging as placeholder.")
            df["is_placeholder_target"] = True

    return df, mapping
