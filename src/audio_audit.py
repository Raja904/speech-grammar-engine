"""Audio and Tabular Data Audit Engine.

Performs a comprehensive, non-destructive audit of competition CSVs and WAV recordings,
identifying signal anomalies, clipping, silence, duration distribution, column types,
target distribution, and sample submission alignment without altering any data.
"""

from __future__ import annotations

import logging
import math
import os
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data import ColumnMapping, detect_columns, resolve_audio_path

logger = logging.getLogger(__name__)

# Try importing soundfile; fallback gracefully to wave module if missing
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


@dataclass
class AudioFileAudit:
    """Individual audio file audit record."""
    filename: str
    filepath: str
    exists: bool
    corrupt: bool = False
    error_msg: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration_sec: Optional[float] = None
    peak_amplitude: Optional[float] = None
    rms_amplitude: Optional[float] = None
    clipped_sample_count: int = 0
    clipping_ratio: float = 0.0
    silence_ratio: float = 0.0


@dataclass
class AudioSetAudit:
    """Aggregated audit statistics for an audio set (train or test)."""
    total_expected: int
    total_found: int
    missing_count: int
    corrupt_count: int
    sample_rate_counts: Dict[int, int] = field(default_factory=dict)
    channel_counts: Dict[int, int] = field(default_factory=dict)
    duration_stats: Dict[str, float] = field(default_factory=dict)
    clipping_files_count: int = 0
    high_silence_files_count: int = 0
    anomalous_files: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TargetAudit:
    """Target variable statistical distribution audit."""
    target_column_name: str
    count: int
    min: float
    max: float
    mean: float
    std: float
    median: float
    quantiles: Dict[str, float] = field(default_factory=dict)
    out_of_bounds_count: int = 0  # < 0.0 or > 5.0
    unique_value_count: int = 0
    is_discrete: bool = False


@dataclass
class TabularAudit:
    """Tabular schema, missing values, duplicates, and column alignment audit."""
    train_shape: Tuple[int, int]
    test_shape: Tuple[int, int]
    train_columns: List[str]
    test_columns: List[str]
    train_dtypes: Dict[str, str]
    test_dtypes: Dict[str, str]
    train_missing: Dict[str, int]
    test_missing: Dict[str, int]
    train_duplicate_filenames: int
    test_duplicate_filenames: int
    train_test_filename_overlap: int
    speaker_metadata: Optional[Dict[str, Any]] = None
    sample_submission_check: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    """Unified data audit report capturing full dataset state."""
    tabular: TabularAudit
    target: Optional[TargetAudit]
    train_audio: AudioSetAudit
    test_audio: AudioSetAudit
    review_items: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Render a formatted, human-readable markdown summary report."""
        lines: List[str] = []
        lines.append("# Kaggle Spoken Grammar Scoring — Dataset Audit Report")
        lines.append("")

        # Review items alert block
        if self.review_items:
            lines.append("## ⚠️ Key Findings Requiring Review")
            for item in self.review_items:
                lines.append(f"- **{item}**")
            lines.append("")

        # 1. Tabular Audit
        lines.append("## 1. Tabular Dataset Structure")
        lines.append(f"- **Train CSV Shape**: {self.tabular.train_shape[0]} rows, {self.tabular.train_shape[1]} columns")
        lines.append(f"- **Test CSV Shape**: {self.tabular.test_shape[0]} rows, {self.tabular.test_shape[1]} columns")
        lines.append(f"- **Train Columns**: `{', '.join(self.tabular.train_columns)}`")
        lines.append(f"- **Test Columns**: `{', '.join(self.tabular.test_columns)}`")
        lines.append(f"- **Train Missing Values**: {self.tabular.train_missing}")
        lines.append(f"- **Test Missing Values**: {self.tabular.test_missing}")
        lines.append(f"- **Filename Overlap (Train vs Test)**: {self.tabular.train_test_filename_overlap}")

        if self.tabular.speaker_metadata:
            sm = self.tabular.speaker_metadata
            lines.append(f"- **Speaker Metadata Detected**: Column `{sm.get('column')}` ({sm.get('unique_speakers')} unique speakers)")

        sub_chk = self.tabular.sample_submission_check
        lines.append("")
        lines.append("### Sample Submission Alignment")
        lines.append(f"- Submission Valid: **{sub_chk.get('valid', False)}**")
        lines.append(f"- Row count matches test: {sub_chk.get('row_count_match', False)} ({sub_chk.get('submission_rows')} rows)")
        lines.append(f"- ID alignment with test: {sub_chk.get('id_alignment_match', False)}")

        # 2. Target Distribution
        if self.target:
            lines.append("")
            lines.append("## 2. Target Distribution (`{}`)".format(self.target.target_column_name))
            lines.append(f"- Count: {self.target.count}")
            lines.append(f"- Range: [{self.target.min:.4f}, {self.target.max:.4f}] (Expected [0, 5])")
            lines.append(f"- Mean ± Std: {self.target.mean:.4f} ± {self.target.std:.4f}")
            lines.append(f"- Median: {self.target.median:.4f}")
            lines.append(f"- Out of Bounds (< 0 or > 5): {self.target.out_of_bounds_count}")
            lines.append(f"- Quantiles: {', '.join(f'{k}: {v:.3f}' for k, v in self.target.quantiles.items())}")

        # 3. Audio Audit
        for name, audio_audit in [("Train Audio", self.train_audio), ("Test Audio", self.test_audio)]:
            lines.append("")
            lines.append(f"## 3. {name} Characteristics")
            lines.append(f"- Total Expected: {audio_audit.total_expected}")
            lines.append(f"- Found on Disk: {audio_audit.total_found}")
            lines.append(f"- Missing Files: {audio_audit.missing_count}")
            lines.append(f"- Corrupt Files: {audio_audit.corrupt_count}")
            lines.append(f"- Sample Rate Counts: {audio_audit.sample_rate_counts}")
            lines.append(f"- Channel Counts: {audio_audit.channel_counts}")
            if audio_audit.duration_stats:
                ds = audio_audit.duration_stats
                lines.append(
                    f"- Duration (seconds): Min {ds.get('min', 0):.2f}s, Max {ds.get('max', 0):.2f}s, "
                    f"Mean {ds.get('mean', 0):.2f}s ± {ds.get('std', 0):.2f}s"
                )
            lines.append(f"- Files with Clipping: {audio_audit.clipping_files_count}")
            lines.append(f"- Files with >50% Silence: {audio_audit.high_silence_files_count}")

            if audio_audit.anomalous_files:
                lines.append(f"- Anomalous Files Flagged: {len(audio_audit.anomalous_files)}")
                for anom in audio_audit.anomalous_files[:5]:
                    lines.append(f"  - `{anom['filename']}`: {anom['issues']}")
                if len(audio_audit.anomalous_files) > 5:
                    lines.append(f"  - *(...and {len(audio_audit.anomalous_files) - 5} more)*")

        return "\n".join(lines)


def audit_single_audio_file(
    filepath: Path,
    silence_threshold_db: float = -40.0,
    clipping_threshold: float = 0.999,
) -> AudioFileAudit:
    """Inspect a single audio WAV recording for sample rate, duration, clipping, and silence."""
    fname = filepath.name
    if not filepath.exists():
        return AudioFileAudit(filename=fname, filepath=str(filepath), exists=False)

    # 1. Try reading with SoundFile if available
    if HAS_SOUNDFILE:
        try:
            data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
            channels = data.shape[1]
            samples = len(data)
            duration = samples / float(sr)

            # Signal calculations
            peak_amp = float(np.max(np.abs(data))) if samples > 0 else 0.0
            rms_amp = float(np.sqrt(np.mean(data**2))) if samples > 0 else 0.0

            # Clipping check
            clipped_samples = int(np.sum(np.abs(data) >= clipping_threshold))
            clipping_ratio = float(clipped_samples / (samples * channels)) if samples > 0 else 0.0

            # Silence check using RMS threshold in dB
            # Frame-level RMS in chunks of 50ms
            frame_size = max(1, int(sr * 0.05))
            n_frames = samples // frame_size
            silence_ratio = 0.0
            if n_frames > 0:
                reshaped = data[: n_frames * frame_size].reshape(n_frames, frame_size, channels)
                frame_rms = np.sqrt(np.mean(reshaped**2, axis=(1, 2)))
                # Guard against log(0)
                frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-9))
                silent_frames = np.sum(frame_db < silence_threshold_db)
                silence_ratio = float(silent_frames / n_frames)

            return AudioFileAudit(
                filename=fname,
                filepath=str(filepath),
                exists=True,
                sample_rate=sr,
                channels=channels,
                duration_sec=duration,
                peak_amplitude=peak_amp,
                rms_amplitude=rms_amp,
                clipped_sample_count=clipped_samples,
                clipping_ratio=clipping_ratio,
                silence_ratio=silence_ratio,
            )
        except Exception as e:
            logger.warning(f"SoundFile failed to read {filepath}: {e}")
            # Try wave fallback below

    # 2. Standard library wave module fallback
    try:
        with wave.open(str(filepath), "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            duration = frames / float(sr) if sr > 0 else 0.0

            # Read raw frames to compute peak/clipping if reasonable size
            raw_data = wf.readframes(min(frames, sr * 60))
            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            if sampwidth in dtype_map:
                arr = np.frombuffer(raw_data, dtype=dtype_map[sampwidth]).astype(np.float32)
                max_val = float(2 ** (8 * sampwidth - 1))
                norm_arr = arr / max_val
                peak_amp = float(np.max(np.abs(norm_arr))) if len(norm_arr) > 0 else 0.0
                rms_amp = float(np.sqrt(np.mean(norm_arr**2))) if len(norm_arr) > 0 else 0.0
                clipped_samples = int(np.sum(np.abs(norm_arr) >= clipping_threshold))
                clipping_ratio = float(clipped_samples / len(norm_arr)) if len(norm_arr) > 0 else 0.0
            else:
                peak_amp, rms_amp, clipped_samples, clipping_ratio = 0.0, 0.0, 0, 0.0

            return AudioFileAudit(
                filename=fname,
                filepath=str(filepath),
                exists=True,
                sample_rate=sr,
                channels=channels,
                duration_sec=duration,
                peak_amplitude=peak_amp,
                rms_amplitude=rms_amp,
                clipped_sample_count=clipped_samples,
                clipping_ratio=clipping_ratio,
                silence_ratio=0.0,
            )
    except Exception as e:
        return AudioFileAudit(
            filename=fname,
            filepath=str(filepath),
            exists=True,
            corrupt=True,
            error_msg=str(e),
        )


def audit_audio_set(
    filenames: List[str],
    search_dirs: List[Path],
    silence_threshold_db: float = -40.0,
    clipping_threshold: float = 0.999,
    max_samples: Optional[int] = None,
) -> AudioSetAudit:
    """Run audio property audit across a list of expected audio recordings."""
    total_expected = len(filenames)
    audited_files: List[AudioFileAudit] = []

    to_audit = filenames[:max_samples] if max_samples else filenames

    for fname in to_audit:
        resolved = resolve_audio_path(fname, search_dirs)
        if resolved is None:
            audited_files.append(AudioFileAudit(filename=fname, filepath="", exists=False))
        else:
            audited_files.append(
                audit_single_audio_file(
                    resolved,
                    silence_threshold_db=silence_threshold_db,
                    clipping_threshold=clipping_threshold,
                )
            )

    found_files = [f for f in audited_files if f.exists]
    missing_count = sum(1 for f in audited_files if not f.exists)
    corrupt_files = [f for f in found_files if f.corrupt]
    corrupt_count = len(corrupt_files)

    valid_files = [f for f in found_files if not f.corrupt and f.sample_rate is not None]

    # Sample rates and channels
    sr_counts: Dict[int, int] = {}
    ch_counts: Dict[int, int] = {}
    durations: List[float] = []
    clipping_files_count = 0
    high_silence_files_count = 0
    anomalous: List[Dict[str, Any]] = []

    for f in valid_files:
        sr_counts[f.sample_rate] = sr_counts.get(f.sample_rate, 0) + 1
        ch_counts[f.channels] = ch_counts.get(f.channels, 0) + 1
        if f.duration_sec is not None:
            durations.append(f.duration_sec)

        issues = []
        if f.clipping_ratio > 0.001:  # More than 0.1% clipped samples
            clipping_files_count += 1
            issues.append(f"Clipping ({f.clipping_ratio:.2%})")
        if f.silence_ratio > 0.50:  # More than 50% silence
            high_silence_files_count += 1
            issues.append(f"High silence ({f.silence_ratio:.2%})")
        if f.duration_sec is not None and f.duration_sec < 0.5:
            issues.append(f"Very short duration ({f.duration_sec:.2f}s)")

        if issues:
            anomalous.append({"filename": f.filename, "issues": ", ".join(issues)})

    for cf in corrupt_files:
        anomalous.append({"filename": cf.filename, "issues": f"Corrupt: {cf.error_msg}"})

    duration_stats: Dict[str, float] = {}
    if durations:
        d_arr = np.array(durations)
        duration_stats = {
            "min": float(np.min(d_arr)),
            "max": float(np.max(d_arr)),
            "mean": float(np.mean(d_arr)),
            "std": float(np.std(d_arr)),
            "median": float(np.median(d_arr)),
            "q25": float(np.percentile(d_arr, 25)),
            "q75": float(np.percentile(d_arr, 75)),
        }

    return AudioSetAudit(
        total_expected=total_expected,
        total_found=len(found_files),
        missing_count=missing_count,
        corrupt_count=corrupt_count,
        sample_rate_counts=sr_counts,
        channel_counts=ch_counts,
        duration_stats=duration_stats,
        clipping_files_count=clipping_files_count,
        high_silence_files_count=high_silence_files_count,
        anomalous_files=anomalous,
    )


def audit_target(series: pd.Series, target_name: str) -> TargetAudit:
    """Compute statistical distribution, quantiles, and boundary checks on target variable."""
    clean = series.dropna().astype(float).values
    count = len(clean)
    if count == 0:
        raise ValueError("Cannot audit empty target series.")

    q_vals = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    quantiles_dict = {f"p{int(q*100)}": float(np.percentile(clean, q * 100)) for q in q_vals}

    out_of_bounds = int(np.sum((clean < 0.0) | (clean > 5.0)))
    unique_vals = len(np.unique(clean))
    is_discrete = unique_vals <= 20 and np.all(np.mod(clean, 0.25) == 0)

    return TargetAudit(
        target_column_name=target_name,
        count=count,
        min=float(np.min(clean)),
        max=float(np.max(clean)),
        mean=float(np.mean(clean)),
        std=float(np.std(clean)),
        median=float(np.median(clean)),
        quantiles=quantiles_dict,
        out_of_bounds_count=out_of_bounds,
        unique_value_count=unique_vals,
        is_discrete=is_discrete,
    )


def run_full_dataset_audit(
    train_csv_path: Path | str,
    test_csv_path: Path | str,
    sample_sub_path: Optional[Path | str] = None,
    audio_dirs_train: Optional[List[Path | str]] = None,
    audio_dirs_test: Optional[List[Path | str]] = None,
    silence_threshold_db: float = -40.0,
    clipping_threshold: float = 0.999,
    max_audio_samples: Optional[int] = None,
) -> AuditReport:
    """Execute complete end-to-end dataset audit."""
    train_p = Path(train_csv_path)
    test_p = Path(test_csv_path)

    if not train_p.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_p}")
    if not test_p.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_p}")

    train_df = pd.read_csv(train_p)
    test_df = pd.read_csv(test_p)

    train_mapping = detect_columns(train_df, is_test=False)
    test_mapping = detect_columns(test_df, is_test=True)

    # Missing values & duplicate checks
    train_missing = train_df.isna().sum().to_dict()
    test_missing = test_df.isna().sum().to_dict()

    train_fnames = train_df[train_mapping.filename_col].astype(str)
    test_fnames = test_df[test_mapping.filename_col].astype(str)

    train_dups = int(train_fnames.duplicated().sum())
    test_dups = int(test_fnames.duplicated().sum())
    overlap = int(len(set(train_fnames).intersection(set(test_fnames))))

    # Speaker / Group metadata check
    speaker_meta = None
    if train_mapping.speaker_col and train_mapping.speaker_col in train_df.columns:
        spk_series = train_df[train_mapping.speaker_col]
        speaker_meta = {
            "column": train_mapping.speaker_col,
            "unique_speakers": int(spk_series.nunique()),
            "recordings_per_speaker_mean": float(spk_series.value_counts().mean()),
            "recordings_per_speaker_max": int(spk_series.value_counts().max()),
        }

    # Sample submission validation
    sub_check: Dict[str, Any] = {"valid": False}
    if sample_sub_path:
        sub_p = Path(sample_sub_path)
        if sub_p.exists():
            sub_df = pd.read_csv(sub_p)
            row_match = len(sub_df) == len(test_df)
            sub_mapping = detect_columns(sub_df, is_test=True)
            id_align = False
            if sub_mapping.filename_col in sub_df.columns and test_mapping.filename_col in test_df.columns:
                id_align = (sub_df[sub_mapping.filename_col].astype(str).values == test_fnames.values).all()

            sub_check = {
                "valid": bool(row_match and id_align),
                "row_count_match": bool(row_match),
                "submission_rows": len(sub_df),
                "test_rows": len(test_df),
                "id_alignment_match": bool(id_align),
                "columns": list(sub_df.columns),
            }

    tabular_audit = TabularAudit(
        train_shape=train_df.shape,
        test_shape=test_df.shape,
        train_columns=list(train_df.columns),
        test_columns=list(test_df.columns),
        train_dtypes={col: str(dtype) for col, dtype in train_df.dtypes.items()},
        test_dtypes={col: str(dtype) for col, dtype in test_df.dtypes.items()},
        train_missing=train_missing,
        test_missing=test_missing,
        train_duplicate_filenames=train_dups,
        test_duplicate_filenames=test_dups,
        train_test_filename_overlap=overlap,
        speaker_metadata=speaker_meta,
        sample_submission_check=sub_check,
    )

    # Target audit
    target_audit = None
    if train_mapping.target_col and train_mapping.target_col in train_df.columns:
        target_audit = audit_target(train_df[train_mapping.target_col], train_mapping.target_col)

    # Audio audit
    dirs_train = [Path(d) for d in (audio_dirs_train or [train_p.parent / "train", train_p.parent])]
    dirs_test = [Path(d) for d in (audio_dirs_test or [test_p.parent / "test", test_p.parent])]

    train_audio_audit = audit_audio_set(
        filenames=list(train_fnames),
        search_dirs=dirs_train,
        silence_threshold_db=silence_threshold_db,
        clipping_threshold=clipping_threshold,
        max_samples=max_audio_samples,
    )

    test_audio_audit = audit_audio_set(
        filenames=list(test_fnames),
        search_dirs=dirs_test,
        silence_threshold_db=silence_threshold_db,
        clipping_threshold=clipping_threshold,
        max_samples=max_audio_samples,
    )

    # Synthesize items requiring engineer review
    review_items: List[str] = []

    if target_audit:
        if target_audit.out_of_bounds_count > 0:
            review_items.append(f"Target contains {target_audit.out_of_bounds_count} values outside [0, 5].")
        if target_audit.min < 0 or target_audit.max > 5:
            review_items.append(f"Target range [{target_audit.min:.2f}, {target_audit.max:.2f}] violates [0, 5].")

    if train_dups > 0:
        review_items.append(f"Train set contains {train_dups} duplicate filenames.")
    if overlap > 0:
        review_items.append(f"Train and Test sets share {overlap} identical filenames (leakage risk!).")

    if train_audio_audit.missing_count > 0:
        review_items.append(f"{train_audio_audit.missing_count} Train audio WAV files are missing from disk.")
    if test_audio_audit.missing_count > 0:
        review_items.append(f"{test_audio_audit.missing_count} Test audio WAV files are missing from disk.")

    if train_audio_audit.corrupt_count > 0:
        review_items.append(f"{train_audio_audit.corrupt_count} Train audio WAV files are corrupted.")
    if test_audio_audit.corrupt_count > 0:
        review_items.append(f"{test_audio_audit.corrupt_count} Test audio WAV files are corrupted.")

    if len(train_audio_audit.sample_rate_counts) > 1:
        review_items.append(f"Multiple sample rates in Train set: {train_audio_audit.sample_rate_counts}. Resampling required.")

    if len(train_audio_audit.channel_counts) > 1 or 2 in train_audio_audit.channel_counts:
        review_items.append(f"Stereo/multi-channel audio detected in Train set: {train_audio_audit.channel_counts}. Mono conversion required.")

    if speaker_meta:
        review_items.append(
            f"Speaker metadata detected ({speaker_meta['unique_speakers']} speakers). GroupKFold strongly advised to prevent leakage."
        )

    if not sub_check.get("valid", False):
        review_items.append("Sample submission failed alignment check with test dataset.")

    return AuditReport(
        tabular=tabular_audit,
        target=target_audit,
        train_audio=train_audio_audit,
        test_audio=test_audio_audit,
        review_items=review_items,
    )
