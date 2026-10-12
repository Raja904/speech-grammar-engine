"""Tests for audio signal auditing, target distribution checks, and dataset verification."""

import math
import struct
import wave
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.audio_audit import (
    audit_single_audio_file,
    audit_target,
    run_full_dataset_audit,
)


def _create_synthetic_wav(
    filepath: Path,
    sample_rate: int = 16000,
    duration_sec: float = 0.5,
    frequency: float = 440.0,
    amplitude: float = 0.5,
    is_silent: bool = False,
    is_clipped: bool = False,
) -> None:
    """Generate a clean synthetic 16-bit PCM mono WAV file using stdlib wave."""
    n_samples = int(sample_rate * duration_sec)
    max_amp = 32767.0

    frames = bytearray()
    for i in range(n_samples):
        if is_silent:
            val = 0
        elif is_clipped:
            # Saturated clipped value
            val = int(max_amp)
        else:
            val = int(amplitude * max_amp * math.sin(2 * math.pi * frequency * i / sample_rate))
        frames.extend(struct.pack("<h", max(-32768, min(32767, val))))

    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)


def test_audit_single_audio_file_clean():
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "clean.wav"
        _create_synthetic_wav(wav_path, sample_rate=16000, duration_sec=0.5, amplitude=0.5)

        audit = audit_single_audio_file(wav_path)
        assert audit.exists is True
        assert audit.corrupt is False
        assert audit.sample_rate == 16000
        assert audit.channels == 1
        assert pytest.approx(audit.duration_sec, 0.05) == 0.5
        assert audit.clipping_ratio < 0.001


def test_audit_single_audio_file_clipped():
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "clipped.wav"
        _create_synthetic_wav(wav_path, sample_rate=16000, duration_sec=0.2, is_clipped=True)

        audit = audit_single_audio_file(wav_path)
        assert audit.exists is True
        assert audit.clipped_sample_count > 0
        assert audit.clipping_ratio > 0.90


def test_audit_single_audio_file_corrupt():
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "corrupt.wav"
        wav_path.write_bytes(b"NOT_A_VALID_RIFF_HEADER_123456789")

        audit = audit_single_audio_file(wav_path)
        assert audit.exists is True
        assert audit.corrupt is True
        assert audit.error_msg is not None


def test_audit_target_distribution():
    scores = pd.Series([1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 5.5])  # 5.5 is out of bounds
    audit = audit_target(scores, "score")
    assert audit.min == 1.5
    assert audit.max == 5.5
    assert audit.out_of_bounds_count == 1
    assert "p50" in audit.quantiles
    assert audit.median == 3.5


def test_run_full_dataset_audit_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        train_audio_dir = tmp_p / "train_audio"
        test_audio_dir = tmp_p / "test_audio"
        train_audio_dir.mkdir()
        test_audio_dir.mkdir()

        # Create audio files
        _create_synthetic_wav(train_audio_dir / "rec1.wav", sample_rate=16000)
        _create_synthetic_wav(train_audio_dir / "rec2.wav", sample_rate=16000)
        _create_synthetic_wav(test_audio_dir / "test1.wav", sample_rate=16000)

        # Create train.csv, test.csv, sample_submission.csv
        train_df = pd.DataFrame({
            "filename": ["rec1.wav", "rec2.wav"],
            "score": [3.0, 4.0],
            "speaker_id": ["spk1", "spk2"],
        })
        train_csv = tmp_p / "train.csv"
        train_df.to_csv(train_csv, index=False)

        test_df = pd.DataFrame({
            "filename": ["test1.wav"],
            "score": [-1],
        })
        test_csv = tmp_p / "test.csv"
        test_df.to_csv(test_csv, index=False)

        sub_df = pd.DataFrame({
            "filename": ["test1.wav"],
            "score": [3.0],
        })
        sub_csv = tmp_p / "sample_submission.csv"
        sub_df.to_csv(sub_csv, index=False)

        report = run_full_dataset_audit(
            train_csv_path=train_csv,
            test_csv_path=test_csv,
            sample_sub_path=sub_csv,
            audio_dirs_train=[train_audio_dir],
            audio_dirs_test=[test_audio_dir],
        )

        assert report.tabular.train_shape == (2, 3)
        assert report.tabular.test_shape == (1, 2)
        assert report.tabular.sample_submission_check["valid"] is True
        assert report.train_audio.total_found == 2
        assert report.test_audio.total_found == 1
        assert report.target is not None
        assert report.target.mean == 3.5

        # Test Markdown rendering
        md_text = report.to_markdown()
        assert "# Kaggle Spoken Grammar Scoring — Dataset Audit Report" in md_text
        assert "Train CSV Shape" in md_text


def test_run_full_dataset_audit_mismatched_submission_shape():
    """Ensure mismatched row count between submission and test does not trigger broadcast ValueError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        train_df = pd.DataFrame({"filename": ["rec1.wav"], "score": [3.0]})
        test_df = pd.DataFrame({"filename": ["t1.wav", "t2.wav"], "score": [-1, -1]})
        sub_df = pd.DataFrame({"filename": ["t1.wav"], "score": [3.0]})

        train_csv = tmp_p / "train.csv"
        test_csv = tmp_p / "test.csv"
        sub_csv = tmp_p / "sub.csv"
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        sub_df.to_csv(sub_csv, index=False)

        report = run_full_dataset_audit(
            train_csv_path=train_csv,
            test_csv_path=test_csv,
            sample_sub_path=sub_csv,
        )
        assert report.tabular.sample_submission_check["valid"] is False
        assert report.tabular.sample_submission_check["row_count_match"] is False
        assert report.tabular.sample_submission_check["submission_rows"] == 1
        assert report.tabular.sample_submission_check["test_rows"] == 2
        assert report.tabular.sample_submission_check["id_alignment_match"] is False

