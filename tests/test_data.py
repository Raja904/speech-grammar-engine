"""Tests for dataset loading, column detection, and placeholder handling."""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.data import detect_columns, load_dataset, resolve_audio_path


def test_detect_columns_standard():
    df = pd.DataFrame({
        "filename": ["rec1.wav", "rec2.wav"],
        "score": [3.5, 4.0],
        "speaker_id": ["spk_01", "spk_02"],
    })
    mapping = detect_columns(df)
    assert mapping.filename_col == "filename"
    assert mapping.target_col == "score"
    assert mapping.speaker_col == "speaker_id"


def test_detect_columns_variations():
    df = pd.DataFrame({
        "file_name": ["audio_01.wav", "audio_02.wav"],
        "grammar_score": [2.5, 4.5],
    })
    mapping = detect_columns(df)
    assert mapping.filename_col == "file_name"
    assert mapping.target_col == "grammar_score"
    assert mapping.speaker_col is None


def test_resolve_audio_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        test_file = tmp_p / "sample.wav"
        test_file.write_bytes(b"dummy")

        # Resolves with exact name
        found1 = resolve_audio_path("sample.wav", [tmp_p])
        assert found1 == test_file.resolve()

        # Resolves when extension is omitted in CSV
        found2 = resolve_audio_path("sample", [tmp_p])
        assert found2 == test_file.resolve()

        # Returns None when file missing
        found3 = resolve_audio_path("non_existent.wav", [tmp_p])
        assert found3 is None


def test_placeholder_target_handling_in_test_set():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        csv_path = tmp_p / "test.csv"
        df_test = pd.DataFrame({
            "filename": ["t1.wav", "t2.wav"],
            "score": [-1, -1],
        })
        df_test.to_csv(csv_path, index=False)

        loaded_df, mapping = load_dataset(csv_path, audio_dirs=[tmp_p], is_test=True)
        assert "is_placeholder_target" in loaded_df.columns
        assert loaded_df["is_placeholder_target"].all()
