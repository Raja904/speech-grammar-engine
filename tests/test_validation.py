"""Tests for cross-validation fold splitting and leakage verification."""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.validation import create_folds, save_fold_assignments, verify_fold_integrity


def test_create_folds_standard():
    df = pd.DataFrame({
        "id": [f"id_{i}" for i in range(20)],
        "score": [2.0 + (i % 4) * 0.5 for i in range(20)],
    })
    df_folds = create_folds(df, n_splits=5, seed=42)
    assert "fold" in df_folds.columns
    assert sorted(df_folds["fold"].unique()) == [0, 1, 2, 3, 4]
    # Check fold distribution is even (20 / 5 = 4 samples per fold)
    counts = df_folds["fold"].value_counts().to_dict()
    for f in range(5):
        assert counts[f] == 4

    report = verify_fold_integrity(df_folds)
    assert report.is_valid is True
    assert len(report.leakage_errors) == 0


def test_create_folds_group_leakage_prevention():
    # 4 distinct speakers, each having 3 recordings
    speakers = ["spk1"] * 3 + ["spk2"] * 3 + ["spk3"] * 3 + ["spk4"] * 3
    df = pd.DataFrame({
        "id": [f"id_{i}" for i in range(12)],
        "speaker": speakers,
        "score": [3.0] * 12,
    })
    df_folds = create_folds(df, group_col="speaker", n_splits=4, seed=42)

    report = verify_fold_integrity(df_folds, group_col="speaker")
    assert report.is_valid is True

    # Intentionally cause a group leak to verify detector catches it
    df_leaked = df_folds.copy()
    df_leaked.loc[0, "fold"] = 1  # spk1 now appears in fold 0 and fold 1
    report_leaked = verify_fold_integrity(df_leaked, group_col="speaker")
    assert report_leaked.is_valid is False
    assert any("leakage detected" in err.lower() for err in report_leaked.leakage_errors)


def test_save_fold_assignments():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_csv = Path(tmp_dir) / "folds.csv"
        df = pd.DataFrame({"id": ["a", "b", "c"], "fold": [0, 1, 0]})
        saved_path = save_fold_assignments(df, out_csv, id_col="id")
        assert saved_path.exists()
        assert out_csv.with_suffix(".json").exists()
