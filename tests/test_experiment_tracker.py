"""Tests for experiment tracking and metric persistence."""

from pathlib import Path
import tempfile

from src.experiment_tracker import (
    ExperimentRecord,
    ExperimentTracker,
    initialize_baseline_log,
)


def test_experiment_tracker_lifecycle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = ExperimentTracker(tmp_dir)
        assert len(tracker.load_records()) == 0

        # Log a record
        rec = ExperimentRecord(
            experiment_id="test_exp_1",
            model_name="Ridge",
            feature_family="wav2vec2",
            pooling_method="mean",
            feature_dim=768,
            oof_rmse=0.82,
            oof_pearson_r=0.75,
            oof_mae=0.66,
            delta_rmse_vs_baseline=0.0,
        )
        tracker.log_experiment(rec)

        loaded = tracker.load_records()
        assert len(loaded) == 1
        assert loaded[0]["experiment_id"] == "test_exp_1"
        assert tracker.csv_path.exists()


def test_initialize_baseline_log():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = initialize_baseline_log(tmp_dir)
        records = tracker.load_records()
        assert len(records) == 5
        # Verify first baseline is Mean Baseline
        assert records[0]["model_name"] == "Mean Baseline"
        # Verify second is Ridge
        assert records[1]["model_name"] == "Ridge Regression"
        assert records[1]["oof_rmse"] == 0.821022
