"""Tests for atomic file persistence, manifest tracking, and baseline model."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.artifacts import (
    ManifestManager,
    atomic_save_json,
    atomic_save_numpy,
    atomic_write_text,
    load_json,
)
from src.models import MeanBaselineModel


def test_atomic_write_text():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = Path(tmp_dir) / "sub" / "test.txt"
        saved = atomic_write_text(dest, "hello world")
        assert saved.exists()
        assert saved.read_text(encoding="utf-8") == "hello world"


def test_atomic_save_json():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = Path(tmp_dir) / "data.json"
        saved = atomic_save_json(dest, {"key": 123, "val": "abc"})
        assert saved.exists()
        loaded = load_json(dest)
        assert loaded["key"] == 123
        assert loaded["val"] == "abc"


def test_atomic_save_numpy():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = Path(tmp_dir) / "array.npy"
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        saved = atomic_save_numpy(dest, arr)
        assert saved.exists()
        loaded_arr = np.load(dest)
        np.testing.assert_array_equal(arr, loaded_arr)


def test_manifest_manager_lifecycle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_p = Path(tmp_dir) / "manifest.json"
        mgr = ManifestManager(manifest_p)

        # Register cache entry
        cache_file = Path(tmp_dir) / "feat.npy"
        atomic_save_numpy(cache_file, np.zeros((10, 128)))
        mgr.register_cache_entry(
            cache_id="wav2vec2_train",
            feature_name="wav2vec2",
            shape=[10, 128],
            dtype="float32",
            file_path=cache_file,
        )

        assert mgr.is_cache_valid("wav2vec2_train", expected_shape=[10, 128]) is True
        assert mgr.is_cache_valid("wav2vec2_train", expected_shape=[10, 256]) is False
        assert mgr.is_cache_valid("non_existent_cache") is False

        # Record run
        mgr.record_run(
            run_id="run_001",
            config_dict={"seed": 42},
            metrics={"rmse": 0.38},
        )
        assert "run_001" in mgr.data["runs"]
        assert mgr.data["runs"]["run_001"]["metrics"]["rmse"] == 0.38


def test_mean_baseline_model():
    model = MeanBaselineModel(strategy="mean")
    y_train = np.array([2.0, 3.0, 4.0])
    model.fit(None, y_train)
    assert model.baseline_value == 3.0

    X_test = np.zeros((4, 10))
    preds = model.predict(X_test)
    assert len(preds) == 4
    np.testing.assert_array_equal(preds, [3.0, 3.0, 3.0, 3.0])
