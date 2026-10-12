"""Tests for audio feature extraction."""

from pathlib import Path
import tempfile
import numpy as np

from src.features import (
    extract_classical_acoustic_features,
    load_audio_mono_16k,
    extract_dataset_features,
)
from tests.test_audio_audit import _create_synthetic_wav


def test_extract_classical_acoustic_features():
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "test.wav"
        _create_synthetic_wav(wav_path, sample_rate=16000, duration_sec=0.2)

        audio, sr = load_audio_mono_16k(wav_path, target_sample_rate=16000)
        assert sr == 16000
        assert len(audio) == int(16000 * 0.2)

        feat = extract_classical_acoustic_features(audio, sr)
        assert isinstance(feat, np.ndarray)
        assert feat.shape == (32,)
        assert not np.isnan(feat).any()


def test_extract_dataset_features_with_cache():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        wav1 = tmp_p / "w1.wav"
        wav2 = tmp_p / "w2.wav"
        _create_synthetic_wav(wav1, sample_rate=16000, duration_sec=0.1)
        _create_synthetic_wav(wav2, sample_rate=16000, duration_sec=0.1)

        cache_p = tmp_p / "feat_cache.npy"
        feats = extract_dataset_features([wav1, wav2], method="acoustic", cache_path=cache_p)
        assert feats.shape == (2, 32)
        assert cache_p.exists()

        # Load from cache
        feats_loaded = extract_dataset_features([wav1, wav2], method="acoustic", cache_path=cache_p)
        assert np.array_equal(feats, feats_loaded)
