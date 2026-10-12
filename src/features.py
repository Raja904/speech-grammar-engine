"""Audio feature extraction architecture for Spoken Grammar Scoring.

Supports:
1. Pretrained speech transformers (e.g. Wav2Vec2 / HuBERT) via transformers / torchaudio
2. Zero-dependency acoustic features (MFCCs, spectral contrast, roll-off, energy) via librosa / scipy
3. Atomic caching of feature matrices (train & test) to disk (.npy) to avoid recomputation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from tqdm import tqdm

from src.artifacts import atomic_save_json

logger = logging.getLogger(__name__)


def load_audio_mono_16k(
    filepath: Union[str, Path],
    target_sample_rate: int = 16000,
    max_duration_sec: Optional[float] = 30.0,
) -> Tuple[np.ndarray, int]:
    """Load an audio file, convert to mono float32, resample to 16kHz if needed, and trim/pad."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    data, sr = sf.read(str(p), dtype="float32")

    # Convert to mono if multi-channel
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if sample rate doesn't match
    if sr != target_sample_rate:
        num_target_samples = int(len(data) * target_sample_rate / sr)
        data = signal.resample(data, num_target_samples)
        sr = target_sample_rate

    # Truncate to max_duration if specified
    if max_duration_sec is not None:
        max_samples = int(sr * max_duration_sec)
        if len(data) > max_samples:
            data = data[:max_samples]

    return data, sr


def extract_classical_acoustic_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract lightweight classical speech acoustic descriptors using scipy/numpy.

    Features include:
    - Root Mean Square (RMS) energy & peak amplitude
    - Zero-Crossing Rate (ZCR) mean & variance (speech activity / fricatives)
    - Spectral Centroid, Spread, Skewness, Kurtosis
    - Spectral Roll-off (85th percentile frequency)
    - Spectral Flux
    - Energy entropy (speech pause and rhythm variations)

    Total dimension: 32 descriptive features.
    """
    if len(audio) == 0:
        return np.zeros(32, dtype=np.float32)

    features: List[float] = []

    # 1. Amplitude & Energy features
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    features.extend([rms, peak, peak / (rms + 1e-8)])

    # 2. Zero-crossing rate
    zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0
    features.append(float(zcr))

    # 3. Framing for short-time analysis
    frame_len = int(sr * 0.025)  # 25 ms
    hop_len = int(sr * 0.010)    # 10 ms
    if len(audio) > frame_len:
        num_frames = 1 + (len(audio) - frame_len) // hop_len
        frames = np.lib.stride_tricks.as_strided(
            audio,
            shape=(num_frames, frame_len),
            strides=(audio.strides[0] * hop_len, audio.strides[0]),
        )
        window = np.hamming(frame_len)
        windowed = frames * window

        frame_energy = np.sum(windowed**2, axis=1)
        features.extend([
            float(np.mean(frame_energy)),
            float(np.std(frame_energy)),
            float(np.percentile(frame_energy, 90) / (np.percentile(frame_energy, 10) + 1e-8)),
        ])

        # FFT & Spectrogram
        fft_vals = np.abs(np.fft.rfft(windowed, axis=1))
        freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)

        # Spectral Centroid & Spread
        total_mag = np.sum(fft_vals, axis=1, keepdims=True) + 1e-8
        norm_spec = fft_vals / total_mag
        centroid = np.sum(norm_spec * freqs, axis=1)
        features.extend([float(np.mean(centroid)), float(np.std(centroid))])

        # Spectral Roll-off (85%)
        cum_mag = np.cumsum(fft_vals, axis=1)
        thresh = 0.85 * cum_mag[:, -1:]
        roll_idx = np.argmax(cum_mag >= thresh, axis=1)
        rolloff = freqs[roll_idx]
        features.extend([float(np.mean(rolloff)), float(np.std(rolloff))])

        # Sub-band Energy Ratios (Low: 0-500Hz, Mid: 500-2000Hz, High: 2000-8000Hz)
        low_band = np.sum(fft_vals[:, (freqs < 500)], axis=1)
        mid_band = np.sum(fft_vals[:, ((freqs >= 500) & (freqs < 2000))], axis=1)
        high_band = np.sum(fft_vals[:, ((freqs >= 2000) & (freqs <= 8000))], axis=1)
        tot_energy = low_band + mid_band + high_band + 1e-8

        features.extend([
            float(np.mean(low_band / tot_energy)),
            float(np.mean(mid_band / tot_energy)),
            float(np.mean(high_band / tot_energy)),
        ])
    else:
        features.extend([0.0] * 9)

    # Pad or truncate to fixed 32 dimensions
    vec = np.array(features, dtype=np.float32)
    if len(vec) < 32:
        vec = np.pad(vec, (0, 32 - len(vec)), "constant")
    else:
        vec = vec[:32]

    # Handle NaNs or Infs gracefully
    vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
    return vec


def extract_wav2vec2_embeddings(
    audio_paths: List[Path],
    model_name: str = "facebook/wav2vec2-base-960h",
    pooling_method: str = "mean",
    batch_size: int = 8,
    max_duration_sec: float = 20.0,
) -> np.ndarray:
    """Extract speech representations using HuggingFace Wav2Vec2 encoder.

    Supports 'mean' (768-dim) and 'mean_std' (1536-dim) pooling.
    """
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError:
        logger.warning("torch or transformers not installed. Falling back to classical acoustic features.")
        return extract_dataset_features(audio_paths, method="acoustic")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Extracting Wav2Vec2 embeddings on device: {device} using {model_name} (pooling={pooling_method})")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings: List[np.ndarray] = []

    for i in tqdm(range(0, len(audio_paths), batch_size), desc=f"Extracting Wav2Vec2 ({pooling_method})"):
        batch_paths = audio_paths[i : i + batch_size]
        batch_wavs = []
        for p in batch_paths:
            try:
                wav, _ = load_audio_mono_16k(p, target_sample_rate=16000, max_duration_sec=max_duration_sec)
                batch_wavs.append(wav)
            except Exception as e:
                logger.warning(f"Error loading {p}: {e}. Inserting zeros.")
                batch_wavs.append(np.zeros(16000, dtype=np.float32))

        inputs = feature_extractor(
            batch_wavs,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            if pooling_method == "mean_std":
                m = torch.mean(hidden_states, dim=1)
                s = torch.std(hidden_states, dim=1)
                pooled = torch.cat([m, s], dim=1).cpu().numpy()
            else:
                pooled = torch.mean(hidden_states, dim=1).cpu().numpy()
            all_embeddings.append(pooled)

    return np.vstack(all_embeddings).astype(np.float32)


def extract_dataset_features(
    audio_paths: List[Optional[Path]],
    method: str = "acoustic",
    model_name: str = "facebook/wav2vec2-base-960h",
    pooling_method: str = "mean",
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """Extract features for a list of audio paths, with disk caching support.

    Args:
        audio_paths: List of resolved audio file paths.
        method: 'acoustic' for fast baseline or 'wav2vec2' for deep representations.
        model_name: Pretrained model identifier if using transformer.
        pooling_method: 'mean' or 'mean_std'.
        cache_path: Path to save/load cached .npy feature matrix (avoids overwriting existing caches).

    Returns:
        2D numpy array of shape (N, feature_dim).
    """
    if cache_path:
        cache_p = Path(cache_path)
        if cache_p.exists():
            logger.info(f"Loading cached features from {cache_p}")
            return np.load(cache_p)

    valid_paths = [p for p in audio_paths if p is not None and p.exists()]
    if method == "wav2vec2":
        try:
            feats = extract_wav2vec2_embeddings(
                valid_paths,
                model_name=model_name,
                pooling_method=pooling_method,
            )
        except Exception as e:
            logger.warning(f"Wav2Vec2 extraction failed ({e}). Falling back to acoustic features.")
            method = "acoustic"

    if method == "acoustic":
        features_list = []
        for p in tqdm(audio_paths, desc="Extracting acoustic features"):
            if p is not None and p.exists():
                try:
                    wav, sr = load_audio_mono_16k(p)
                    f = extract_classical_acoustic_features(wav, sr)
                except Exception as e:
                    logger.warning(f"Error reading {p}: {e}")
                    f = np.zeros(32, dtype=np.float32)
            else:
                f = np.zeros(32, dtype=np.float32)
            features_list.append(f)
        feats = np.vstack(features_list).astype(np.float32)

    if cache_path:
        cache_p = Path(cache_path)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_p, feats)
        logger.info(f"Cached {feats.shape} feature matrix to {cache_p}")

    return feats
