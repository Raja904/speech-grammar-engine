"""Tests for transcription and stratified sampling."""

import numpy as np
import pandas as pd
import pytest

from src.transcription import select_stratified_audit_sample, DISFLUENCY_PATTERNS
import re


def test_select_stratified_audit_sample():
    # 40 synthetic samples with scores between 1.0 and 5.0
    df = pd.DataFrame({
        "filename": [f"file_{i}.wav" for i in range(40)],
        "score": np.linspace(1.0, 5.0, 40),
    })

    sampled = select_stratified_audit_sample(df, target_col="score", sample_size=16, random_state=42)
    assert len(sampled) == 16
    assert "score_tier" in sampled.columns
    # Check all tiers are represented
    assert sampled["score_tier"].nunique() == 4


def test_disfluency_detection_patterns():
    text = "um I think that uh the person is is walking"
    hits = []
    for pat in DISFLUENCY_PATTERNS:
        matches = re.findall(pat, text.lower())
        if matches:
            hits.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])

    assert "um" in hits
    assert "uh" in hits
