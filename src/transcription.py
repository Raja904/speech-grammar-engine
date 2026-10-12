"""Whisper transcription and speech-to-text audit architecture for Spoken Grammar Scoring.

Implements:
1. Robust batch audio transcription using HuggingFace Whisper pipelines
2. Grammar and disfluency preservation inspection
3. Stratified quality auditing across the label score distribution
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.artifacts import atomic_save_json, atomic_write_text
from src.features import load_audio_mono_16k

logger = logging.getLogger(__name__)

DISFLUENCY_PATTERNS = [
    r"\b(um|uh|er|ah|hm|hmm|mm)\b",
    r"\b(\w+)\s+\1\b",  # Immediate word repetitions (e.g. "I I", "the the")
]


def load_whisper_pipeline(
    model_name: str = "openai/whisper-base",
    device: Optional[str] = None,
) -> Any:
    """Initialize HuggingFace ASR pipeline for Whisper."""
    try:
        import torch
        from transformers import pipeline

        if device is None:
            dev = 0 if torch.cuda.is_available() else -1
        else:
            dev = 0 if device == "cuda" and torch.cuda.is_available() else -1

        logger.info(f"Loading Whisper pipeline with model='{model_name}' on device={dev}")
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=dev,
            generate_kwargs={"language": "english", "task": "transcribe"},
        )
        return asr_pipe
    except ImportError:
        logger.error("transformers or torch not available for Whisper transcription.")
        raise


def transcribe_single_audio(
    pipeline_obj: Any,
    audio_path: Path | str,
    max_duration_sec: float = 30.0,
) -> Dict[str, Any]:
    """Transcribe a single audio file and extract linguistic/acoustic indicators."""
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    wav, sr = load_audio_mono_16k(p, target_sample_rate=16000, max_duration_sec=max_duration_sec)
    duration_sec = float(len(wav) / sr)

    # Whisper pipeline accepts dictionary {"raw": np.ndarray, "sampling_rate": int}
    result = pipeline_obj({"raw": wav, "sampling_rate": sr})
    raw_text = result.get("text", "").strip()

    # Analyze surface disfluencies and repetition
    lower_text = raw_text.lower()
    disfluency_hits = []
    for pat in DISFLUENCY_PATTERNS:
        matches = re.findall(pat, lower_text)
        if matches:
            disfluency_hits.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])

    words = raw_text.split()
    word_count = len(words)
    speaking_rate = float(word_count / (duration_sec + 1e-8)) * 60.0  # words per minute

    return {
        "filename": p.name,
        "filepath": str(p),
        "duration_sec": round(duration_sec, 2),
        "transcript": raw_text,
        "word_count": word_count,
        "words_per_minute": round(speaking_rate, 1),
        "disfluencies_detected": disfluency_hits,
        "disfluency_count": len(disfluency_hits),
    }


def select_stratified_audit_sample(
    df: pd.DataFrame,
    target_col: str,
    sample_size: int = 16,
    random_state: int = 42,
) -> pd.DataFrame:
    """Select stratified sample across the score distribution: Low, Mid-Low, Mid-High, High."""
    df_clean = df.dropna(subset=[target_col]).copy()
    n_bins = 4
    bins = [0.0, 2.0, 3.0, 4.0, 5.01]
    labels = ["Tier 1: Low (<=2.0)", "Tier 2: Mid-Low (2.0-3.0)", "Tier 3: Mid-High (3.0-4.0)", "Tier 4: High (>4.0)"]
    df_clean["score_tier"] = pd.cut(df_clean[target_col], bins=bins, labels=labels, right=True)

    samples_per_tier = max(1, sample_size // n_bins)
    sampled_dfs = []
    rng = np.random.RandomState(random_state)

    for tier in labels:
        tier_df = df_clean[df_clean["score_tier"] == tier]
        if len(tier_df) > 0:
            k = min(len(tier_df), samples_per_tier)
            sampled_dfs.append(tier_df.sample(n=k, random_state=random_state))

    result = pd.concat(sampled_dfs).reset_index(drop=True)
    return result


def run_whisper_transcription_audit(
    sample_df: pd.DataFrame,
    target_col: str,
    filename_col: str,
    model_name: str = "openai/whisper-base",
    output_dir: Path | str = "outputs",
) -> Dict[str, Any]:
    """Execute complete Whisper transcription audit on sample recordings and output reports."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_whisper_pipeline(model_name=model_name)
    audit_results: List[Dict[str, Any]] = []

    logger.info(f"Running Whisper transcription audit on {len(sample_df)} recordings...")

    for idx, row in sample_df.iterrows():
        p = Path(row["resolved_audio_path"])
        info = transcribe_single_audio(pipe, p)
        info["label"] = float(row[target_col])
        info["tier"] = str(row.get("score_tier", "Unknown"))
        audit_results.append(info)

    # Structural observations on Whisper's transcription behavior
    has_disfluencies = sum(1 for item in audit_results if item["disfluency_count"] > 0)
    avg_wpm = float(np.mean([item["words_per_minute"] for item in audit_results]))

    summary_findings = [
        f"Transcribed {len(audit_results)} audio samples across 4 proficiency tiers.",
        f"Surface disfluencies ('um'/'uh'/repetitions) detected in {has_disfluencies}/{len(audit_results)} transcripts.",
        f"Average speaking rate across audit sample: {avg_wpm:.1f} WPM.",
        "Observation on Grammar Preservation: Pretrained Whisper inherently tends to normalize punctuation and "
        "may smooth minor spoken grammatical disfluencies due to language model decoding prior. Direct manual inspection required.",
    ]

    report_payload = {
        "model_name": model_name,
        "n_samples": len(audit_results),
        "summary_findings": summary_findings,
        "sample_records": audit_results,
    }

    # Save JSON artifact
    json_path = out_dir / "whisper_transcription_audit_v1.json"
    atomic_save_json(json_path, report_payload)

    # Render Markdown report
    md_lines = [
        "# Whisper Transcription Quality Audit — Spoken Grammar Scoring",
        "",
        f"**Model:** `{model_name}` | **Samples Audited:** {len(audit_results)}",
        "",
        "## 1. Key Audit Findings & Observations",
    ]
    for find in summary_findings:
        md_lines.append(f"- {find}")

    md_lines.extend([
        "",
        "## 2. Sample Transcriptions by Proficiency Tier",
        "",
        "| Filename | Score | Tier | Duration (s) | WPM | Disfluencies | Transcript |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])

    for item in audit_results:
        clean_tx = item["transcript"].replace("|", "\\|")
        md_lines.append(
            f"| `{item['filename']}` | {item['label']:.2f} | {item['tier']} | {item['duration_sec']} | {item['words_per_minute']} | {item['disfluency_count']} | \"{clean_tx}\" |"
        )

    md_path = out_dir / "whisper_transcription_audit_v1.md"
    atomic_write_text(md_path, "\n".join(md_lines))

    logger.info(f"Saved transcription audit reports to {json_path} and {md_path}")
    return report_payload
