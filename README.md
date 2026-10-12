# Kaggle Spoken Grammar Scoring Engine

A reproducible, modular machine-learning pipeline for predicting continuous spoken English grammar proficiency scores ($0.0$ to $5.0$) from audio recordings.

---

## 1. Project Overview & Objectives

- **Task**: Audio regression predicting continuous grammar scores from spoken English WAV recordings.
- **Dataset**:
  - `769` labelled training recordings (`train.csv`).
  - `216` competition test recordings (`test.csv`, placeholder targets `-1.0`).
  - Required submission format: `sample_submission.csv`.
- **Evaluation Metrics**:
  - Root Mean Squared Error (**RMSE**)
  - Pearson Correlation Coefficient (**$r$**)
  - Mean Absolute Error (**MAE**)
- **Target Benchmark**: Aspirational Out-of-Fold (OOF) RMSE $\le 0.40$.

---

## 2. Repository Structure

```text
grammer_score_engine/
├── configs/
│   └── default.yaml             # Central configuration (paths, audio, CV, models)
├── src/
│   ├── __init__.py
│   ├── config.py                # Environment detection (Kaggle/Local) & typed configs
│   ├── data.py                  # Robust CSV loader, column auto-detection & path resolution
│   ├── audio_audit.py           # Signal property inspector (sample rate, clipping, silence)
│   ├── metrics.py               # Safe RMSE, Pearson correlation (zero-variance safe) & MAE
│   ├── validation.py            # Leak-free K-Fold / GroupKFold cross-validation & verification
│   ├── models.py                # Model interfaces, MeanBaselineModel, Ridge & SVR scaffolds
│   └── artifacts.py             # Atomic file writes, JSON/Numpy persistence & manifest manager
├── scripts/
│   └── audit_dataset.py         # Standalone CLI script to execute full dataset audit
├── notebooks/
│   └── grammar_scoring.ipynb    # Kaggle notebook deliverable with audit & experiment pipeline
├── tests/
│   ├── test_config.py           # Config loading & serialization tests
│   ├── test_metrics.py          # Metric calculations & numerical edge cases
│   ├── test_data.py             # Tabular loading & placeholder target safeguards
│   ├── test_audio_audit.py      # Audio inspection & synthetic signal tests
│   ├── test_validation.py       # Cross-validation splitting & leakage detection
│   └── test_artifacts.py        # Atomic persistence & manifest tests
├── requirements.txt             # Minimal Kaggle-compatible dependencies
├── .gitignore                   # Ignores raw audio, checkpoints, and caches
└── README.md                    # Project documentation & runbook
```

---

## 3. Local Development & Testing

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Unit Tests

Run the comprehensive unit test suite:

```bash
pytest tests/ -v
```

All tests execute on lightweight synthetic data without requiring real competition audio or downloading large pretrained models.

---

## 4. Connecting to Kaggle & Executing Phase 1

Because the actual competition audio recordings reside on Kaggle, follow these steps to connect the repository and run the data audit.

### Option A: Direct Git Clone in Kaggle Notebook (Recommended)

1. **Create a new Kaggle Notebook** attached to the competition dataset.
2. In the first code cell of the Kaggle notebook, clone your GitHub repository:
   ```python
   !git clone https://github.com/Raja904/speech-grammar-engine.git
   %cd speech-grammar-engine
   !pip install -r requirements.txt
   ```
3. Run the standalone audit script:
   ```bash
   !python scripts/audit_dataset.py
   ```
4. Or open and execute `notebooks/grammar_scoring.ipynb` directly.

### Option B: Kaggle Dataset as Code Utility

1. Bundle the `src/`, `configs/`, and `scripts/` directories as a private Kaggle Dataset (`spoken-grammar-engine-code`).
2. Attach both the **Competition Dataset** and **Code Utility Dataset** to your notebook.
3. In the notebook, append the code directory to `sys.path`:
   ```python
   import sys
   sys.path.append("/kaggle/input/spoken-grammar-engine-code")
   ```

---

## 5. Phase 1 Checklist: Items Requiring Review After Kaggle Audit

When `scripts/audit_dataset.py` or Section 2 of `notebooks/grammar_scoring.ipynb` completes on Kaggle, inspect `outputs/audit_report.md` and review the following critical findings:

| Checkpoint | What to Look For | Engineering Decision / Action |
| :--- | :--- | :--- |
| **Tabular Shapes & Columns** | Verify train has 769 rows and test has 216 rows. Check actual column names detected. | If column names differ from standard conventions, adjust `configs/default.yaml`. |
| **Placeholder Targets** | Confirm test target values are all `-1.0`. | Verify `is_placeholder_target` is flagged and test scores are never used in training. |
| **Speaker Metadata** | Look for speaker IDs or grouping columns (`speaker`, `user_id`). | If present, use **`GroupKFold`** to prevent speaker leakage across CV folds. |
| **Target Distribution** | Confirm target minimum $\ge 0.0$ and maximum $\le 5.0$. Check if discrete rubrics exist. | Verify target distribution skewness; check whether bounded clipping $[0, 5]$ is sufficient. |
| **Audio File Completeness** | Check if any WAV files are missing from disk or unreadable. | Log missing/corrupted files; ensure pipeline handles unreadable files safely. |
| **Sampling Rates** | Check if all files are 16 kHz or if diverse rates exist (44.1 kHz, 48 kHz). | If diverse rates exist, configure strict resampling to 16 kHz in Phase 2. |
| **Channel Count** | Check for stereo vs. mono recordings. | If stereo files exist, ensure downmixing to mono is applied. |
| **Audio Duration** | Review min, max, and quantile duration (seconds). | Decide whether long recordings need chunking/truncation for Wav2Vec2/HuBERT limits. |
| **Signal Quality (Clipping & Silence)** | Review count of files with clipping ($>0.1\%$) or silence ($>50\%$). | Verify that quiet recordings are not erroneously discarded or normalized without evidence. |
| **Sample Submission Alignment** | Check row count (216), column names, and row ordering. | Ensure final submission exporter precisely mirrors `sample_submission.csv`. |

---

## 6. Execution Roadmap

- **Phase 1 (Complete)**: Repository scaffold, typed configuration, safe data loading, audio audit engine, unit tests, notebook skeleton.
- **Phase 2 (Next)**: Conservative audio preprocessing (sample rate verification, mono conversion) and Mean Baseline benchmark.
- **Phase 3**: Audio embedding extraction (Wav2Vec2/HuBERT), persistent feature caching, and checkpoint recovery.
- **Phase 4**: 5-Fold cross-validation with Ridge regression, Out-of-Fold (OOF) evaluation, and residual diagnostics.
- **Phase 5**: Speech-to-text transcript generation, linguistic features, and comparative analysis.
- **Phase 6**: Model comparison, final model training, submission file validation, and artifact export.
