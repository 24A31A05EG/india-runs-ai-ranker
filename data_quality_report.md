# Task 1: Data Understanding & Preprocessing — Final Report

## 1. Executive Summary
The preprocessing pipeline has successfully transformed 100,000 raw JSONL records into a clean, feature-rich Parquet dataset (`candidates_clean.parquet`) ready for scoring. The process included full schema discovery, JD spec audit, content-based duplicate detection, and vectorized feature extraction.

## 2. Ingestion & Quality Findings
- **Record Count**: 100,000 unique `candidate_id`s.
- **Exact Duplicates**: **5,680 records** (mapped to 2,694 content clusters) share identical resume text (Summary + Career Descriptions). These have been tagged with `is_duplicate_resume: True`.
- **Integrity Checks**:
    - **25 records** flagged with experience mismatches (Exp vs Career span > 5y).
    - **0 records** with overlapping current jobs.

## 3. Feature Engineering
- **Signal Normalization**: All 23 behavioral signals are cleaned. Sentinels (-1) converted to Null with companion flags.
- **Career Metrics**: Derived average tenure, job-switch rate, and career gaps.
- **Skill Normalization**: Skills are tokenized and normalized (e.g., `ML` -> `machine-learning`).
- **Assessments**: Sparse field `skill_assessment_scores` (24.24% population) summarized into mean score and count.

## 4. Final Deliverables
- **`outputs/job_requirements.json`**: Verified JD spec with source quotes.
- **`outputs/behavioral_signals_spec.json`**: 23-signal behavior mapping.
- **`candidates_clean.parquet`**: Analysis-ready feature table (100k x 50 columns).
- **`data_dictionary.md`**: Field-level documentation.
- **`cleaning_and_features.py`**: Reproducible feature extraction script.
