# Redrob Hackathon: Intelligent Candidate Discovery
**Stage:** Task 1 — Preprocessing & Data Understanding

## 🚀 Execution Summary
- **Current Script:** `01_exploration.py` (Full 100k pass complete)
- **Status:** Steps 1-2 VERIFIED.

## 📁 Project Structure
- `outputs/01_exploration.py` — Reproducible full-pass script.
- `outputs/job_requirements.json` — **Manually verified** JD spec.
- `outputs/behavioral_signals_spec.json` — Validated 23-signal spec.
- `data_quality_report.md` — Definitive audit of the full dataset.

## 🔍 Key Findings (Full Pass)
- **100,000 Records / 0 Exact Duplicates**: Clean ingestion verified via MD5.
- **100% Signal Presence**: All 23 behavioral keys are populated in the JSON structure.
- **JD Authority**: Disqualifiers (Consulting background, LangChain-only) and Requirements (NDCG/Vector DBs) confirmed.

---
*Next: Step 3 (EDA) — distribution analysis and shingling-based duplicate detection.*
