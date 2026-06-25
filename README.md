# india-runs-ai-ranker

**Team:** Rolelytix
**Challenge:** Redrob Hackathon — Intelligent Candidate Discovery & Ranking

A rule-based, explainable candidate ranking system that scores and ranks 100,000
candidate profiles against a Senior AI Engineer job description, without using
any external LLM API calls during ranking.

## What this does

Given `candidates.jsonl` (100K candidate profiles) and the job description's
must-have / preferred skill lists, this pipeline:

1. **Cleans and de-duplicates** the candidate pool, engineers features
   (total years of experience, role/title signals, career history stats),
   and parses the 23 behavioral signals from each candidate's `redrob_signals`.
2. **Scores** every candidate on skill match, experience relevance, and
   behavioral signals.
3. **Detects anomalies** — flags experience mismatches, behavioral outliers,
   skill-text inconsistencies, and computes a `fake_probability` / `trust_tier`
   per candidate, penalizing high-risk profiles before ranking.
4. **Ranks** the top 100 candidates and generates a plain-English, one-line
   reasoning for each — built from that candidate's own matched skills,
   experience, and trust signals (template-based, no external API calls).
5. **Validates** the output against the official submission format.

## Project structure

```
india-runs-ai-ranker/
├── rank.py                      # Main pipeline: Phase 1 (cleaning/features) + Phase 2 (scoring/ranking)
├── step1_preprocessing.py       # Data cleaning & feature engineering
├── step2_scoring.py             # Skill/experience/behavioral scoring
├── step3_trust_engine.py        # Anomaly detection & trust scoring
├── rewrite_reasoning.py         # Rewrites raw reasoning into plain-English sentences
├── validate_submission.py       # Official format validator (from hackathon bundle)
├── honeypot_selfcheck.py        # Self-check: estimates risky/honeypot rate in our own top 100
├── requirements.txt             # Python dependencies
├── submission_metadata.yaml     # Team & submission metadata
├── submission.csv               # Final output: top 100 ranked candidates
├── audit_results.md             # Evidence/audit trail for pipeline claims
├── data_quality_report.md       # Dataset quality audit
└── outputs/                     # Supporting spec & reference files
```

## Setup

```bash
git clone https://github.com/24A31A05EG/india-runs-ai-ranker.git
cd india-runs-ai-ranker
pip install -r requirements.txt
```

### Data placement (required, not included in repo due to size)

`candidates.jsonl` (~465 MB) is excluded from the repo via `.gitignore`.
Place it at the exact path below before running:

```
india-runs-ai-ranker/
└── [PUB] India_runs_data_and_ai_challenge/
    └── India_runs_data_and_ai_challenge/
        └── candidates.jsonl
```

This path is read from `CONFIG["data_path"]` at the top of `rank.py`.

## Reproduce the submission

```bash
python rank.py
python rewrite_reasoning.py
python validate_submission.py submission.csv
```

- `rank.py` reads `candidates.jsonl`, runs Phase 1 (cleaning/features) and
  Phase 2 (scoring/ranking/trust), and writes `submission.csv`.
- `rewrite_reasoning.py` rewrites the `reasoning` column into plain-English,
  fact-based sentences (pulls from each candidate's own matched skills and
  trust signals — no LLM calls, no invented facts).
- `validate_submission.py` checks the final CSV against the official format.

**Measured runtime (full 100K pool, CPU only, no GPU, no network):**
~25-30 seconds total (well within the 5-minute / 16GB RAM constraint).

## Sandbox / small-sample demo

A hosted Google Colab notebook runs this same pipeline end-to-end on a small
(50-candidate) sample: see `submission_metadata.yaml` → `sandbox_link`.

## Self-checks performed

- `validate_submission.py` → **passes** (correct headers, 100 unique ranks,
  non-increasing scores, valid candidate IDs).
- `honeypot_selfcheck.py` → estimated honeypot/fake-risk rate in our own
  top 100 is well under the 10% disqualification threshold, based on our own
  `fake_probability` and anomaly-flag signals.

## AI tool usage

Declared in `submission_metadata.yaml`. Claude was used for debugging,
format-compliance fixes, and designing the rule-based reasoning generator.
No candidate data was sent to any external LLM API as part of the ranking
pipeline itself.
