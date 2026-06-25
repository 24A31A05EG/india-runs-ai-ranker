"""
Rewrites the `reasoning` column in submission.csv into plain-English sentences
using actual candidate facts from candidates_with_trust.parquet.

Run from the repo root:
    python rewrite_reasoning.py

Reads:  submission.csv, candidates_with_trust.parquet
Writes: submission.csv (overwritten in place, old version is NOT kept)
"""

import re
import pandas as pd

SUB_PATH = "submission.csv"
PARQUET_PATH = "candidates_with_trust.parquet"

sub = pd.read_csv(SUB_PATH)
cand = pd.read_parquet(PARQUET_PATH)

# Keep only columns we need from the candidate pool, to keep the merge light
needed_cols = [
    "candidate_id", "headline", "total_years_exp", "skills_list",
    "sig_notice_period_days", "num_career_gaps", "fake_probability",
    "trust_tier", "sig_recruiter_response_rate",
]
cand = cand[[c for c in needed_cols if c in cand.columns]]

merged = sub.merge(cand, on="candidate_id", how="left")

def extract_must_haves(old_reasoning: str):
    """Pull the matched must-have skills out of the existing reasoning string."""
    if not isinstance(old_reasoning, str):
        return []
    m = re.search(r"Must-haves:\s*(.*?)\.\s*Skill:", old_reasoning)
    if not m:
        return []
    skills = [s.strip() for s in m.group(1).split(",") if s.strip()]
    return skills

def build_reasoning(row):
    skills = extract_must_haves(row.get("reasoning", ""))
    top_skills = skills[:3] if skills else []
    skill_phrase = ", ".join(top_skills) if top_skills else "core JD-listed skills"

    years = row.get("total_years_exp")
    years_phrase = f"{years:.1f} yrs experience" if pd.notna(years) else "experience not specified"

    headline = row.get("headline")
    headline_phrase = headline if isinstance(headline, str) and headline.strip() else "Candidate"

    # Pick ONE honest concern or strength, in priority order, using only real fields
    notice = row.get("sig_notice_period_days")
    gaps = row.get("num_career_gaps")
    fake_p = row.get("fake_probability")
    trust_tier = row.get("trust_tier")
    response_rate = row.get("sig_recruiter_response_rate")

    concern = None
    if pd.notna(notice) and notice > 60:
        concern = f"Notice period of {int(notice)} days may delay onboarding."
    elif pd.notna(gaps) and gaps > 0:
        concern = f"Has {int(gaps)} career gap(s) worth a closer look."
    elif pd.notna(fake_p) and fake_p > 0.05:
        concern = f"Minor trust flag (fake-probability {fake_p:.2f}); otherwise consistent profile."
    elif pd.notna(trust_tier):
        concern = f"Trust tier {trust_tier} with no major red flags."
    elif pd.notna(response_rate):
        concern = f"Recruiter response rate of {response_rate:.0%} indicates active engagement."
    else:
        concern = "No major concerns identified from available signals."

    return f"{headline_phrase} with {years_phrase}; strong match on {skill_phrase}. {concern}"

merged["reasoning"] = merged.apply(build_reasoning, axis=1)

# Keep exact required column order/names
final = merged[["candidate_id", "rank", "score", "reasoning"]]
final.to_csv(SUB_PATH, index=False)

print(f"Rewrote reasoning for {len(final)} rows -> {SUB_PATH}")
print(final.head(5).to_string())
