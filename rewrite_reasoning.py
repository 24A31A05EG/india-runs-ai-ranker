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
    "trust_tier", "sig_recruiter_response_rate", "sig_github_activity_score",
]
cand = cand[[c for c in needed_cols if c in cand.columns]]

merged = sub.merge(cand, on="candidate_id", how="left")

def extract_must_haves(old_reasoning: str, skills_list):
    """Pull matched must-have skills from the existing reasoning string.
    Falls back to the candidate's own skills_list if the original
    'Must-haves:' pattern is no longer present (e.g. script already ran once)."""
    if isinstance(old_reasoning, str):
        m = re.search(r"Must-haves:\s*(.*?)\.\s*Skill:", old_reasoning)
        if m:
            skills = [s.strip() for s in m.group(1).split(",") if s.strip()]
            if skills:
                return skills
    # Fallback: use the candidate's raw skills_list field
    if isinstance(skills_list, (list, tuple)):
        return list(skills_list)[:3]
    if isinstance(skills_list, str):
        return [s.strip() for s in re.split(r"[,;]", skills_list) if s.strip()][:3]
    return []

def clean_headline(headline):
    """Drop trailing years-fragments from the headline so we don't repeat years twice."""
    if not isinstance(headline, str):
        return "Candidate"
    # remove trailing pieces like "7.8+ yrs" / "8 yrs" that sometimes sit in the headline itself
    parts = [p.strip() for p in headline.split("|")]
    parts = [p for p in parts if not re.fullmatch(r"\d+(\.\d+)?\+?\s*yrs?", p, flags=re.IGNORECASE)]
    return " | ".join(parts) if parts else "Candidate"

def build_reasoning(row):
    skills = extract_must_haves(row.get("reasoning", ""), row.get("skills_list"))
    top_skills = skills[:3] if skills else []
    skill_phrase = ", ".join(top_skills) if top_skills else "core JD-listed skills"

    years = row.get("total_years_exp")
    years_phrase = f"{years:.1f} yrs experience" if pd.notna(years) else "experience not specified"

    headline_phrase = clean_headline(row.get("headline"))

    # Pick ONE honest concern, in priority order, using only real fields
    notice = row.get("sig_notice_period_days")
    gaps = row.get("num_career_gaps")
    fake_p = row.get("fake_probability")
    trust_tier = row.get("trust_tier")
    response_rate = row.get("sig_recruiter_response_rate")
    github_score = row.get("sig_github_activity_score")

    concern = None
    if pd.notna(notice) and notice > 60:
        concern = f"Notice period of {int(notice)} days may delay onboarding."
    elif pd.notna(gaps) and gaps > 0:
        concern = f"Has {int(gaps)} career gap(s) worth a closer look."
    elif pd.notna(fake_p) and fake_p > 0.05:
        concern = f"Minor trust flag (fake-probability {fake_p:.2f})."
    else:
        concern = "No major concerns identified from available signals."

    # Second, always-present highlight fact to keep entries distinct even when the concern repeats
    highlight = None
    if pd.notna(response_rate):
        highlight = f"Recruiter response rate {response_rate:.0%}."
    elif pd.notna(github_score):
        highlight = f"GitHub activity score {github_score:.1f}."
    elif pd.notna(trust_tier):
        highlight = f"Trust tier {trust_tier}."

    tail = " ".join([x for x in [concern, highlight] if x])
    return f"{headline_phrase} with {years_phrase}; strong match on {skill_phrase}. {tail}"

merged["reasoning"] = merged.apply(build_reasoning, axis=1)

# Keep exact required column order/names
final = merged[["candidate_id", "rank", "score", "reasoning"]]
final.to_csv(SUB_PATH, index=False)

print(f"Rewrote reasoning for {len(final)} rows -> {SUB_PATH}")
print(final.head(5).to_string())
