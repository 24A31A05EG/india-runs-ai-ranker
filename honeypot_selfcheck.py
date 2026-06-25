"""
Self-check: estimate how many of our TOP 100 picks look risky according to
OUR OWN trust/anomaly signals (fake_probability, trust_tier, anomaly flags).

This is NOT the official hidden honeypot rate (that ground truth is not
available to us) -- it's a sanity check using our own detection system,
so we can catch obvious problems before submitting.

Run from the repo root:
    python honeypot_selfcheck.py
"""

import pandas as pd

sub = pd.read_csv("submission.csv")
cand = pd.read_parquet("candidates_with_trust.parquet")

flag_cols = [
    "flag_duplicate_resume", "flag_experience_mismatch", "flag_ghost_profile",
    "flag_spam_applicant", "flag_zero_engagement", "flag_behavioral_outlier",
    "flag_skill_text_inconsistency", "flag_inverted_salary",
]
flag_cols = [c for c in flag_cols if c in cand.columns]

keep_cols = ["candidate_id", "fake_probability", "trust_tier", "trust_score", "anomaly_count"] + flag_cols
cand_small = cand[keep_cols]

merged = sub.merge(cand_small, on="candidate_id", how="left")

print(f"Top 100 picks merged: {len(merged)} rows\n")

print("== fake_probability distribution in Top 100 ==")
print(merged["fake_probability"].describe())
print()

high_fake = merged[merged["fake_probability"] > 0.5]
print(f"Candidates with fake_probability > 0.5: {len(high_fake)} ({len(high_fake)}%)")
print()

print("== trust_tier counts in Top 100 ==")
print(merged["trust_tier"].value_counts())
print()

print("== anomaly_count distribution in Top 100 ==")
print(merged["anomaly_count"].value_counts().sort_index())
print()

any_flag = merged[flag_cols].any(axis=1) if flag_cols else pd.Series([False] * len(merged))
print(f"Candidates with ANY anomaly flag = True: {any_flag.sum()} ({any_flag.sum()}%)")
print()

if any_flag.sum() > 0:
    print("Flagged candidates detail:")
    print(merged.loc[any_flag, ["candidate_id", "rank", "fake_probability", "trust_tier"] + flag_cols].to_string())

print("\n--- Rule of thumb: if 'fake_probability > 0.5' count or 'any flag' count exceeds 10, investigate before submitting. ---")
