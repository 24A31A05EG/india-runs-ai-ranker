"""
step2_scoring.py  — PATCHED with Trust Engine integration
==========================================================
Drop-in replacement for step2_scoring.py.

Changes from original
---------------------
  [INT-1] Loads candidates_with_trust.parquet from step3 and merges on candidate_id.
  [INT-2] Uses score_behavioral_v2 from step3 instead of the original get_beh_score()
          (fixes the 50-point fallback bug and adds 9-signal coverage).
  [INT-3] Replaces fake_penalty (flat 40) with trust_penalty from step3
          (graduated: 35 for duplicates, 40 for multi-flag fakes, 20 for suspicious).
  [INT-4] trust_bonus (±5) added to composite based on trust_tier.
  [INT-5] Consulting firm check extended to summary text (was headline-only).
  [INT-6] Reason string now includes trust_tier and fake_probability for auditability.
"""

import pandas as pd
import numpy as np
import json
import re
import argparse
import time
from pathlib import Path


def score_candidates(input_path: str, output_path: str, req_path: str) -> pd.DataFrame:
    print(f">>>> Step 2 (patched): Loading {input_path} ...")
    df = pd.read_parquet(input_path)

    with open(req_path, "r") as f:
        req = json.load(f)

    # ── [INT-1] Merge trust engine output ────────────────────────────────
    trust_path = input_path.replace("preprocessed", "with_trust")
    if Path(trust_path).exists():
        trust_cols = [
            "candidate_id", "trust_score", "trust_tier", "fake_probability",
            "anomaly_count", "trust_penalty", "score_behavioral_v2",
            "flag_duplicate_resume", "flag_ghost_profile", "flag_spam_applicant",
            "flag_skill_text_inconsistency", "flag_experience_mismatch",
            "flag_behavioral_outlier",
        ]
        trust_df = pd.read_parquet(trust_path, columns=trust_cols)
        df = df.merge(trust_df, on="candidate_id", how="left")
        has_trust = True
        print(f"    Merged trust data from {trust_path}")
    else:
        has_trust = False
        # Safe defaults so the rest of the code runs without branching
        df["trust_score"]       = 50.0
        df["trust_tier"]        = "B"
        df["fake_probability"]  = 0.0
        df["anomaly_count"]     = 0
        df["trust_penalty"]     = df["is_duplicate_resume"].astype(float) * 35
        df["score_behavioral_v2"] = 40.0
        print("    [WARN] Trust parquet not found — run step3_trust_engine.py first.")

    # ─────────────────────────────────────────────────────────────────────
    # 1. SKILL MATCH SCORE
    # ─────────────────────────────────────────────────────────────────────
    tag_only = set(req.get("tag_only_skills", []))

    def get_skill_score(row):
        text   = f"{row['headline']} {row['summary']} {row['career_descriptions']}".lower()
        skills = [s.strip() for s in str(row["skills_list"]).split(",") if s.strip()]

        matched_must = []
        for term in req["must_have"]:
            if term in tag_only:
                if term in skills:
                    matched_must.append(term)
            elif term in skills or re.search(rf"\b{re.escape(term)}\b", text):
                matched_must.append(term)

        matched_pref = []
        for term in req["preferred"]:
            if term in skills or re.search(rf"\b{re.escape(term)}\b", text):
                matched_pref.append(term)

        # Semantic domain bonus for IR vocabulary not covered by exact keywords
        semantic_terms = [
            "dense retrieval", "sparse retrieval", "bm25", "hybrid search",
            "approximate nearest neighbor", "ann", "semantic search",
            "reranking", "re-ranking",
        ]
        bonus = min(sum(1 for t in semantic_terms if t in text) * 0.5, 3.0)
        raw   = (len(matched_must) * 2) + len(matched_pref) + bonus
        return raw, matched_must, matched_pref

    skill_results      = df.apply(get_skill_score, axis=1)
    df["skill_raw"]    = [r[0] for r in skill_results]
    df["matched_must"] = [", ".join(r[1]) for r in skill_results]
    df["matched_pref"] = [", ".join(r[2]) for r in skill_results]
    max_possible       = (len(req["must_have"]) * 2) + len(req["preferred"])
    df["score_skills"] = (df["skill_raw"] / max_possible * 100).clip(0, 100)

    # ─────────────────────────────────────────────────────────────────────
    # 2. EXPERIENCE QUALITY SCORE
    # ─────────────────────────────────────────────────────────────────────
    def get_exp_score(row):
        score = 50
        y = float(row["total_years_exp"])

        if   5 <= y <= 9:    score += 20
        elif 4 <= y < 5:     score += 10
        elif 9 < y <= 12:    score += max(10, 20 - (y - 9) * (10 / 3))
        elif y > 12:         score += 5

        if row["job_switch_rate_per_year"] > 1.0 / req["disqualifiers"]["job_hop_threshold_years"]:
            score -= 30

        # [INT-5] Check headline + summary, not just headline
        firm_text = f"{row['headline']} {row.get('summary', '')}".lower()
        for firm in req["disqualifiers"]["firms"]:
            if firm in firm_text:
                score -= 50
                break

        if row.get("exp_career_mismatch"):
            score -= 20

        full_text = f"{row['headline']} {row.get('summary', '')}".lower()
        if any(t in full_text for t in ["computer vision", "robotics", "hardware"]):
            if not any(t in full_text for t in ["nlp", "information retrieval", "llm", "search"]):
                score -= 30

        if "architect" in str(row["headline"]).lower() and row.get("num_career_gaps", 0) > 0:
            score -= 20

        np_days = row.get("sig_notice_period_days", np.nan)
        if pd.notnull(np_days) and float(np_days) > 30:
            score -= 10

        return float(np.clip(score, 0, 100))

    df["score_experience"] = df.apply(get_exp_score, axis=1)

    # ─────────────────────────────────────────────────────────────────────
    # 3. BEHAVIORAL SCORE  [INT-2]
    # ─────────────────────────────────────────────────────────────────────
    if has_trust and "score_behavioral_v2" in df.columns:
        df["score_behavioral"] = df["score_behavioral_v2"].fillna(40.0)
    else:
        # Minimal fallback — 0.40 neutral prior, not 50
        BWEIGHTS = {
            "sig_recruiter_response_rate":    2.0,
            "sig_github_activity_score":      1.5,
            "avg_assessment_score":           1.5,
            "sig_profile_completeness_score": 1.0,
            "sig_interview_completion_rate":  1.2,
        }
        BSCALE = {"sig_github_activity_score": 100.0,
                  "avg_assessment_score":       100.0,
                  "sig_profile_completeness_score": 100.0}

        total_w = sum(BWEIGHTS.values())
        wsum = pd.Series(0.0, index=df.index)
        for sig, w in BWEIGHTS.items():
            if sig in df.columns:
                raw  = df[sig].astype(float)
                sc   = BSCALE.get(sig, 1.0)
                norm = (raw / sc).clip(0, 1).fillna(0.40)
            else:
                norm = pd.Series(0.40, index=df.index)
            wsum += norm * w
        df["score_behavioral"] = ((wsum / total_w) * 100).clip(0, 100)

    # ─────────────────────────────────────────────────────────────────────
    # 4. FINAL COMPOSITE  [INT-3] [INT-4]
    # ─────────────────────────────────────────────────────────────────────
    # trust_bonus: small adjustment based on authenticity tier
    trust_bonus_map = {"A": 5.0, "B": 2.0, "C": -2.0, "D": -5.0}
    df["trust_bonus"] = df["trust_tier"].map(trust_bonus_map).fillna(0.0)

    df["final_score"] = (
        df["score_skills"]     * 0.40 +
        df["score_experience"] * 0.30 +
        df["score_behavioral"] * 0.30 -
        df["trust_penalty"]          +   # [INT-3] graduated penalty from step3
        df["trust_bonus"]                # [INT-4] ±5 pts for authenticity
    )
    df["final_score"] = np.clip(df["final_score"].astype(float), 0.0, 100.0)

    # ─────────────────────────────────────────────────────────────────────
    # 5. SORT & EXPORT
    # ─────────────────────────────────────────────────────────────────────
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)

    def build_reason(row):
        trust_str = (f" | Trust:{row['trust_tier']}({row['trust_score']:.0f})"
                     f" FakeP:{row['fake_probability']:.2f}")
        return (
            f"Must-haves: {row['matched_must']}. "
            f"Skill:{row['score_skills']:.0f} "
            f"Exp:{row['score_experience']:.0f} "
            f"Beh:{row['score_behavioral']:.0f} "
            f"Pen:{row['trust_penalty']:.0f}"
            f"{trust_str}"
        )

    out = df.head(100).copy()
    out["Reason"] = out.apply(build_reason, axis=1)
    out = out.rename(columns={"candidate_id": "Candidate ID", "final_score": "Score"})
    out[["Rank", "Candidate ID", "Score", "Reason"]].to_csv(output_path, index=False)
    print(f"    Saved top 100 → {output_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2 (patched): Scoring")
    parser.add_argument("--input",  default="candidates_preprocessed.parquet")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--reqs",   default="job_requirements.json")
    args = parser.parse_args()

    t0 = time.time()
    df_r = score_candidates(args.input, args.output, args.reqs)

    print(f"\n--- STEP 2 RUNTIME: {time.time()-t0:.1f}s ---")
    print("\n--- PERCENTILES ---")
    print(df_r["final_score"].describe(percentiles=[.25,.5,.75,.9,.99]))

    if "trust_tier" in df_r.columns:
        print("\n--- TRUST TIER IN TOP 100 ---")
        print(df_r.head(100)["trust_tier"].value_counts().sort_index())

    if "trust_penalty" in df_r.columns:
        penalised = (df_r["trust_penalty"] > 0).sum()
        print(f"\nCandidates penalised: {penalised:,}")
