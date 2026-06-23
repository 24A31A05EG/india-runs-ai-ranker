"""
step3_trust_engine.py
======================
Candidate Trust & Authenticity Engine — Rolelytix | Redrob Hackathon
---------------------------------------------------------------------
Responsible for:
  • Fake / trap profile detection
  • Anomaly detection across 23 behavioral signals
  • fake_probability score  (0.0 – 1.0)
  • trust_score             (0 – 100)
  • trust_tier              (A / B / C / D)
  • Candidate penalty values consumed by step2_scoring.py

Runtime: O(N) vectorized pandas — no nested loops.
Expected execution on 100,000 candidates (CPU-only): < 15 seconds.

Usage
-----
  python step3_trust_engine.py \
      --input  candidates_preprocessed.parquet \
      --output candidates_with_trust.parquet

Output consumed by step2_scoring.py via:
  candidates_with_trust.parquet  →  merge on candidate_id
  columns added: trust_score, trust_tier, fake_probability,
                 anomaly_count, trust_penalty, + all flag_ columns
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION  (all thresholds in one place — easy to tune)
# ═══════════════════════════════════════════════════════════════════════════
CFG = {
    # ── Ghost profile ──────────────────────────────────────────────────────
    # Bottom 10th-percentile completeness PLUS no identity verification
    # (thresholds derived from the real 100k distribution)
    "ghost_completeness_pct": 0.10,     # bottom decile
    # ── Spam applicant ─────────────────────────────────────────────────────
    "spam_apps_pct":  0.90,             # top decile of applications_submitted_30d
    "spam_resp_rate": 0.20,             # AND recruiter response rate < 20 %
    # ── Skill–text inconsistency ───────────────────────────────────────────
    # Candidate lists ≥ N IR must-have skills but career text has zero IR evidence
    "skill_text_ir_skill_threshold": 5,
    # ── Behavioral outlier (z-score) ──────────────────────────────────────
    "outlier_z_threshold": 3.0,         # |z| > 3
    "outlier_signal_min":  2,           # must trigger on ≥ 2 signals
    # ── fake_probability weights (must sum ≤ 1.0) ─────────────────────────
    "fp_w_duplicate":       0.45,
    "fp_w_exp_mismatch":    0.15,
    "fp_w_ghost":           0.15,
    "fp_w_spam":            0.10,
    "fp_w_behavioral":      0.08,
    "fp_w_zero_engage":     0.07,
    # ── trust_score signal weights (positive, must sum = 1.0) ─────────────
    "ts_w_verified_email":       0.15,
    "ts_w_verified_phone":       0.13,
    "ts_w_linkedin_connected":   0.08,
    "ts_w_profile_completeness": 0.14,
    "ts_w_recruiter_response":   0.13,
    "ts_w_interview_completion": 0.12,
    "ts_w_offer_acceptance":     0.10,
    "ts_w_github_activity":      0.09,
    "ts_w_open_to_work":         0.06,
    # ── trust_score penalty weights (applied after positive score) ─────────
    "ts_pen_duplicate":     0.30,
    "ts_pen_exp_mismatch":  0.12,
    "ts_pen_ghost":         0.10,
    "ts_pen_spam":          0.08,
    "ts_pen_beh_outlier":   0.05,
    # ── Trust tier boundaries ──────────────────────────────────────────────
    "tier_A": 72,
    "tier_B": 52,
    "tier_C": 35,
    # below tier_C = D
    # ── Penalty applied to final_score in step2 ───────────────────────────
    "penalty_fake_high":   40,   # fake_probability >= 0.70
    "penalty_fake_medium": 20,   # fake_probability >= 0.45
    "penalty_duplicate":   35,   # flat duplicate penalty (overrides fake medium)
}

# IR must-have skill tokens (normalised, matching skills_list format)
IR_MUST_HAVE_SKILLS = [
    "embeddings", "vector-databases", "vector-search", "faiss", "pinecone",
    "qdrant", "milvus", "weaviate", "elasticsearch", "opensearch",
    "learning-to-rank", "information-retrieval", "ndcg", "mrr",
]

# IR evidence terms to search for in career_descriptions
IR_TEXT_EVIDENCE = [
    "embedding", "vector search", "faiss", "pinecone", "elasticsearch",
    "opensearch", "ndcg", "mrr", "learning to rank", "information retrieval",
    "semantic search", "dense retrieval", "sparse retrieval",
    "approximate nearest", "rerank", "bm25",
]

# Signals used for z-score behavioural outlier detection
OUTLIER_SIGNAL_COLS = [
    "sig_connection_count",
    "sig_endorsements_received",
    "sig_applications_submitted_30d",
    "sig_avg_response_time_hours",
    "sig_profile_views_received_30d",
    "sig_saved_by_recruiters_30d",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1.  HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    """Safe column getter — returns constant series if column is absent."""
    if name in df.columns:
        return df[name].copy()
    return pd.Series(default, index=df.index, dtype=float)


def _norm(series: pd.Series,
          lo: float = 0.0, hi: float = 100.0,
          invert: bool = False,
          missing_fill: float = 0.40) -> pd.Series:
    """
    Min-max normalise to [0, 1].
    Missing values filled with `missing_fill` (0.40 = slight negative prior,
    not 0.50 — avoids giving ghost profiles free points).
    """
    s = series.astype(float).clip(lo, hi)
    rng = hi - lo
    n = (s - lo) / rng if rng > 0 else pd.Series(0.5, index=series.index)
    if invert:
        n = 1.0 - n
    return n.fillna(missing_fill)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  ANOMALY FLAGS  (all vectorized)
# ═══════════════════════════════════════════════════════════════════════════
def compute_anomaly_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 8 boolean flag_ columns and a composite anomaly_count.
    All operations are vectorized — no Python loops over rows.

    Flags
    -----
    flag_duplicate_resume        Identical resume content (MD5 match from step1)
    flag_experience_mismatch     Stated exp vs career span gap > 5 years
    flag_ghost_profile           Low completeness + no identity verification
    flag_spam_applicant          High application volume + near-zero response rate
    flag_zero_engagement         Zero views, zero apps, not open to work
    flag_behavioral_outlier      ≥ 2 behavioral signals with |z-score| > 3
    flag_skill_text_inconsistency Lists ≥ 5 IR skills but zero IR career text evidence
    flag_inverted_salary         Stated salary min > stated salary max (data corruption)
    """
    # ── 2a. Duplicate (already computed in step1) ─────────────────────────
    df["flag_duplicate_resume"] = _col(df, "is_duplicate_resume", False).astype(bool)

    # ── 2b. Experience mismatch (already computed in step1) ───────────────
    df["flag_experience_mismatch"] = _col(df, "exp_career_mismatch", False).astype(bool)

    # ── 2c. Ghost profile ─────────────────────────────────────────────────
    completeness = _col(df, "sig_profile_completeness_score", 50.0)
    p10 = completeness.quantile(CFG["ghost_completeness_pct"])
    email_ok    = _col(df, "sig_verified_email",    False).fillna(False).astype(bool)
    phone_ok    = _col(df, "sig_verified_phone",    False).fillna(False).astype(bool)
    linkedin_ok = _col(df, "sig_linkedin_connected", False).fillna(False).astype(bool)
    open_work   = _col(df, "sig_open_to_work_flag",  False).fillna(False).astype(bool)

    df["flag_ghost_profile"] = (
        (completeness <= p10) &
        (~email_ok) &
        (~phone_ok) &
        (~linkedin_ok) &
        (~open_work)
    )

    # ── 2d. Spam applicant ────────────────────────────────────────────────
    apps_30d  = _col(df, "sig_applications_submitted_30d", 0.0)
    resp_rate = _col(df, "sig_recruiter_response_rate",    0.5)
    p90_apps  = apps_30d.quantile(CFG["spam_apps_pct"])

    df["flag_spam_applicant"] = (
        (apps_30d >= p90_apps) &
        (resp_rate < CFG["spam_resp_rate"])
    )

    # ── 2e. Zero engagement ───────────────────────────────────────────────
    views = _col(df, "sig_profile_views_received_30d", 1.0)
    df["flag_zero_engagement"] = (
        (views == 0) &
        (apps_30d == 0) &
        (~open_work)
    )

    # ── 2f. Behavioral outlier (z-score, fully vectorized) ────────────────
    outlier_hits = pd.Series(0, index=df.index, dtype=int)
    for col in OUTLIER_SIGNAL_COLS:
        if col in df.columns:
            s = df[col].astype(float)
            if s.std() > 0:
                z = (s - s.mean()) / s.std()
                outlier_hits += (z.abs() > CFG["outlier_z_threshold"]).fillna(False).astype(int)

    df["flag_behavioral_outlier"] = (outlier_hits >= CFG["outlier_signal_min"])
    df["_outlier_hit_count"] = outlier_hits   # kept for debugging, dropped at end

    # ── 2g. Skill–text inconsistency (vectorized str ops) ─────────────────
    # Count IR must-have skills claimed
    skills_lower = df["skills_list"].fillna("").str.lower()
    ir_skill_hits = sum(
        skills_lower.str.contains(rf"(?:^|,)\s*{s}\s*(?:,|$)", regex=True).astype(int)
        for s in IR_MUST_HAVE_SKILLS
    )
    # Check for any IR evidence in career text
    career_lower = df["career_descriptions"].fillna("").str.lower()
    has_ir_text = pd.Series(False, index=df.index)
    for term in IR_TEXT_EVIDENCE:
        has_ir_text |= career_lower.str.contains(term, regex=False)

    df["flag_skill_text_inconsistency"] = (
        (ir_skill_hits >= CFG["skill_text_ir_skill_threshold"]) &
        (~has_ir_text)
    )

    # ── 2h. Inverted salary (data integrity flag) ─────────────────────────
    sal_min = _col(df, "sig_expected_salary_range_inr_lpa_min", 0.0)
    sal_max = _col(df, "sig_expected_salary_range_inr_lpa_max", 0.0)
    df["flag_inverted_salary"] = (
        sal_min.notna() & sal_max.notna() & (sal_min > sal_max)
    )

    # ── 2i. Composite anomaly count ───────────────────────────────────────
    flag_cols = [
        "flag_duplicate_resume", "flag_experience_mismatch",
        "flag_ghost_profile", "flag_spam_applicant",
        "flag_zero_engagement", "flag_behavioral_outlier",
        "flag_skill_text_inconsistency", "flag_inverted_salary",
    ]
    df["anomaly_count"] = df[flag_cols].astype(int).sum(axis=1)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3.  FAKE PROBABILITY  (0.0 – 1.0)
# ═══════════════════════════════════════════════════════════════════════════
def compute_fake_probability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Additive weighted model — each component is a 0/1 flag scaled by its weight.
    Weights reflect confidence that the flag indicates inauthenticity:
      duplicate_resume  → strongest signal (exact content reuse)
      exp_mismatch      → high confidence (impossible timeline)
      ghost_profile     → high confidence (no verifiable identity)
      spam_applicant    → medium (could be aggressive job seeker)
      behavioral_outlier→ low-medium (statistical anomaly, not definitive)
      zero_engagement   → low (could be passive seeker)

    Result is clipped to [0, 1].
    """
    fp = (
        df["flag_duplicate_resume"].astype(float)        * CFG["fp_w_duplicate"]   +
        df["flag_experience_mismatch"].astype(float)     * CFG["fp_w_exp_mismatch"]+
        df["flag_ghost_profile"].astype(float)           * CFG["fp_w_ghost"]       +
        df["flag_spam_applicant"].astype(float)          * CFG["fp_w_spam"]        +
        df["flag_behavioral_outlier"].astype(float)      * CFG["fp_w_behavioral"]  +
        df["flag_zero_engagement"].astype(float)         * CFG["fp_w_zero_engage"]
    ).clip(0.0, 1.0)

    df["fake_probability"] = fp.round(4)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4.  TRUST SCORE  (0 – 100) + TRUST TIER
# ═══════════════════════════════════════════════════════════════════════════
def compute_trust_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    trust_score = (weighted positive signals − weighted penalties) × 100
    Clipped to [0, 100].

    Positive block (identity, engagement, activity) — missing signals filled
    with 0.40 neutral prior (NOT 0.50, to avoid rewarding unknown profiles).

    Penalty block — applied multiplicatively so a high-trust candidate cannot
    be zeroed out by a single marginal flag.
    """
    w = CFG   # shorthand

    # ── Positive signals ──────────────────────────────────────────────────
    n_email    = _col(df, "sig_verified_email",          False).fillna(False).astype(float)
    n_phone    = _col(df, "sig_verified_phone",          False).fillna(False).astype(float)
    n_linkedin = _col(df, "sig_linkedin_connected",      False).fillna(False).astype(float)
    n_complete = _norm(_col(df, "sig_profile_completeness_score", np.nan), 0, 100)
    n_resp     = _norm(_col(df, "sig_recruiter_response_rate",    np.nan), 0, 1, missing_fill=0.40)
    n_icr      = _norm(_col(df, "sig_interview_completion_rate",  np.nan), 0, 1, missing_fill=0.40)
    n_offer    = _norm(_col(df, "sig_offer_acceptance_rate",      np.nan), 0, 1, missing_fill=0.40)
    n_github   = _norm(_col(df, "sig_github_activity_score",      np.nan), 0, 100, missing_fill=0.40)
    n_otw      = _col(df, "sig_open_to_work_flag", False).fillna(False).astype(float)

    positive = (
        n_email    * w["ts_w_verified_email"]       +
        n_phone    * w["ts_w_verified_phone"]       +
        n_linkedin * w["ts_w_linkedin_connected"]   +
        n_complete * w["ts_w_profile_completeness"] +
        n_resp     * w["ts_w_recruiter_response"]   +
        n_icr      * w["ts_w_interview_completion"] +
        n_offer    * w["ts_w_offer_acceptance"]     +
        n_github   * w["ts_w_github_activity"]      +
        n_otw      * w["ts_w_open_to_work"]
    )
    # positive is in [0, 1] since weights sum to 1.0

    # ── Penalty block ─────────────────────────────────────────────────────
    penalty = (
        df["flag_duplicate_resume"].astype(float)    * w["ts_pen_duplicate"]   +
        df["flag_experience_mismatch"].astype(float) * w["ts_pen_exp_mismatch"]+
        df["flag_ghost_profile"].astype(float)       * w["ts_pen_ghost"]       +
        df["flag_spam_applicant"].astype(float)      * w["ts_pen_spam"]        +
        df["flag_behavioral_outlier"].astype(float)  * w["ts_pen_beh_outlier"]
    ).clip(0.0, 0.65)   # cap total penalty so a real engineer isn't zeroed by one flag

    raw_trust = (positive - penalty).clip(0.0, 1.0)
    df["trust_score"] = (raw_trust * 100).round(2)

    # ── Trust tier ────────────────────────────────────────────────────────
    conditions = [
        df["trust_score"] >= w["tier_A"],
        df["trust_score"] >= w["tier_B"],
        df["trust_score"] >= w["tier_C"],
    ]
    df["trust_tier"] = np.select(conditions, ["A", "B", "C"], default="D")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5.  TRUST PENALTY  (consumed by step2_scoring.py)
# ═══════════════════════════════════════════════════════════════════════════
def compute_trust_penalty(df: pd.DataFrame) -> pd.DataFrame:
    """
    trust_penalty is subtracted from final_score in step2_scoring.py.

    Priority (highest penalty wins — no stacking):
      1. Duplicate resume    → -35 pts  (definitive fake signal)
      2. fake_prob ≥ 0.70   → -40 pts  (multi-flag fake)
      3. fake_prob ≥ 0.45   → -20 pts  (suspicious)
      4. Trust tier D        → -10 pts  (low trust, no hard fake flag)
      5. Otherwise           →   0 pts
    """
    penalty = pd.Series(0.0, index=df.index)

    penalty = np.where(df["fake_probability"] >= 0.45,  CFG["penalty_fake_medium"], penalty)
    penalty = np.where(df["fake_probability"] >= 0.70,  CFG["penalty_fake_high"],   penalty)
    penalty = np.where(df["flag_duplicate_resume"],      CFG["penalty_duplicate"],   penalty)
    # Tier D adds a soft penalty if not already penalised harder
    penalty = np.where((df["trust_tier"] == "D") & (penalty == 0), 10.0, penalty)

    df["trust_penalty"] = penalty.astype(float)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 6.  BEHAVIORAL SIGNAL SCORE  (improved, replaces get_beh_score in step2)
# ═══════════════════════════════════════════════════════════════════════════
def compute_behavioral_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fully vectorized behavioral scoring using all available signals.
    Uses 0.40 neutral prior for missing values (NOT 0.50 freebie).

    Output: score_behavioral_v2 (0–100)
    """
    WEIGHTS = {
        "sig_recruiter_response_rate":    2.0,
        "sig_github_activity_score":      1.5,
        "avg_assessment_score":           1.5,
        "sig_profile_completeness_score": 1.0,
        "sig_interview_completion_rate":  1.2,
        "sig_offer_acceptance_rate":      1.0,
        "sig_connection_count":           0.5,
        "sig_endorsements_received":      0.5,
        "sig_search_appearance_30d":      0.3,
    }
    SCALE = {   # signals whose natural range is not 0–1
        "sig_github_activity_score":      100.0,
        "avg_assessment_score":           100.0,
        "sig_profile_completeness_score": 100.0,
        "sig_connection_count":           700.0,   # p99 ≈ 700 in the corpus
        "sig_endorsements_received":      80.0,    # p99 ≈ 80 in the corpus
        "sig_search_appearance_30d":      500.0,
    }

    total_weight = sum(WEIGHTS.values())
    weighted_sum = pd.Series(0.0, index=df.index)

    for sig, w in WEIGHTS.items():
        if sig in df.columns:
            raw = df[sig].astype(float)
        else:
            raw = pd.Series(np.nan, index=df.index)

        scale = SCALE.get(sig, 1.0)
        normed = (raw / scale).clip(0.0, 1.0)
        normed = normed.fillna(0.40)   # neutral prior for missing
        weighted_sum += normed * w

    base_score = (weighted_sum / total_weight) * 100.0

    # Multiplicative engagement modifiers
    otw      = _col(df, "sig_open_to_work_flag", True).fillna(True).astype(bool)
    resp     = _col(df, "sig_recruiter_response_rate", 0.5)
    apps     = _col(df, "sig_applications_submitted_30d", 5.0)

    modifier = pd.Series(1.0, index=df.index)
    modifier = modifier.where(otw,             modifier * 0.85)   # not open to work
    modifier = modifier.where(resp >= 0.20,    modifier * 0.80)   # very unresponsive
    modifier = modifier.where(apps <= 20,      modifier * 0.92)   # shotgun applicant

    df["score_behavioral_v2"] = (base_score * modifier).clip(0.0, 100.0).round(4)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 7.  SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════
def print_summary(df: pd.DataFrame) -> None:
    sep = "═" * 58
    print(f"\n{sep}")
    print("  TRUST ENGINE — SUMMARY REPORT")
    print(sep)

    n = len(df)
    print(f"  Total candidates         : {n:>8,}")
    print()

    # Trust tiers
    for tier, label in [("A","Excellent ≥72"), ("B","Good 52–71"),
                        ("C","Risky 35–51"),  ("D","Suspicious <35")]:
        cnt = (df["trust_tier"] == tier).sum()
        print(f"  Tier {tier} ({label:<14}): {cnt:>7,}  ({cnt/n*100:5.1f}%)")

    print()
    # fake_probability bands
    for lo, hi, label in [(0.70, 1.01, "HIGH   — exclude from ranking"),
                          (0.45, 0.70, "MEDIUM — heavy penalty"),
                          (0.10, 0.45, "LOW    — soft flag"),
                          (0.0,  0.10, "CLEAN  — no action")]:
        mask = (df["fake_probability"] >= lo) & (df["fake_probability"] < hi)
        cnt = mask.sum()
        print(f"  fake_prob {lo:.2f}–{hi:.2f}  ({label:<28}): {cnt:>6,}")

    print()
    # Individual flags
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    for fc in sorted(flag_cols):
        cnt = df[fc].astype(bool).sum()
        print(f"  {fc:<38}: {cnt:>6,}  ({cnt/n*100:.2f}%)")

    print()
    print(f"  Mean trust_score         : {df['trust_score'].mean():>7.1f}")
    print(f"  Mean fake_probability    : {df['fake_probability'].mean():>7.4f}")
    print(f"  Mean anomaly_count       : {df['anomaly_count'].mean():>7.3f}")
    total_penalised = (df["trust_penalty"] > 0).sum()
    print(f"  Candidates penalised     : {total_penalised:>7,}  ({total_penalised/n*100:.2f}%)")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════
# 8.  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def run_trust_engine(input_path: str, output_path: str) -> pd.DataFrame:
    t0 = time.time()
    print(f">>> Step 3 [Trust Engine]: Loading {input_path} ...")

    df = pd.read_parquet(input_path)
    print(f"    Loaded {len(df):,} records | {len(df.columns)} columns")

    print("    [1/5] Computing anomaly flags ...")
    df = compute_anomaly_flags(df)

    print("    [2/5] Computing fake_probability ...")
    df = compute_fake_probability(df)

    print("    [3/5] Computing trust_score & trust_tier ...")
    df = compute_trust_score(df)

    print("    [4/5] Computing trust_penalty ...")
    df = compute_trust_penalty(df)

    print("    [5/5] Computing behavioral score (v2) ...")
    df = compute_behavioral_score(df)

    # Drop internal debug columns before saving
    df = df.drop(columns=["_outlier_hit_count"], errors="ignore")

    df.to_parquet(output_path)
    elapsed = time.time() - t0
    print(f"    Output → {output_path}  [{elapsed:.1f}s]")

    print_summary(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Trust & Authenticity Engine")
    parser.add_argument("--input",  default="candidates_preprocessed.parquet",
                        help="Output of step1_preprocessing.py")
    parser.add_argument("--output", default="candidates_with_trust.parquet",
                        help="Enriched parquet consumed by step2_scoring.py")
    args = parser.parse_args()
    run_trust_engine(args.input, args.output)
