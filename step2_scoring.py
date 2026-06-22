import pandas as pd
import numpy as np
import json
import re
import argparse
import time

def score_candidates(input_path, output_path, req_path):
    print(f">>>> Step 2: Scoring from {input_path}...")
    df = pd.read_parquet(input_path)
    with open(req_path, "r") as f:
        req = json.load(f)
    
    # --- 1. SKILL MATCH SCORE ---
    def get_skill_score(row):
        text = f"{row['headline']} {row['summary']} {row['career_descriptions']}".lower()
        skills = row["skills_list"].split(",")
        
        matched_must = []
        for term in req["must_have"]:
            if term in req.get("tag_only_skills", []):
                if term in skills: matched_must.append(term)
            elif term in skills or re.search(fr"\b{re.escape(term)}\b", text):
                matched_must.append(term)
        
        matched_pref = []
        for term in req["preferred"]:
            if term in skills or re.search(fr"\b{re.escape(term)}\b", text):
                matched_pref.append(term)
                
        raw_score = (len(matched_must) * 2) + len(matched_pref)
        return raw_score, matched_must
    
    skill_results = df.apply(get_skill_score, axis=1)
    df["skill_raw"] = [r[0] for r in skill_results]
    df["matched_must"] = [", ".join(r[1]) for r in skill_results]
    max_possible = (len(req["must_have"]) * 2) + len(req["preferred"])
    df["score_skills"] = (df["skill_raw"] / max_possible) * 100

    # --- 2. EXPERIENCE QUALITY SCORE ---
    def get_exp_score(row):
        score = 50 
        y = row["total_years_exp"]
        if 5 <= y <= 9: score += 20
        elif 4 <= y < 5: score += 10
        elif 9 < y <= 12:
            decay_val = 20 - ((y - 9) * (10 / 3)) 
            score += max(10, decay_val)
        
        if row["job_switch_rate_per_year"] > (1 / req["disqualifiers"]["job_hop_threshold_years"]):
            score -= 30
        
        h = str(row["headline"]).lower()
        for firm in req["disqualifiers"]["firms"]:
            if firm in h: score -= 50
        if row.get("exp_career_mismatch"): score -= 20
        
        text = f"{row['headline']} {row['summary']}".lower()
        if any(t in text for t in ["computer vision", "robotics", "cv", "hardware"]):
            if not any(t in text for t in ["nlp", "information retrieval", "llm", "search"]):
                score -= 30
                
        if "architect" in h and row["num_career_gaps"] > 0: score -= 20
        
        # Notice Period Penalty (JD: sub-30 highly preferred)
        if pd.notnull(row.get("sig_notice_period_days")) and row["sig_notice_period_days"] > 30:
            score -= 10
            
        return np.clip(score, 0, 100)

    df["score_experience"] = df.apply(get_exp_score, axis=1)

    # --- 3. BEHAVIORAL SCORE ---
    weights = {
        "sig_recruiter_response_rate": 2.0,
        "sig_github_activity_score": 1.5,
        "avg_assessment_score": 1.5,
        "sig_profile_completeness_score": 1.0,
        "sig_interview_completion_rate": 1.2
    }
    
    def get_beh_score(row):
        total_score, total_weight = 0, 0
        for sig, w in weights.items():
            val = row.get(sig)
            if pd.notnull(val):
                norm_val = val / 100.0 if val > 1 else val
                total_score += float(norm_val) * w
                total_weight += w
        
        mod = 1.0
        if row.get("sig_open_to_work_flag") == False: mod *= 0.8
        resp_rate = row.get("sig_recruiter_response_rate")
        if pd.notnull(resp_rate) and resp_rate < 0.2: mod *= 0.7
            
        return (total_score / total_weight * 100 * mod) if total_weight > 0 else 50

    df["score_behavioral"] = df.apply(get_beh_score, axis=1).astype(float)

    # --- 4. FINAL COMPOSITE ---
    df["fake_penalty"] = df["is_duplicate_resume"].apply(lambda x: 40 if x else 0)
    df["final_score"] = (
        df["score_skills"] * 0.4 +
        df["score_experience"] * 0.3 +
        df["score_behavioral"] * 0.3 -
        df["fake_penalty"]
    )
    df["final_score"] = np.clip(df["final_score"].astype(float), 0, 100)
    
    # Sort and Export (EXACT SPEC: Rank, Candidate ID, Score, Reason)
    df = df.sort_values("final_score", ascending=False)
    df["Rank"] = range(1, len(df)+1)
    
    output_df = df.head(100).copy()
    output_df = output_df.rename(columns={
        "candidate_id": "Candidate ID",
        "final_score": "Score",
        "reasoning": "Reason"
    })
    
    def get_reason(row):
        return f"Must-haves: {row['matched_must']}. Skill:{row['score_skills']:.0f}, Exp:{row['score_experience']:.0f}, Beh:{row['score_behavioral']:.0f}, Dup:{row['fake_penalty']:.0f}"
    
    output_df["Reason"] = output_df.apply(get_reason, axis=1)
    output_df[["Rank", "Candidate ID", "Score", "Reason"]].to_csv(output_path, index=False)
    print(f"    Saved top 100 to {output_path}")
    return df # For percentile reporting

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Scoring for Redrob Challenge")
    parser.add_argument("--input", default="candidates_preprocessed.parquet", help="Input Parquet path")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--reqs", default="job_requirements.json", help="Job Requirements JSON path")
    args = parser.parse_args()
    
    start_time = time.time()
    df_results = score_candidates(args.input, args.output, args.reqs)
    
    print(f"\n--- STEP 2 TOTAL RUNTIME: {time.time() - start_time:.2f}s ---")
    
    print("\n--- SCORE PERCENTILES ---")
    print(df_results["final_score"].describe(percentiles=[.25, .5, .75, .9, .99]))
    
    fake_count = (df_results["fake_penalty"] > 0).sum()
    print(f"\nCandidates with Fake Penalty > 0: {fake_count}")
