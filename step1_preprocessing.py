import pandas as pd
import numpy as np
import json
import hashlib
from datetime import datetime
import os
import time

def run_preprocessing(input_path, output_path):
    print(f">>> Step 1: Preprocessing {input_path}...")
    
    # 1. Content-based Duplicate Detection (Pre-scan)
    # DECISION: We use resume TEXT alone (Headline + Summary + Full Career History Titles/Descs)
    # RATIONALE: 100% text identity across these fields is a catastrophic indicator of templating
    # or resume farming. Differences in "Years of Exp" labels are often trivial bot variations.
    hashes = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            p = data.get("profile", {})
            career = data.get("career_history", [])
            # Include Title + Company + Description for high-entropy anchor
            career_details = []
            for j in career:
                item = f"{j.get('title')}:{j.get('company')}:{j.get('description')}"
                career_details.append(item)
            
            content = f"{p.get('headline')} | {p.get('summary')} | {' '.join(career_details)}"
            h = hashlib.md5(content.encode("utf-8")).hexdigest()
            hashes[h] = hashes.get(h, 0) + 1
    
    dup_hashes = {h for h, count in hashes.items() if count > 1}

    # 2. Main Processing
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            p = data.get("profile", {})
            career = data.get("career_history", [])
            sigs = data.get("redrob_signals", {})
            
            row = {
                "candidate_id": data["candidate_id"],
                "headline": str(p.get("headline", "")),
                "summary": str(p.get("summary", "")),
                "total_years_exp": p.get("years_of_experience", 0),
                "skills_list": ",".join([s.get("name", "").lower().replace(" ", "-") for s in data.get("skills", [])])
            }
            
            # Recompute hash for flag
            career_details = [f"{j.get('title')}:{j.get('company')}:{j.get('description')}" for j in career]
            content = f"{p.get('headline')} | {p.get('summary')} | {' '.join(career_details)}"
            h = hashlib.md5(content.encode("utf-8")).hexdigest()
            row["is_duplicate_resume"] = h in dup_hashes

            # Career Features
            dates = []
            for j in career:
                try:
                    s = datetime.strptime(j["start_date"], "%Y-%m-%d")
                    e = datetime.strptime(j["end_date"], "%Y-%m-%d") if j["end_date"] else datetime.now()
                    dates.append((s, e))
                except: continue
            
            dates.sort()
            row["num_roles"] = len(career)
            row["avg_tenure_months"] = np.mean([j.get("duration_months", 0) for j in career]) if career else 0
            
            gaps = 0
            overlaps = False
            career_span_days = 0
            if dates:
                career_span_days = (max([d[1] for d in dates]) - min([d[0] for d in dates])).days
                for j in range(len(dates)-1):
                    if dates[j+1][0] > dates[j][1]:
                        gap_days = (dates[j+1][0] - dates[j][1]).days
                        if gap_days > 60: gaps += 1
                    elif (dates[j][1] - dates[j+1][0]).days > 30:
                        overlaps = True
            
            row["num_career_gaps"] = gaps
            row["has_overlapping_roles"] = overlaps
            row["job_switch_rate_per_year"] = (row["num_roles"] / (row["total_years_exp"] if row["total_years_exp"] > 0 else 1))
            row["career_span_years"] = career_span_days / 365.25
            row["exp_career_mismatch"] = abs(row["total_years_exp"] - row["career_span_years"]) > 5
            row["career_descriptions"] = " ".join([str(j.get("description", "")) for j in career])

            # Signals
            for k, v in sigs.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        col_name = f"sig_{k}_{sub_k}"
                        row[col_name] = sub_v if sub_v != -1 else np.nan
                elif k != "skill_assessment_scores":
                    row[f"sig_{k}"] = v if v != -1 else np.nan
            
            scores = [v for k, v in sigs.get("skill_assessment_scores", {}).items() if v != -1]
            row["avg_assessment_score"] = np.mean(scores) if scores else np.nan
            rows.append(row)

    pd.DataFrame(rows).to_parquet(output_path)
    print(f"    Saved {len(rows)} records to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 1: Preprocessing for Redrob Challenge")
    parser.add_argument("--input", default="./[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl", help="Input JSONL path")
    parser.add_argument("--output", default="candidates_preprocessed.parquet", help="Output Parquet path")
    args = parser.parse_args()
    
    start_time = time.time()
    run_preprocessing(args.input, args.output)
    print(f"\n--- STEP 1 TOTAL RUNTIME: {time.time() - start_time:.2f}s ---")
