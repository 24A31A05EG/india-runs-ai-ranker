import pandas as pd
import numpy as np
import json
import os
import time
import hashlib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURATION & WEIGHTS ---
CONFIG = {
    "data_path": "./[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl",
    "must_have_keywords": [
        "embeddings", "vector-databases", "vector-search", "faiss", "pinecone", 
        "milvus", "weaviate", "qdrant", "elasticsearch", "opensearch", "python",
        "ranking-systems", "ranking-evaluation", "ndcg", "mrr", "map", "learning-to-rank", "information-retrieval"
    ],
    "pref_keywords": ["lora", "qlora", "peft", "xgboost", "neural", "hr-tech", "distributed", "open-source"],
    "weights": {"skill_match": 0.40, "experience_quality": 0.30, "behavioral": 0.30},
    "skill_weights": {"must_have": 2.0, "preferred": 1.0},
    "behavioral_weights": {
        "sig_github_activity_score": 1.5, "sig_recruiter_response_rate": 1.2,
        "sig_profile_completeness_score": 1.0, "sig_interview_completion_rate": 1.2,
        "sig_offer_acceptance_rate": 1.0, "avg_assessment_score": 1.5
    },
    "consulting_firms": ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "tech mahindra"]
}

def clean_and_extract_features():
    print(">>> PHASE 1: Cleaning & Feature Engineering...")
    start_time = time.time()
    
    # Pre-scan for duplicates via hashing
    hashes = {}
    with open(CONFIG["data_path"], "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            p = data.get("profile", {})
            career_texts = [str(j.get("description", "")) for j in data.get("career_history", [])]
            content = f"{p.get('headline')} | {p.get('summary')} | {' '.join(career_texts)}"
            h = hashlib.md5(content.encode("utf-8")).hexdigest()
            hashes[h] = hashes.get(h, 0) + 1
    
    dup_hashes = {h for h, count in hashes.items() if count > 1}
    print(f"    Found {len(dup_hashes)} duplicate content clusters.")

    # Main Processing
    rows = []
    with open(CONFIG["data_path"], "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            p = data.get("profile", {})
            career = data.get("career_history", [])
            sigs = data.get("redrob_signals", {})
            
            # Content Hash
            career_texts = [str(j.get("description", "")) for j in career]
            content = f"{p.get('headline')} | {p.get('summary')} | {' '.join(career_texts)}"
            h = hashlib.md5(content.encode("utf-8")).hexdigest()
            
            # Feature extraction
            row = {
                "candidate_id": data["candidate_id"],
                "name_anonymized": p.get("anonymized_name"),
                "headline_clean": str(p.get("headline", "")).lower(),
                "total_years_exp": p.get("years_of_experience", 0),
                "is_duplicate_resume": h in dup_hashes,
                "num_roles": len(career),
                "skills_list": ",".join([s.get("name", "").lower().replace(" ", "-") for s in data.get("skills", [])])
            }
            
            # Career Tenure
            durations = [j.get("duration_months", 0) for j in career if j.get("duration_months") is not None]
            row["avg_tenure_months"] = np.mean(durations) if durations else 0
            
            # Signals (Sentinel Handling)
            for key in ["github_activity_score", "recruiter_response_rate", "profile_completeness_score", 
                        "interview_completion_rate", "offer_acceptance_rate"]:
                val = sigs.get(key)
                row[f"sig_{key}"] = val if val != -1 else np.nan
                row[f"has_{key}"] = val != -1
            
            # Assessment Score
            scores = [v for k, v in sigs.get("skill_assessment_scores", {}).items() if v != -1]
            row["avg_assessment_score"] = np.mean(scores) if scores else np.nan
            
            rows.append(row)
            if (i+1) % 25000 == 0: print(f"    Processed {i+1} records...")

    df = pd.DataFrame(rows)
    df.to_parquet("candidates_features.parquet")
    print(f"    PHASE 1 Complete [{time.time()-start_time:.1f}s]")

def score_and_rank():
    print(">>> PHASE 2: Scoring & Ranking...")
    start_time = time.time()
    df = pd.read_parquet("candidates_features.parquet")
    with open("outputs/behavioral_signals_spec.json", "r") as f:
        sigs_spec = json.load(f)

    # 1. Skill Match
    def match_skills(s_str):
        if not s_str: return 0, []
        s_list = s_str.split(",")
        matched_m = [k for k in CONFIG["must_have_keywords"] if any(k in s for s in s_list)]
        matched_p = [k for k in CONFIG["pref_keywords"] if any(k in s for s in s_list)]
        score = (len(matched_m) * CONFIG["skill_weights"]["must_have"]) + (len(matched_p) * CONFIG["skill_weights"]["preferred"])
        return score, matched_m + matched_p

    results = df["skills_list"].apply(match_skills)
    df["score_skills"] = [r[0] for r in results]
    df["matched_skills"] = [", ".join(r[1]) for r in results]
    
    # 2. Experience Quality
    df["score_stability"] = np.clip(df["avg_tenure_months"] / 24.0, 0, 1) * 50
    df["consulting_penalty"] = df["headline_clean"].apply(lambda h: -50 if any(f in str(h) for f in CONFIG["consulting_firms"]) else 0)
    df["score_experience"] = np.clip(df["score_stability"] + 50 + df["consulting_penalty"], 0, 100)

    # 3. Behavioral
    for sig in sigs_spec:
        name = f"sig_{sig['signal']}"
        if name in df.columns:
            col = df[name].astype(float)
            if col.notnull().any():
                s_min, s_max = col.min(), col.max()
                norm = (col - s_min) / (s_max - s_min) if s_max > s_min else 0.5
                if "low" in str(sig.get("good", "")).lower(): norm = 1 - norm
                df[f"norm_{name}"] = norm

    beh_scores = sum(df[f"norm_{n}"].fillna(0.3) * w for n, w in CONFIG["behavioral_weights"].items() if f"norm_{n}" in df.columns)
    df["score_behavioral"] = (beh_scores / sum(CONFIG["behavioral_weights"].values())) * 100

    # 4. Final Composite
    for c in ["score_skills", "score_experience", "score_behavioral"]:
        df[f"final_{c}"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min()) * 100
    
    df["composite"] = (df["final_score_skills"] * CONFIG["weights"]["skill_match"] + 
                       df["final_score_experience"] * CONFIG["weights"]["experience_quality"] + 
                       df["final_score_behavioral"] * CONFIG["weights"]["behavioral"] -
                       df["is_duplicate_resume"].astype(int) * 25)

    # Ranking & Output
    df = df.sort_values("composite", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df["reasoning"] = df.apply(lambda r: f"Matched Must-Haves: {r['matched_skills']}. Score: {r['composite']:.1f}", axis=1)
    
    df.head(100)[["candidate_id", "rank", "composite", "reasoning"]].rename(columns={"composite": "score"}).to_csv("submission.csv", index=False)
    print(f">>> PHASE 2 Complete. Submission generated. [{time.time()-start_time:.1f}s]")

if __name__ == "__main__":
    clean_and_extract_features()
    score_and_rank()
