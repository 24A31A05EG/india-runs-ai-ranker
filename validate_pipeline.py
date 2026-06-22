import json
import os
import pandas as pd
from step1_preprocessing import run_preprocessing
from step2_scoring import score_candidates

def create_synthetic_data(path):
    candidates = [
        {
            "candidate_id": "CAND_TRAP_2", # Micropython trap
            "profile": {"headline": "Embedded dev", "summary": "Micropython expert", "years_of_experience": 5.0},
            "career_history": [{"title": "Dev", "company": "A", "start_date": "2019-01-01", "end_date": "2024-01-01", "description": "Used map() function."}],
            "skills": [{"name": "C++"}],
            "redrob_signals": {}
        },
        {
            "candidate_id": "CAND_REAL_1", # Good candidate
            "profile": {"headline": "AI Engineer", "summary": "Experienced in Vector Databases.", "years_of_experience": 7.0},
            "career_history": [{"title": "AI Engineer", "company": "B", "start_date": "2017-01-01", "end_date": "2024-01-01", "description": "Implemented embeddings search."}],
            "skills": [{"name": "Python"}, {"name": "Pinecone"}],
            "redrob_signals": {"recruiter_response_rate": 0.8}
        },
        {
            "candidate_id": "CAND_OVER_EXP", # High exp candidate
            "profile": {"headline": "Staff AI Engineer", "summary": "Experienced in Vector Databases.", "years_of_experience": 11.0},
            "career_history": [{"title": "Staff AI Engineer", "company": "C", "start_date": "2013-01-01", "end_date": "2024-01-01", "description": "Implemented embeddings search."}],
            "skills": [{"name": "Python"}, {"name": "Pinecone"}],
            "redrob_signals": {"recruiter_response_rate": 0.8}
        }
    ]
    with open(path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

def run_test():
    create_synthetic_data("test_candidates.jsonl")
    run_preprocessing("test_candidates.jsonl", "test_preprocessed.parquet")
    score_candidates("test_preprocessed.parquet", "test_submission.csv", "job_requirements.json")
    
    df = pd.read_csv("test_submission.csv")
    print("\n--- RAW TEST RESULTS (FULL COMPONENT DISPLAY) ---")
    cols = ["candidate_id", "score_skills", "score_experience", "score_behavioral", "fake_penalty", "final_score", "reasoning"]
    # Ensure they exist in the CSV (Scoring script writes some but not all, I'll print from a merged view if needed)
    # Actually, the scorer saves everything to CSV now in my local edit.
    print(df[cols].to_string(index=False))

if __name__ == "__main__":
    run_test()
