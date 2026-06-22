"""
Checks whether vector-DB / eval-framework must-haves are (a) actually absent
from the corpus, or (b) present but not being matched due to a bug.
"""
import json
import re
from pathlib import Path
import pandas as pd

# 1. Print the literal must_have_skills list your scoring script is using
jd = json.loads(Path("outputs/job_requirements.json").read_text())
print("=== must_have_skills as declared ===")
for s in jd["requirements"]["must_have_skills"]:
    print(" -", s)
print("\n=== preferred_skills as declared ===")
for s in jd["requirements"]["preferred_skills"]:
    print(" -", s)

# 2. Raw keyword presence counts across the FULL 100k corpus
vector_db_terms = ["pinecone", "milvus", "qdrant", "weaviate", "faiss",
                    "elasticsearch", "opensearch", "vector database", "vector db"]
eval_terms = ["ndcg", "mrr", "map@", "mean average precision",
              "evaluation framework", "a/b test", "ab test"]
finetune_terms = ["lora", "qlora", "peft", "fine-tun", "finetun"]

df = pd.read_parquet("candidates_clean.parquet")

# adjust this to whatever the real concatenated-text column is named
text_col_candidates = [c for c in df.columns if "text" in c.lower() or "summary" in c.lower() or "resume" in c.lower()]
print(f"\n=== Text columns available: {text_col_candidates} ===")
if not text_col_candidates:
    raise SystemExit("No obvious text column found — list df.columns and tell me which one holds resume/summary text.")

text_col = text_col_candidates[0]
print(f"Using column: {text_col}\n")

corpus = df[text_col].fillna("").str.lower()

def count_term(term):
    return corpus.str.contains(re.escape(term), regex=True).sum()

print("=== Vector-DB term frequency across all 100,000 candidates ===")
for t in vector_db_terms:
    print(f"  '{t}': {count_term(t)} candidates")

print("\n=== Eval-framework term frequency across all 100,000 candidates ===")
for t in eval_terms:
    print(f"  '{t}': {count_term(t)} candidates")

print("\n=== Fine-tuning term frequency across all 100,000 candidates (for comparison) ===")
for t in finetune_terms:
    print(f"  '{t}': {count_term(t)} candidates")

# 3. Specifically check the top 20 ranked candidates' raw text for these terms
print("\n=== Checking top 20 ranked candidates directly ===")
sub = pd.read_csv("submission.csv").head(20)
id_col = sub.columns[0]
for cid in sub[id_col]:
    row = df[df["candidate_id"] == cid]
    if row.empty:
        print(f"{cid}: NOT FOUND in candidates_clean.parquet (id mismatch?)")
        continue
    t = str(row[text_col].iloc[0]).lower()
    has_vdb = any(term in t for term in vector_db_terms)
    has_eval = any(term in t for term in eval_terms)
    print(f"{cid}: vector-db terms present={has_vdb}, eval-framework terms present={has_eval}")
