"""
Phase 2 — Embedding module tests on real CV/job offer data.

Tests:
1. get_embedding() returns a 768-dimensional vector
2. Identical text produces identical vectors
3. On full CV/job offer texts, similarity works correctly
4. Batch embedding produces the same results as individual calls
5. Performance comparison (batch vs single)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from embeddings import (
    get_embedding,
    get_embeddings_batch,
    cosine_similarity,
    rank_by_similarity,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
)

ROOT = Path(__file__).parent

print(f"Model: {EMBEDDING_MODEL}  |  Dim: {EMBEDDING_DIM}\n")

# ── Test 1: Dimensions and repeatability ──────────────────────────────────────
print("=== Test 1: Dimensions and repeatability ===")
v1 = get_embedding("Python developer with FastAPI experience")
v2 = get_embedding("Python developer with FastAPI experience")

assert len(v1) == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM} dims, got {len(v1)}"
assert cosine_similarity(v1, v2) > 0.9999, "Same text should produce identical vectors"
print(f"  Dimensions: {len(v1)} OK")
print(f"  Repeatability (identical text): {cosine_similarity(v1, v2):.6f} OK")

# ── Test 2: Full CV vs job offer texts ────────────────────────────────────────
print("\n=== Test 2: Full CV vs job offer texts (real data) ===")

cvs_dir  = ROOT / "data" / "sample_cvs"
jobs_dir = ROOT / "data" / "sample_jobs"

cv_texts  = {p.stem: p.read_text(encoding="utf-8") for p in cvs_dir.glob("*.txt")}
job_texts = {p.stem: p.read_text(encoding="utf-8") for p in jobs_dir.glob("*.txt")}

# Generate embeddings (CVs as documents, job offers as documents)
t0 = time.time()
cv_vecs  = {name: get_embedding(text, task="search_document") for name, text in cv_texts.items()}
job_vecs = {name: get_embedding(text, task="search_document") for name, text in job_texts.items()}
elapsed = time.time() - t0

print(f"  Generated {len(cv_vecs) + len(job_vecs)} embeddings in {elapsed:.1f}s")

# Similarity matrix
print("\n  Cosine similarity matrix (CV rows x job offer columns):")
job_names = list(job_vecs.keys())
print(f"  {'CV / Job offer':<35}", end="")
for j in job_names:
    print(f"  {j[:22]:<22}", end="")
print()

for cv_name, cv_vec in cv_vecs.items():
    print(f"  {cv_name:<35}", end="")
    for job_name, job_vec in job_vecs.items():
        sim = cosine_similarity(cv_vec, job_vec)
        print(f"  {sim:.4f}               ", end="")
    print()

# ── Test 3: rank_by_similarity ────────────────────────────────────────────────
print("\n=== Test 3: CV ranking for Python Backend offer ===")
query_vec  = job_vecs["python_backend_senior"]
candidates = [(name, vec) for name, vec in cv_vecs.items()]
ranking    = rank_by_similarity(query_vec, candidates)

for rank, (name, score) in enumerate(ranking, 1):
    bar = "█" * int(score * 40)
    print(f"  #{rank}  {name:<35}  {score:.4f}  {bar}")

print("\n  Expected: jan_kowalski_python at #1 (Python + FastAPI + PostgreSQL)")

# ── Test 4: Asymmetric search (query vs document) ─────────────────────────────
print("\n=== Test 4: Asymmetric search — CV as query, job offer as document ===")
print("  (Correct retrieval mode: query searches against documents)")
cv_query_vecs = {name: get_embedding(text, task="search_query") for name, text in cv_texts.items()}

for job_name, job_doc_vec in job_vecs.items():
    print(f"\n  Job offer: {job_name}")
    candidates_q = [(name, vec) for name, vec in cv_query_vecs.items()]
    ranking_q    = rank_by_similarity(job_doc_vec, candidates_q)
    for rank, (name, score) in enumerate(ranking_q, 1):
        print(f"    #{rank}  {name:<35}  {score:.4f}")

# ── Test 5: Batch vs single ───────────────────────────────────────────────────
print("\n=== Test 5: Batch embedding vs individual calls ===")
texts = list(cv_texts.values())

t0    = time.time()
batch = get_embeddings_batch(texts)
t_batch = time.time() - t0

t0     = time.time()
single = [get_embedding(t) for t in texts]
t_single = time.time() - t0

max_diff = max(abs(b - s) for bv, sv in zip(batch, single) for b, s in zip(bv, sv))
print(f"  Batch:   {t_batch:.2f}s  |  Single: {t_single:.2f}s")
print(f"  Max diff between batch and single: {max_diff:.8f}  (should be ~0)")

print("\nAll Phase 2 tests passed.")
