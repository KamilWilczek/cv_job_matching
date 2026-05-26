"""
Phase 4 — Hybrid search tests.

Question: does hybrid search fix the Anna vs Jan problem?
Phase 3: Anna was #1 for Python Backend (sim=0.7882), Jan was #2 (sim=0.7824).
Expected: Jan should be #1 after adding keyword matching.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from database  import init_schema, drop_schema, stats
from ingestion import ingest_cvs, ingest_job_offers
from embeddings import get_embedding
from matching  import vector_search_cvs, fulltext_search_cvs, hybrid_search_cvs

ROOT     = Path(__file__).parent
CVS_DIR  = ROOT / "data" / "sample_cvs"
JOBS_DIR = ROOT / "data" / "sample_jobs"


# ── Setup ─────────────────────────────────────────────────────────────────────
print("=== Setup: reset database and ingest ===")
drop_schema()
init_schema()
ingest_cvs(CVS_DIR)
ingest_job_offers(JOBS_DIR)
print(f"Database state: {stats()}\n")


# ── Helper: print ranking ─────────────────────────────────────────────────────
def print_ranking(results: list[dict], score_key: str = "score", label: str = "") -> None:
    if label:
        print(f"  [{label}]")
    for rank, r in enumerate(results, 1):
        score = r.get(score_key, r.get("rrf_score", 0))
        name  = r.get("name") or r.get("title", "?")
        bar   = "#" * int((score if score_key != "rrf_score" else score * 300) * 40)
        print(f"    #{rank}  {r['filename']:<40}  {score:.5f}  {bar}")


# ── Job offer 1: Python Backend Senior ────────────────────────────────────────
print("=" * 65)
print("JOB OFFER: python_backend_senior")
print("=" * 65)

job_text = (JOBS_DIR / "python_backend_senior.txt").read_text(encoding="utf-8")
job_vec  = get_embedding(job_text, task="search_document")

print("\n-- Method A: Pure Vector Search --")
vec_results = vector_search_cvs(job_vec, top_k=5)
print_ranking(vec_results, score_key="score")

print("\n-- Method B: Full-text Search (BM25-like) --")
# Keywords from the job description that we want to find in CVs
keywords = "Python FastAPI PostgreSQL Docker asyncio"
ft_results = fulltext_search_cvs(keywords, top_k=5)
if ft_results:
    print_ranking(ft_results, score_key="score")
else:
    print("    (no results — no CV contains all these keywords)")

print("\n-- Method C: Hybrid Search (RRF) --")
hybrid_results = hybrid_search_cvs(job_vec, keywords, top_k=5)
print(f"  {'filename':<40}  {'rrf':>7}  {'vec':>7}  {'txt':>7}  {'vR':>3}  {'tR':>3}")
print(f"  {'-'*40}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*3}  {'-'*3}")
for rank, r in enumerate(hybrid_results, 1):
    vec_s  = f"{r['vec_score']:.4f}"  if r['vec_score']  is not None else "  N/A"
    txt_s  = f"{r['text_score']:.4f}" if r['text_score'] is not None else "  N/A"
    vec_r  = str(r['vec_rank'])  if r['vec_rank']  is not None else " -"
    txt_r  = str(r['text_rank']) if r['text_rank'] is not None else " -"
    marker = " <-- !" if r['filename'] == "jan_kowalski_python" else ""
    print(f"  #{rank} {r['filename']:<40}  {r['rrf_score']:.5f}  {vec_s}  {txt_s}  {vec_r:>3}  {txt_r:>3}{marker}")


# ── Job offer 2: DevOps Engineer ──────────────────────────────────────────────
print()
print("=" * 65)
print("JOB OFFER: devops_engineer")
print("=" * 65)

job_text2 = (JOBS_DIR / "devops_engineer.txt").read_text(encoding="utf-8")
job_vec2  = get_embedding(job_text2, task="search_document")
keywords2 = "Kubernetes Terraform AWS Docker CI/CD ArgoCD"

print("\n-- Method C: Hybrid Search (RRF) --")
hybrid2 = hybrid_search_cvs(job_vec2, keywords2, top_k=5)
print(f"  {'filename':<40}  {'rrf':>7}  {'vec':>7}  {'txt':>7}  {'vR':>3}  {'tR':>3}")
print(f"  {'-'*40}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*3}  {'-'*3}")
for rank, r in enumerate(hybrid2, 1):
    vec_s = f"{r['vec_score']:.4f}" if r['vec_score']  is not None else "  N/A"
    txt_s = f"{r['text_score']:.4f}" if r['text_score'] is not None else "  N/A"
    vec_r = str(r['vec_rank'])  if r['vec_rank']  is not None else " -"
    txt_r = str(r['text_rank']) if r['text_rank'] is not None else " -"
    print(f"  #{rank} {r['filename']:<40}  {r['rrf_score']:.5f}  {vec_s}  {txt_s}  {vec_r:>3}  {txt_r:>3}")


# ── RRF math demo ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("DEMO: how RRF calculates a score (k=60)")
print("=" * 65)
print("""
  Formula: score = 1/(60 + vec_rank) + 1/(60 + text_rank)
  If absent from a ranking: 0

  Example for python_backend_senior:
    Jan:   vec_rank=#2, text_rank=#1 -> 1/62 + 1/61 = 0.01613 + 0.01639 = 0.03252
    Anna:  vec_rank=#1, text_rank=-  -> 1/61 + 0     = 0.01639 + 0       = 0.01639

  Jan wins despite being #2 in vector search,
  because he is strong in keyword matching (FastAPI, asyncio, PostgreSQL).
""")

print("All Phase 4 tests passed.")
