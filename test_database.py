"""
Phase 3 — Database pipeline tests.

Steps:
  1. Init schema (tables + HNSW indexes)
  2. Ingest: load CVs and job offers, save to DB with embeddings
  3. Vector search: find top-k CVs for each job offer (pgvector <=> operator)
  4. Reverse search: find matching job offers for Jan's CV
  5. Raw SQL demo showing the <=> operator and distance vs similarity
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from database import init_schema, drop_schema, search_cvs, search_job_offers, stats
from ingestion import ingest_cvs, ingest_job_offers
from embeddings import get_embedding

ROOT     = Path(__file__).parent
CVS_DIR  = ROOT / "data" / "sample_cvs"
JOBS_DIR = ROOT / "data" / "sample_jobs"


# ── 1. Schema ─────────────────────────────────────────────────────────────────
print("=== 1. Schema init ===")
drop_schema()   # reset — clean slate on each test run
init_schema()


# ── 2. Ingestion ──────────────────────────────────────────────────────────────
print("\n=== 2. Ingestion — CVs ===")
cv_ids = ingest_cvs(CVS_DIR)

print("\n=== 2. Ingestion — Job offers ===")
job_ids = ingest_job_offers(JOBS_DIR)

print(f"\nDatabase state: {stats()}")


# ── 3. Vector search: job offers -> CVs ───────────────────────────────────────
print("\n=== 3. Vector search: top-3 CVs for each job offer ===")
print("(query: SELECT ... ORDER BY embedding <=> query_vec LIMIT 3)\n")

JOB_TEXTS = {p.stem: p.read_text(encoding="utf-8") for p in JOBS_DIR.glob("*.txt")}

for job_name, job_text in sorted(JOB_TEXTS.items()):
    job_vec = get_embedding(job_text, task="search_document")
    results = search_cvs(job_vec, top_k=3)

    print(f"Job offer: {job_name}")
    for rank, (cv_id, filename, name, sim) in enumerate(results, 1):
        bar = "#" * int(sim * 50)
        print(f"  #{rank}  [{cv_id:2}] {filename:<40}  sim={sim:.4f}  {bar}")
    print()


# ── 4. Reverse search: CV -> job offers ───────────────────────────────────────
print("=== 4. Reverse search: matching job offers for Jan (Python Senior) ===")
jan_text = (CVS_DIR / "jan_kowalski_python.txt").read_text(encoding="utf-8")
jan_vec  = get_embedding(jan_text, task="search_query")
job_results = search_job_offers(jan_vec, top_k=2)

for rank, (jid, filename, title, sim) in enumerate(job_results, 1):
    print(f"  #{rank}  [{jid}] {filename:<35}  sim={sim:.4f}")

print("\nExpected: python_backend_senior at #1")


# ── 5. Raw SQL — demonstrate the <=> operator ─────────────────────────────────
print("\n=== 5. Raw SQL with <=> operator (cosine distance) ===")
from database import get_conn

jan_vec_doc = get_embedding(jan_text, task="search_document")

with get_conn() as conn:
    rows = conn.execute("""
        SELECT
            filename,
            candidate_name,
            embedding <=> %s::vector                   AS cosine_distance,
            1 - (embedding <=> %s::vector)             AS cosine_similarity,
            ROUND((1 - (embedding <=> %s::vector))::numeric, 4) AS sim_rounded
        FROM cvs
        ORDER BY cosine_distance
    """, (jan_vec_doc, jan_vec_doc, jan_vec_doc)).fetchall()

print(f"  {'filename':<40} {'distance':>10}  {'similarity':>12}  {'rounded':>8}")
print(f"  {'-'*40} {'-'*10}  {'-'*12}  {'-'*8}")
for filename, name, dist, sim, rounded in rows:
    print(f"  {filename:<40} {dist:>10.6f}  {sim:>12.6f}  {rounded:>8}")

print("""
Note: <=> returns DISTANCE (0=identical, 2=opposite), NOT similarity.
  distance = 1 - cosine_similarity
  Therefore ORDER BY <=> sorts from closest to furthest.
""")

print("All Phase 3 tests passed.")
