"""
Search module — three modes:

  1. vector_search   — cosine similarity on embeddings (semantic)
  2. fulltext_search — BM25-like ts_rank_cd (keyword matching)
  3. hybrid_search   — RRF (Reciprocal Rank Fusion) combines both

Why hybrid search?
  - Vector search: understands semantics ("senior engineer" ~ "experienced developer"),
    but struggles to distinguish "Python senior" from "Frontend senior" on small differences.
  - Fulltext search: catches exact keywords (FastAPI, asyncio, PostgreSQL),
    but doesn't understand synonyms or context.
  - Hybrid (RRF): best of both — a document winning both rankings wins overall,
    while documents from only one ranking still have a chance.

RRF (Reciprocal Rank Fusion):
  score(doc) = Sum 1 / (k + rank_i(doc))
  k = 60 (standard value, dampens the dominance of top ranks)
  Example: doc at #1 in both  = 1/61 + 1/61 = 0.0328
           doc at #1 in one   = 1/61 + 0     = 0.0164
"""

from src.database import get_conn


# ── 1. Vector search ─────────────────────────────────────────────────────────

def vector_search_cvs(
    job_vec: list[float],
    top_k: int = 10,
) -> list[dict]:
    """
    Find CVs by cosine similarity on embeddings.
    Good for: semantic matching, synonyms, general meaning.
    Weak for: precise technical terms, proper nouns.
    """
    sql = """
        SELECT id, filename, candidate_name,
               1 - (embedding <=> %s::vector) AS score
        FROM cvs
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with get_conn() as conn:
        rows = conn.execute(sql, (job_vec, job_vec, top_k)).fetchall()
    return [{"id": r[0], "filename": r[1], "name": r[2], "score": float(r[3])} for r in rows]


def vector_search_jobs(
    cv_vec: list[float],
    top_k: int = 10,
) -> list[dict]:
    """Find job offers matching a CV vector."""
    sql = """
        SELECT id, filename, job_title,
               1 - (embedding <=> %s::vector) AS score
        FROM job_offers
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with get_conn() as conn:
        rows = conn.execute(sql, (cv_vec, cv_vec, top_k)).fetchall()
    return [{"id": r[0], "filename": r[1], "title": r[2], "score": float(r[3])} for r in rows]


# ── 2. Full-text search (BM25-like) ──────────────────────────────────────────

def fulltext_search_cvs(
    query_text: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Find CVs by keyword matching (PostgreSQL tsvector + ts_rank_cd).

    ts_rank_cd ≈ BM25: accounts for term frequency and document length.
    plainto_tsquery: converts plain text into a query (AND between words).

    Good for: exact technical terms (FastAPI, Kubernetes, React).
    Weak for: synonyms and semantic associations.
    """
    sql = """
        SELECT id, filename, candidate_name,
               ts_rank_cd(text_search, plainto_tsquery('simple', %s)) AS score
        FROM cvs
        WHERE text_search @@ plainto_tsquery('simple', %s)
        ORDER BY score DESC
        LIMIT %s
    """
    with get_conn() as conn:
        rows = conn.execute(sql, (query_text, query_text, top_k)).fetchall()
    return [{"id": r[0], "filename": r[1], "name": r[2], "score": float(r[3])} for r in rows]


def fulltext_search_jobs(
    query_text: str,
    top_k: int = 10,
) -> list[dict]:
    """Find job offers by keyword matching."""
    sql = """
        SELECT id, filename, job_title,
               ts_rank_cd(text_search, plainto_tsquery('simple', %s)) AS score
        FROM job_offers
        WHERE text_search @@ plainto_tsquery('simple', %s)
        ORDER BY score DESC
        LIMIT %s
    """
    with get_conn() as conn:
        rows = conn.execute(sql, (query_text, query_text, top_k)).fetchall()
    return [{"id": r[0], "filename": r[1], "title": r[2], "score": float(r[3])} for r in rows]


# ── 3. Hybrid search — RRF ───────────────────────────────────────────────────

def hybrid_search_cvs(
    job_vec: list[float],
    query_text: str,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Hybrid search: combines vector search and fulltext search via RRF.

    RRF formula:
      score(doc) = Sum_i  1 / (rrf_k + rank_i(doc))

    Documents present in both rankings receive a double bonus.
    Documents from only one ranking receive a single bonus.
    rrf_k=60 dampens top-1 dominance (without k, #1=1.0, #2=0.5 — too aggressive).

    Entire logic in SQL — one round-trip to the database.
    """
    sql = """
        WITH vector_ranked AS (
            SELECT id,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank,
                   1 - (embedding <=> %(vec)s::vector)                        AS vec_score
            FROM cvs
            ORDER BY embedding <=> %(vec)s::vector
            LIMIT 20
        ),
        text_ranked AS (
            SELECT id,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank_cd(text_search, plainto_tsquery('simple', %(q)s)) DESC
                   ) AS rank,
                   ts_rank_cd(text_search, plainto_tsquery('simple', %(q)s)) AS text_score
            FROM cvs
            WHERE text_search @@ plainto_tsquery('simple', %(q)s)
            ORDER BY text_score DESC
            LIMIT 20
        ),
        rrf AS (
            SELECT
                COALESCE(v.id, t.id)                             AS id,
                COALESCE(1.0 / (%(k)s + v.rank), 0)
                    + COALESCE(1.0 / (%(k)s + t.rank), 0)        AS rrf_score,
                v.vec_score,
                t.text_score,
                v.rank  AS vec_rank,
                t.rank  AS text_rank
            FROM vector_ranked v
            FULL OUTER JOIN text_ranked t ON v.id = t.id
        )
        SELECT
            cvs.id,
            cvs.filename,
            cvs.candidate_name,
            rrf.rrf_score,
            rrf.vec_score,
            rrf.text_score,
            rrf.vec_rank,
            rrf.text_rank
        FROM rrf
        JOIN cvs ON cvs.id = rrf.id
        ORDER BY rrf_score DESC
        LIMIT %(top_k)s
    """
    params = {"vec": job_vec, "q": query_text, "k": rrf_k, "top_k": top_k}
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [
        {
            "id":         r[0],
            "filename":   r[1],
            "name":       r[2],
            "rrf_score":  float(r[3]),
            "vec_score":  float(r[4]) if r[4] is not None else None,
            "text_score": float(r[5]) if r[5] is not None else None,
            "vec_rank":   r[6],
            "text_rank":  r[7],
        }
        for r in rows
    ]
