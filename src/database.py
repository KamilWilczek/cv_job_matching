"""
Database access layer — PostgreSQL + pgvector.

Schema:
  cvs        — candidates: CV text + embedding vector
  job_offers — job postings: text + embedding vector

pgvector operators:
  <=>  cosine distance      (1 - cosine_similarity) — lower is better
  <->  L2 (Euclidean) distance
  <#>  negative inner product

We use cosine distance (<=>), because:
  - independent of document length
  - standard for text embeddings
  - pgvector can use HNSW index with vector_cosine_ops
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()


def _dsn() -> str:
    return (
        f"host={os.getenv('DB_HOST', 'localhost')} "
        f"port={os.getenv('DB_PORT', '5434')} "
        f"dbname={os.getenv('DB_NAME', 'cv_matching')} "
        f"user={os.getenv('DB_USER', 'cvuser')} "
        f"password={os.getenv('DB_PASSWORD', 'localdev')}"
    )


@contextmanager
def get_conn():
    """Context manager — opens a connection, registers pgvector, closes on exit."""
    conn = psycopg.connect(_dsn())
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

CREATE_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cvs (
    id             SERIAL PRIMARY KEY,
    filename       TEXT UNIQUE NOT NULL,
    candidate_name TEXT,
    raw_text       TEXT NOT NULL,
    embedding      VECTOR(768),
    -- GENERATED ALWAYS AS ... STORED: PostgreSQL auto-computes this column on every INSERT/UPDATE.
    -- 'simple' config = no stemming — better for technical terms (FastAPI, asyncio).
    text_search    TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', raw_text)) STORED,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS job_offers (
    id          SERIAL PRIMARY KEY,
    filename    TEXT UNIQUE NOT NULL,
    job_title   TEXT,
    raw_text    TEXT NOT NULL,
    embedding   VECTOR(768),
    text_search TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', raw_text)) STORED,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for vector search (ANN — Approximate Nearest Neighbor).
-- Builds a layered graph — SELECT is ~100x faster than brute-force for >10k vectors.
-- m=16 (connections per node), ef_construction=64 (graph build accuracy).
CREATE INDEX IF NOT EXISTS cvs_embedding_idx
    ON cvs USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS job_offers_embedding_idx
    ON job_offers USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search (tsvector).
-- GIN = Generalized Inverted Index — fast for @@ operator and ts_rank.
CREATE INDEX IF NOT EXISTS cvs_text_search_idx
    ON cvs USING gin(text_search);

CREATE INDEX IF NOT EXISTS job_offers_text_search_idx
    ON job_offers USING gin(text_search);
"""


def init_schema() -> None:
    """Create tables and indexes (idempotent — safe to call multiple times)."""
    with get_conn() as conn:
        conn.execute(CREATE_SCHEMA)
    print("Schema initialized.")


def drop_schema() -> None:
    """Drop all tables (useful when resetting test data)."""
    with get_conn() as conn:
        conn.execute("DROP TABLE IF EXISTS cvs, job_offers CASCADE;")
    print("Schema dropped.")


# ── CV — insert / select ──────────────────────────────────────────────────────

@dataclass
class CV:
    id: int
    filename: str
    candidate_name: str
    raw_text: str
    embedding: list[float]


def insert_cv(
    filename: str,
    raw_text: str,
    embedding: list[float],
    candidate_name: str = "",
) -> int:
    """Insert a CV. If filename already exists, update it (upsert)."""
    sql = """
        INSERT INTO cvs (filename, candidate_name, raw_text, embedding)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (filename) DO UPDATE
            SET candidate_name = EXCLUDED.candidate_name,
                raw_text       = EXCLUDED.raw_text,
                embedding      = EXCLUDED.embedding
        RETURNING id
    """
    with get_conn() as conn:
        row = conn.execute(sql, (filename, candidate_name, raw_text, embedding)).fetchone()
    return row[0]


def search_cvs(
    query_embedding: list[float],
    top_k: int = 5,
) -> list[tuple[int, str, str, float]]:
    """
    Find the top-k CVs most similar to the query vector.

    Returns: [(id, filename, candidate_name, cosine_similarity), ...]

    Note: <=> is DISTANCE (0 = identical, 2 = opposite).
    We convert to similarity: 1 - distance.
    """
    sql = """
        SELECT id, filename, candidate_name,
               1 - (embedding <=> %s::vector) AS similarity
        FROM cvs
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with get_conn() as conn:
        rows = conn.execute(sql, (query_embedding, query_embedding, top_k)).fetchall()
    return rows


# ── Job offers — insert / select ──────────────────────────────────────────────

def insert_job_offer(
    filename: str,
    raw_text: str,
    embedding: list[float],
    job_title: str = "",
) -> int:
    """Insert a job offer (upsert on filename)."""
    sql = """
        INSERT INTO job_offers (filename, job_title, raw_text, embedding)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (filename) DO UPDATE
            SET job_title = EXCLUDED.job_title,
                raw_text  = EXCLUDED.raw_text,
                embedding = EXCLUDED.embedding
        RETURNING id
    """
    with get_conn() as conn:
        row = conn.execute(sql, (filename, job_title, raw_text, embedding)).fetchone()
    return row[0]


def search_job_offers(
    query_embedding: list[float],
    top_k: int = 5,
) -> list[tuple[int, str, str, float]]:
    """Find top-k job offers similar to the query vector (e.g. a CV vector)."""
    sql = """
        SELECT id, filename, job_title,
               1 - (embedding <=> %s::vector) AS similarity
        FROM job_offers
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with get_conn() as conn:
        rows = conn.execute(sql, (query_embedding, query_embedding, top_k)).fetchall()
    return rows


# ── Stats ─────────────────────────────────────────────────────────────────────

def stats() -> dict:
    """Return the row count for each table."""
    with get_conn() as conn:
        n_cvs  = conn.execute("SELECT COUNT(*) FROM cvs").fetchone()[0]
        n_jobs = conn.execute("SELECT COUNT(*) FROM job_offers").fetchone()[0]
    return {"cvs": n_cvs, "job_offers": n_jobs}
