"""
Ingestion pipeline — read text files, generate embeddings, save to database.

Flow:
  .txt file -> raw_text -> get_embedding() -> insert_cv() / insert_job_offer()

Candidate/position name is extracted from the first line of the file
(convention: "First Last — Title" or "Position: Title").
"""

import re
from pathlib import Path

from src.embeddings import get_embedding, get_embeddings_batch
from src.database import insert_cv, insert_job_offer


def _extract_name(text: str, pattern: str = r"^(.+?)\s*[—–-]") -> str:
    """Extract name from the first line: 'Jan Kowalski — Senior Python' -> 'Jan Kowalski'."""
    first_line = text.strip().splitlines()[0]
    m = re.match(pattern, first_line)
    return m.group(1).strip() if m else first_line.strip()


def ingest_cvs(cvs_dir: Path, batch: bool = True) -> list[int]:
    """
    Read all .txt files from a directory, embed them, and save to the cvs table.

    Args:
        cvs_dir: Path to directory containing CV files.
        batch:   True = one request to Ollama for all CVs (faster).

    Returns:
        List of inserted/updated record IDs.
    """
    files = sorted(cvs_dir.glob("*.txt"))
    if not files:
        print(f"  No .txt files found in {cvs_dir}")
        return []

    texts     = [f.read_text(encoding="utf-8") for f in files]
    filenames = [f.stem for f in files]
    names     = [_extract_name(t) for t in texts]

    print(f"  Generating embeddings for {len(files)} CVs...", end=" ", flush=True)
    if batch:
        embeddings = get_embeddings_batch(texts, task="search_document")
    else:
        embeddings = [get_embedding(t, task="search_document") for t in texts]
    print("done.")

    ids = []
    for filename, name, text, emb in zip(filenames, names, texts, embeddings):
        row_id = insert_cv(filename=filename, raw_text=text, embedding=emb, candidate_name=name)
        ids.append(row_id)
        print(f"    [{row_id}] {filename} ({name})")

    return ids


def ingest_job_offers(jobs_dir: Path, batch: bool = True) -> list[int]:
    """
    Read all .txt files from a directory, embed them, and save to the job_offers table.
    """
    files = sorted(jobs_dir.glob("*.txt"))
    if not files:
        print(f"  No .txt files found in {jobs_dir}")
        return []

    texts     = [f.read_text(encoding="utf-8") for f in files]
    filenames = [f.stem for f in files]
    titles    = [_extract_name(t, pattern=r"Stanowisko:\s*(.+)") or _extract_name(t) for t in texts]

    print(f"  Generating embeddings for {len(files)} job offers...", end=" ", flush=True)
    if batch:
        embeddings = get_embeddings_batch(texts, task="search_document")
    else:
        embeddings = [get_embedding(t, task="search_document") for t in texts]
    print("done.")

    ids = []
    for filename, title, text, emb in zip(filenames, titles, texts, embeddings):
        row_id = insert_job_offer(filename=filename, raw_text=text, embedding=emb, job_title=title)
        ids.append(row_id)
        print(f"    [{row_id}] {filename} ({title})")

    return ids
