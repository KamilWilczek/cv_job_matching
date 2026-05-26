"""
Embedding module — convert text into numerical vectors.

Uses nomic-embed-text via Ollama (local, free).
Model returns 768-dimensional vectors.

Task prefixes (important!):
  "search_document: <text>"  — text being indexed (CV, job offer)
  "search_query: <text>"     — query being compared against documents

Without prefixes the model operates in "clustering" mode and poorly
distinguishes semantically close vs. distant texts.
"""

import os
from dotenv import load_dotenv
from litellm import embedding

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ollama/nomic-embed-text")
EMBEDDING_DIM   = int(os.getenv("EMBEDDING_DIM", "768"))
OLLAMA_BASE     = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE


def get_embedding(text: str, task: str = "search_document") -> list[float]:
    """
    Convert text into an embedding vector.

    Args:
        text: Text to embed (CV, job description).
        task: "search_document" for content, "search_query" for queries.

    Returns:
        List of 768 floats representing the text in semantic space.
    """
    prefixed = f"{task}: {text}"
    response = embedding(model=EMBEDDING_MODEL, input=[prefixed])
    return response.data[0]["embedding"]


def get_embeddings_batch(texts: list[str], task: str = "search_document") -> list[list[float]]:
    """
    Embed multiple texts in a single API call (more efficient than N individual calls).

    Note: Ollama processes the batch sequentially, but saves network overhead.
    """
    prefixed = [f"{task}: {t}" for t in texts]
    response = embedding(model=EMBEDDING_MODEL, input=prefixed)
    return [item["embedding"] for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two vectors. Range: -1.0 to 1.0.

    Why cosine and not Euclidean distance?
    Cosine measures the ANGLE between vectors, not geometric distance.
    For text: two documents on the same topic but different lengths
    have a similar angle but may have large Euclidean distance.
    Cosine is therefore length-independent.
    """
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_by_similarity(
    query_vec: list[float],
    candidates: list[tuple[str, list[float]]],
) -> list[tuple[str, float]]:
    """
    Sort candidates by similarity to the query vector (descending).

    Args:
        query_vec:  Query vector (e.g. embedding of a job offer).
        candidates: List of (label, vector) pairs to compare (e.g. CVs).

    Returns:
        List of (label, score) sorted from most to least similar.
    """
    scored = [(label, cosine_similarity(query_vec, vec)) for label, vec in candidates]
    return sorted(scored, key=lambda x: x[1], reverse=True)
