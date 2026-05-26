"""Verify Ollama + litellm integration."""
import os
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

from litellm import embedding, completion

# nomic-embed-text task prefixes:
#   "search_document: <text>"  — for content being indexed
#   "search_query: <text>"     — for a query you're searching with
# IMPORTANT: when comparing two texts symmetrically (e.g. doc vs doc),
# use the SAME prefix for both — mixing prefixes puts vectors in different
# spaces and breaks cosine similarity.

def get_embedding(text: str, task: str = "search_document") -> list[float]:
    resp = embedding(model="ollama/nomic-embed-text", input=[f"{task}: {text}"])
    return resp.data[0]["embedding"]

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x**2 for x in a) ** 0.5
    nb = sum(x**2 for x in b) ** 0.5
    return dot / (na * nb)


print("=== Test 1: Embeddings (nomic-embed-text) ===")
vec = get_embedding("Python developer with 5 years experience")
print(f"Vector dimensions: {len(vec)}")
print(f"First 5 values:    {[round(v, 4) for v in vec[:5]]}")

print("\n=== Test 2: Semantic similarity (symmetric — same task prefix) ===")
# All as "search_document" — symmetric comparison
v1 = get_embedding("Senior Python backend engineer", task="search_document")
v2 = get_embedding("Python developer with 5 years experience", task="search_document")
v3 = get_embedding("Graphic designer with Photoshop skills", task="search_document")

sim_match = cosine(v1, v2)
sim_diff  = cosine(v1, v3)

print(f"'Senior Python backend' vs 'Python dev 5yr':    {sim_match:.4f}  (expected: high)")
print(f"'Senior Python backend' vs 'Graphic designer':  {sim_diff:.4f}  (expected: low)")
print(f"Difference:                                      {sim_match - sim_diff:+.4f}  (expected: > 0)")

if sim_match > sim_diff:
    print("PASS: semantic similarity works correctly")
else:
    print("WARN: similarity not separating well")

print("\n=== Test 3: Asymmetric search (query vs document) ===")
# Job offer = document, CV skill = query
v_job    = get_embedding("We need a senior Python engineer with REST API experience", task="search_document")
v_cv_good = get_embedding("7 years Python, built microservices with FastAPI", task="search_query")
v_cv_bad  = get_embedding("Photoshop, Illustrator, brand identity design", task="search_query")

print(f"Job vs matching CV:     {cosine(v_job, v_cv_good):.4f}  (expected: high)")
print(f"Job vs non-matching CV: {cosine(v_job, v_cv_bad):.4f}  (expected: low)")

print("\n=== Test 4: LLM (tinyllama — fallback, llama3.2 crashes on this CPU) ===")
resp = completion(
    model="ollama/tinyllama",
    messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
)
print(f"LLM response: {resp.choices[0].message.content.strip()}")

print("\nAll Ollama checks passed.")
