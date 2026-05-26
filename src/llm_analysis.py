"""
LLM Analysis — evaluate how well a CV matches a job offer.

Uses the RICEFACT prompt framework and qwen2.5:0.5b (via litellm).
Output is validated by guardrails before being returned.

Flow:
  cv_text + job_text
       |
       v
  InputGuard (length, injection)
       |
       v
  _build_prompt() — RICEFACT
       |
       v
  litellm.completion() — qwen2.5:0.5b via Ollama
       |
       v
  validate_output() — OutputGuard
       |
       v
  dict { score, key_matches, gaps, recommendation }
"""

import os
import re

from dotenv import load_dotenv
from litellm import completion

from src.guardrails import InputGuard, validate_output

load_dotenv()

os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

LLM_MODEL   = os.getenv("LLM_MODEL", "ollama/tinyllama")
MAX_TOKENS  = 400   # enough for JSON with ~3 items per field
TEMPERATURE = 0.1   # low randomness — we want deterministic output

_guard = InputGuard()


# ── Prompt (RICEFACT) ─────────────────────────────────────────────────────────

def _build_prompt(cv_text: str, job_text: str) -> str:
    """
    Build a prompt following the RICEFACT framework — optimised for small models.

    Problem with small models (~1B): they respond to "write JSON" instructions
    by describing how to write JSON rather than actually writing it.

    Solution: "prefix completion" — start the JSON in the prompt.
    The model doesn't describe how to fill in the JSON; it just continues it.

    We use short excerpts of the CV/offer (first 600 chars) to stay within
    the 2048-token context window of tinyllama.

    RICEFACT applied:
      R: "You are a recruiter" — role in one line
      I: "Score this candidate" — concrete instruction
      C: truncated CV and offer excerpts
      E: format made visible by the opening { in "JSON:"
      F: explicit via the opening brace
      C: JSON field comments (score 0-100 etc.)
    """
    cv_short  = cv_text[:600].replace("{", "(").replace("}", ")")
    job_short = job_text[:600].replace("{", "(").replace("}", ")")

    return f"""[JOB]
{job_short}

[CV]
{cv_short}

[TASK] Score this candidate 0-100. Output only JSON:
{{"score": <0-100>, "key_matches": ["match1"], "gaps": ["gap1"], "recommendation": "strong_match|weak_match|no_match"}}

JSON output:"""


# ── Main analysis function ────────────────────────────────────────────────────

def analyze_match(cv_text: str, job_text: str) -> dict:
    """
    Analyse how well a CV matches a job offer using an LLM.

    Returns:
        dict with keys: score, key_matches, gaps, recommendation
        On LLM error: { score: -1, _error: "..." }
    """
    # Input guardrails
    cv_safe  = _guard.check(cv_text,  label="CV")
    job_safe = _guard.check(job_text, label="job_offer")

    prompt = _build_prompt(cv_safe, job_safe)

    # Few-shot in conversation: show the model a COMPLETE example answer
    # before asking the real question. The model copies the format rather
    # than describing how to use it — most effective technique for small models.
    messages = [
        {
            "role": "system",
            "content": "You are a JSON API. You output only valid JSON. No explanations.",
        },
        {
            "role": "user",
            "content": (
                "[JOB]\nWe need graphic designer with Photoshop and brand identity skills.\n\n"
                "[CV]\nMaria: Illustrator 6yr, Figma, brand campaigns for L'Oreal.\n\n"
                "[TASK] Score 0-100. JSON output:"
            ),
        },
        {
            # Example in a completely different domain (design, not IT backend)
            # so the model doesn't copy it when asked about Python developers
            "role": "assistant",
            "content": '{"score": 75, "key_matches": ["Illustrator similar to Photoshop", "brand identity experience"], "gaps": ["Photoshop not explicitly mentioned"], "recommendation": "weak_match"}',
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        response = completion(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        return {"score": -1, "key_matches": [], "gaps": [], "recommendation": "no_match",
                "_error": str(e)[:200]}

    result = validate_output(raw)
    return result


def analyze_top_candidates(
    job_text: str,
    candidates: list[dict],
    cv_dir,
) -> list[dict]:
    """
    Analyse top candidates (hybrid search results) using an LLM.

    Args:
        job_text:   Full text of the job offer.
        candidates: List of results from hybrid_search_cvs().
        cv_dir:     Path to directory containing CV files (for reading text).

    Returns:
        List of candidates enriched with an "analysis" field from the LLM.
    """
    results = []
    for c in candidates:
        cv_path = cv_dir / f"{c['filename']}.txt"
        if not cv_path.exists():
            print(f"  [WARN] File not found: {cv_path}")
            continue

        cv_text = cv_path.read_text(encoding="utf-8")
        print(f"  Analysing: {c['filename']} ...", end=" ", flush=True)

        analysis = analyze_match(cv_text, job_text)
        c_enriched = {**c, "analysis": analysis}
        results.append(c_enriched)

        score = analysis.get("score", -1)
        rec   = analysis.get("recommendation", "?")
        print(f"score={score}  [{rec}]")

    return results


def format_analysis_report(enriched_candidates: list[dict]) -> str:
    """Format results into a human-readable text report."""
    lines = []
    for rank, c in enumerate(enriched_candidates, 1):
        a = c.get("analysis", {})
        lines.append(f"\n#{rank}  {c['filename']}  (rrf={c.get('rrf_score', 0):.4f})")
        lines.append(f"     LLM score:       {a.get('score', '?')}/100")
        lines.append(f"     Recommendation:  {a.get('recommendation', '?')}")
        if a.get("key_matches"):
            lines.append("     Key matches:")
            for m in a["key_matches"]:
                lines.append(f"       + {m}")
        if a.get("gaps"):
            lines.append("     Gaps:")
            for g in a["gaps"]:
                lines.append(f"       - {g}")
        if a.get("_warnings"):
            lines.append(f"     [guardrail warnings: {a['_warnings']}]")
        if a.get("_error"):
            lines.append(f"     [ERROR: {a['_error']}]")
    return "\n".join(lines)
