"""
Guardrails — input and output validation for LLM calls.

Two guard types:

  InputGuard  — checks text before sending to the model
    - length (avoid exceeding context window)
    - absence of explicit prompt injection

  OutputGuard — parses and validates the model's response
    - extracts JSON even when the model added "dirty" text around it
    - checks types and field ranges
    - returns a fallback on error instead of raising an exception

Why guardrails?
  Language models are non-deterministic — even with an identical prompt
  they may return a different format. Guardrails are a contract enforced
  in code, not in the model.
"""

import json
import re

# ── Expected output schema ────────────────────────────────────────────────────

ANALYSIS_SCHEMA = {
    "score":          (int,   lambda v: 0 <= v <= 100),
    "key_matches":    (list,  lambda v: 1 <= len(v) <= 3),
    "gaps":           (list,  lambda v: 0 <= len(v) <= 3),
    "recommendation": (str,   lambda v: v in {"strong_match", "weak_match", "no_match"}),
}

FALLBACK_RESULT = {
    "score": -1,
    "key_matches": [],
    "gaps": [],
    "recommendation": "no_match",
    "_error": "guardrail_failed",
}


# ── Input Guard ───────────────────────────────────────────────────────────────

class InputGuard:
    MAX_CHARS = 3000  # ~750 tokens — safe for tinyllama (2048 token ctx)

    INJECTION_PATTERNS = [
        r"ignore (previous|all) instructions",
        r"forget (everything|your|the) (previous|system)",
        r"you are now",
        r"new instructions:",
        r"###\s*system",
    ]

    def __init__(self):
        self._re = re.compile(
            "|".join(self.INJECTION_PATTERNS), re.IGNORECASE
        )

    def check(self, text: str, label: str = "input") -> str:
        """Validate input text. Returns the (possibly truncated) text."""
        if self._re.search(text):
            raise ValueError(f"[InputGuard] Prompt injection detected in {label}")
        if len(text) > self.MAX_CHARS:
            print(f"  [InputGuard] {label} truncated: {len(text)} -> {self.MAX_CHARS} chars")
            return text[: self.MAX_CHARS] + "\n[...text truncated...]"
        return text


# ── Output Guard ──────────────────────────────────────────────────────────────

def _extract_json(raw: str, prompt_prefix: str = '{"score": ') -> str:
    """
    Extract JSON from a raw model response.

    Handles 3 cases:
      1. Markdown code block: ```json { ... } ```
      2. Full JSON: { ... } somewhere in the response
      3. Prefix completion: model received '{"score": ' and completed the rest —
         we prepend the prefix to the response and locate the complete JSON
    """
    # Attempt 1: markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        return m.group(1)

    # Attempt 2: full JSON somewhere in the response
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if m:
        return m.group(0)

    # Attempt 3: prefix completion — prepend missing prefix and search for JSON
    reconstructed = prompt_prefix + raw
    m = re.search(r"\{.*?\}", reconstructed, re.DOTALL)
    if m:
        return m.group(0)

    return raw  # return raw text; JSON parsing will fail and we'll catch the exception


def _coerce_score(val) -> int:
    """Convert float/str to int, clamp to 0-100."""
    try:
        v = int(float(val))
        return max(0, min(100, v))
    except (TypeError, ValueError):
        return -1


def _coerce_list(val) -> list:
    """Ensure val is a list of strings, max 3 elements."""
    if not isinstance(val, list):
        return []
    return [str(x) for x in val[:3]]


def validate_output(raw: str) -> dict:
    """
    Parse and validate the model's response.

    Strategy: rather than rejecting on error, attempt to repair
    what we can (type coercion). Only return FALLBACK_RESULT on
    complete JSON parse failure.
    """
    json_str = _extract_json(raw)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print(f"  [OutputGuard] JSON parse failed. Raw snippet: {raw[:120]!r}")
        return {**FALLBACK_RESULT, "_raw": raw[:300]}

    result = {}
    errors = []

    # score
    result["score"] = _coerce_score(data.get("score", -1))
    if not (0 <= result["score"] <= 100):
        errors.append(f"score out of range: {result['score']}")

    # key_matches
    result["key_matches"] = _coerce_list(data.get("key_matches", []))

    # gaps
    result["gaps"] = _coerce_list(data.get("gaps", []))

    # recommendation
    rec = str(data.get("recommendation", "")).strip().lower()
    # Repair if model wrote "Strong Match" instead of "strong_match"
    rec = re.sub(r"\s+", "_", rec)
    if rec not in {"strong_match", "weak_match", "no_match"}:
        # Heuristic based on score
        if result["score"] >= 75:
            rec = "strong_match"
        elif result["score"] >= 45:
            rec = "weak_match"
        else:
            rec = "no_match"
        errors.append(f"recommendation coerced from model output to: {rec}")
    result["recommendation"] = rec

    # score <-> recommendation consistency
    # If model wrote "no_match" but score=85, that's clearly a mistake
    score_implied = (
        "strong_match" if result["score"] >= 75
        else "weak_match"  if result["score"] >= 45
        else "no_match"
    )
    if result["recommendation"] != score_implied and result["score"] >= 0:
        errors.append(
            f"recommendation '{result['recommendation']}' inconsistent with score={result['score']}, "
            f"corrected to '{score_implied}'"
        )
        result["recommendation"] = score_implied

    if errors:
        result["_warnings"] = errors

    return result
