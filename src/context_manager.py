"""
Context Manager — conversation history management for LLM.

Problem:
  Models have a limited context window.
  qwen2.5:0.5b: ~32k tokens, tinyllama: ~2k tokens.
  When conversation history exceeds the limit, the model "forgets" or errors.

Solution — History Summary Pattern:
  Instead of truncating older messages (losing information),
  compress them into a short summary.

  Managed history structure:
  +-------------------------------------+
  | [0] system prompt (always)          |
  | [1] {"role":"assistant",            | <- compressed old messages
  |       "content":"[Summary]: ..."}   |
  | [2] user message (n-2)              | <- last KEEP_LAST fresh messages
  | [3] assistant message (n-2)         |
  | [4] user message (n-1)              |
  | [5] assistant message (n-1)         |
  +-------------------------------------+

Other techniques:
  - Sliding window: drops old messages (simple, but loses information)
  - RAG memory: retrieves relevant history fragments (complex)
  - Structured memory: extracts facts to a database (agents)
"""

import os
from dotenv import load_dotenv
from litellm import completion, token_counter

load_dotenv()

LLM_MODEL  = os.getenv("LLM_MODEL", "ollama/qwen2.5:0.5b")
os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

# Tokens reserved for the model's response
RESPONSE_BUDGET = 400
# Number of most recent messages always kept in full (never compressed)
KEEP_LAST = 4


def count_tokens(messages: list[dict]) -> int:
    """
    Count tokens in a list of messages.

    litellm's token_counter uses tiktoken for known models.
    For Ollama/local models: approximation via text length (~4 chars/token).
    """
    try:
        return token_counter(model=LLM_MODEL, messages=messages)
    except Exception:
        # Fallback: ~4 chars per token (English); Polish is closer to ~5
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 4


def _summarize_messages(messages: list[dict]) -> str:
    """
    Compress a list of messages into a short summary using the LLM.
    Triggers one extra request to the model — cost: ~1 extra request.
    """
    if not messages:
        return ""

    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )
    prompt = (
        "Summarize this conversation in 3 sentences. "
        "Keep all important facts, names, scores, and decisions.\n\n"
        f"{transcript}"
    )
    try:
        resp = completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return f"[Previous conversation truncated. Last topics: {transcript[-300:]}]"


class HistoryManager:
    """
    Manages conversation history — automatic compression when it grows too long.

    Usage:
        hm = HistoryManager(system_prompt="You are a helpful assistant", max_tokens=1500)
        hm.add_user("Which CV fits this job offer?")
        messages = hm.get_messages()          # ready to send to LLM
        response = completion(model=..., messages=messages)
        hm.add_assistant(response.choices[0].message.content)
    """

    def __init__(self, system_prompt: str, max_tokens: int = 1800):
        self.system_prompt = system_prompt
        self.max_tokens    = max_tokens
        self._history: list[dict] = []  # excludes system prompt
        self._summary: str = ""         # compressed older messages

    def add_user(self, content: str) -> None:
        self._history.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._history.append({"role": "assistant", "content": content})

    def get_messages(self) -> list[dict]:
        """
        Return the full message list ready to send to the LLM.
        If too long — trigger compression.
        """
        messages = self._build_messages()
        used = count_tokens(messages)

        if used + RESPONSE_BUDGET > self.max_tokens:
            print(f"\n  [ContextManager] History too long ({used} tokens) "
                  f"> limit {self.max_tokens - RESPONSE_BUDGET}. Compressing...")
            self._compress()
            messages = self._build_messages()
            used_after = count_tokens(messages)
            print(f"  [ContextManager] After compression: {used_after} tokens.\n")

        return messages

    def token_usage(self) -> dict:
        """Return token diagnostics."""
        messages = self._build_messages()
        used = count_tokens(messages)
        return {
            "used":         used,
            "budget":       self.max_tokens,
            "remaining":    self.max_tokens - used - RESPONSE_BUDGET,
            "turns":        len(self._history) // 2,
            "has_summary":  bool(self._summary),
        }

    def reset(self) -> None:
        self._history = []
        self._summary = ""

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_messages(self) -> list[dict]:
        """Assemble [system] + [summary?] + [history] into one list."""
        msgs = [{"role": "system", "content": self.system_prompt}]
        if self._summary:
            msgs.append({
                "role":    "assistant",
                "content": f"[Summary of earlier conversation]: {self._summary}",
            })
        msgs.extend(self._history)
        return msgs

    def _compress(self) -> None:
        """
        Compress older history messages.
        Keeps the last KEEP_LAST messages intact.
        """
        if len(self._history) <= KEEP_LAST:
            return  # not enough history to compress

        to_compress = self._history[:-KEEP_LAST]
        fresh       = self._history[-KEEP_LAST:]

        new_summary = _summarize_messages(to_compress)

        # If we already have a summary, compress it too
        if self._summary:
            combined = f"{self._summary}\n\nLATER: {new_summary}"
            self._summary = combined[:600]  # hard cap on summary length
        else:
            self._summary = new_summary

        self._history = fresh
