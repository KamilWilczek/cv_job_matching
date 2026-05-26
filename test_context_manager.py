"""
Phase 6 — HistoryManager and count_tokens tests.

Tests:
  1. count_tokens — basic token counting
  2. HistoryManager — message building
  3. HistoryManager — compression when limit is exceeded
  4. HistoryManager — full flow: add many messages, verify compression
  5. reset() — clears history and summary
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

from context_manager import HistoryManager, count_tokens


# ── Test 1: count_tokens ──────────────────────────────────────────────────────

def test_count_tokens():
    print("\n=== Test 1: count_tokens ===")

    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you!"},
    ]
    n = count_tokens(msgs)
    print(f"  3 short messages -> {n} tokens")
    assert n > 0, "count_tokens must return > 0"
    assert n < 200, f"Too many tokens for simple messages: {n}"
    print("  [OK]")


# ── Test 2: HistoryManager — build_messages ───────────────────────────────────

def test_build_messages():
    print("\n=== Test 2: HistoryManager — message building ===")

    hm = HistoryManager(system_prompt="You are a test assistant.", max_tokens=2000)

    # Empty history
    msgs = hm._build_messages()
    assert len(msgs) == 1, f"Empty history should give 1 message (system), got {len(msgs)}"
    assert msgs[0]["role"] == "system"
    print(f"  Empty history: {len(msgs)} message (system) [OK]")

    # Add one exchange
    hm.add_user("Hello!")
    hm.add_assistant("Hi, how can I help?")
    msgs = hm._build_messages()
    assert len(msgs) == 3, f"After 1 turn: 3 messages expected, got {len(msgs)}"
    print(f"  After 1 turn: {len(msgs)} messages [OK]")

    # Check token_usage
    usage = hm.token_usage()
    print(f"  token_usage: {usage}")
    assert usage["turns"] == 1
    assert usage["has_summary"] == False
    assert usage["used"] > 0
    print("  [OK]")


# ── Test 3: HistoryManager — compression ─────────────────────────────────────

def test_compression():
    print("\n=== Test 3: HistoryManager — compression ===")

    # Set a very low limit to force compression
    hm = HistoryManager(system_prompt="You are a test assistant.", max_tokens=300)

    # Fill history with long messages
    long_msg = "This is a very long test message for compression testing. " * 20
    for i in range(4):
        hm.add_user(f"Question number {i+1}: {long_msg}")
        hm.add_assistant(f"Answer number {i+1}: {long_msg}")

    before = hm.token_usage()
    print(f"  Before get_messages(): {before['used']} tokens, {before['turns']} turns")

    # get_messages() should trigger compression
    print("  Calling get_messages() — expecting compression...")
    msgs = hm.get_messages()

    after = hm.token_usage()
    print(f"  After compression: {after['used']} tokens, {after['turns']} turns")
    print(f"  has_summary: {after['has_summary']}")

    assert after["has_summary"], "Should have a summary after compression"
    # History should be shorter — KEEP_LAST=4 most recent messages
    assert after["turns"] <= 2, f"After compression max 2 turns (KEEP_LAST=4), got {after['turns']}"
    print("  [OK]")


# ── Test 4: Full flow with real LLM ──────────────────────────────────────────

def test_full_flow():
    print("\n=== Test 4: Full flow (requires Ollama) ===")

    hm = HistoryManager(
        system_prompt="You are a concise assistant. Answer in one sentence.",
        max_tokens=1800
    )

    exchanges = [
        ("What is 2+2?", ),
        ("What is the capital of France?", ),
        ("Name one planet in our solar system.", ),
    ]

    from dotenv import load_dotenv
    from litellm import completion
    load_dotenv()
    llm_model = os.getenv("LLM_MODEL", "ollama/qwen2.5:0.5b")

    for i, (question,) in enumerate(exchanges):
        hm.add_user(question)
        messages = hm.get_messages()
        try:
            resp = completion(
                model=llm_model,
                messages=messages,
                max_tokens=100,
                temperature=0.1,
            )
            answer = resp.choices[0].message.content.strip()
            hm.add_assistant(answer)
            usage = hm.token_usage()
            print(f"  [{i+1}] Q: {question}")
            print(f"       A: {answer[:80]}")
            print(f"       Tokens: {usage['used']}/{usage['budget']}, turns: {usage['turns']}")
        except Exception as e:
            print(f"  [{i+1}] LLM ERROR: {e}")
            hm.add_assistant("[error]")

    final = hm.token_usage()
    print(f"\n  Final: {final['used']} tokens, {final['turns']} turns, summary: {final['has_summary']}")
    assert final["turns"] == 3
    print("  [OK]")


# ── Test 5: reset() ───────────────────────────────────────────────────────────

def test_reset():
    print("\n=== Test 5: reset() ===")

    hm = HistoryManager(system_prompt="System.", max_tokens=2000)
    hm.add_user("Hello")
    hm.add_assistant("Hi")
    hm._summary = "Previous summary"

    assert hm.token_usage()["turns"] == 1
    assert hm.token_usage()["has_summary"] == True

    hm.reset()

    assert hm.token_usage()["turns"] == 0
    assert hm.token_usage()["has_summary"] == False
    assert hm._history == []
    assert hm._summary == ""
    print("  reset() clears history and summary [OK]")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_count_tokens()
    test_build_messages()
    test_reset()
    test_compression()
    test_full_flow()
    print("\n=== All tests passed ===")
