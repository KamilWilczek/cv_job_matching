"""
CV/Job Matching — interactive chat.

Combines all project modules:
  Phase 2: embeddings (nomic-embed-text)
  Phase 3: database (pgvector)
  Phase 4: hybrid search (RRF)
  Phase 5: LLM analysis (qwen2.5:0.5b + guardrails)
  Phase 6: context management (history summary)

Run:
  cd C:\\dev\\Projects\\Learning\\LLM\\cv_job_matching
  .venv\\Scripts\\python main.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

from dotenv import load_dotenv
from litellm import completion

from database      import init_schema, stats
from ingestion     import ingest_cvs, ingest_job_offers
from embeddings    import get_embedding
from matching      import hybrid_search_cvs
from llm_analysis  import analyze_top_candidates, format_analysis_report
from context_manager import HistoryManager, count_tokens

load_dotenv()

ROOT     = Path(__file__).parent
CVS_DIR  = ROOT / "data" / "sample_cvs"
JOBS_DIR = ROOT / "data" / "sample_jobs"
LLM_MODEL = os.getenv("LLM_MODEL", "ollama/qwen2.5:0.5b")

# Keywords extracted from job offers — in production these would be extracted by the LLM
KEYWORDS = {
    "python_backend_senior": "Python FastAPI PostgreSQL Docker asyncio",
    "devops_engineer":       "Kubernetes Terraform AWS Docker CI/CD ArgoCD",
}

SYSTEM_PROMPT = """You are an AI assistant for a CV/job matching system.
You help recruiters find the best candidates for job positions.
You have access to a database of CVs and job offers.
Keep answers concise. When showing candidates, always include their match score."""


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap() -> bool:
    """Check the database and load data if empty."""
    try:
        init_schema()
        s = stats()
        if s["cvs"] == 0:
            print("Database empty — loading data...")
            ingest_cvs(CVS_DIR)
            ingest_job_offers(JOBS_DIR)
            s = stats()
        print(f"Database: {s['cvs']} CVs, {s['job_offers']} job offers\n")
        return True
    except Exception as e:
        print(f"DATABASE ERROR: {e}")
        print("Make sure Docker is running: docker compose up -d")
        return False


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list_jobs() -> str:
    files = sorted(JOBS_DIR.glob("*.txt"))
    lines = ["Available job offers:"]
    for f in files:
        lines.append(f"  - {f.stem}")
    return "\n".join(lines)


def cmd_list_cvs() -> str:
    files = sorted(CVS_DIR.glob("*.txt"))
    lines = ["Candidates in database:"]
    for f in files:
        first_line = f.read_text(encoding="utf-8").splitlines()[0]
        lines.append(f"  - {f.stem}  ({first_line})")
    return "\n".join(lines)


def cmd_match(job_name: str) -> str:
    """Hybrid search + LLM analysis for the selected job offer."""
    job_path = JOBS_DIR / f"{job_name}.txt"
    if not job_path.exists():
        avail = [f.stem for f in JOBS_DIR.glob("*.txt")]
        return f"Job offer '{job_name}' not found. Available: {avail}"

    job_text = job_path.read_text(encoding="utf-8")
    job_vec  = get_embedding(job_text, task="search_document")
    keywords = KEYWORDS.get(job_name, job_name.replace("_", " "))

    print(f"  Searching candidates for: {job_name}...")
    top3 = hybrid_search_cvs(job_vec, keywords, top_k=3)

    print(f"  Analysing top-3 via LLM ({LLM_MODEL})...")
    enriched = analyze_top_candidates(job_text, top3, CVS_DIR)

    return format_analysis_report(enriched)


def cmd_tokens(hm: HistoryManager) -> str:
    """Show current token usage."""
    info = hm.token_usage()
    lines = [
        "Token usage:",
        f"  Used:      {info['used']}",
        f"  Budget:    {info['budget']}",
        f"  Remaining: {info['remaining']}",
        f"  Turns:     {info['turns']}",
        f"  Summary:   {'yes' if info['has_summary'] else 'no'}",
    ]
    return "\n".join(lines)


def cmd_help() -> str:
    return """Available commands:
  /jobs           - list job offers
  /cvs            - list candidates
  /match <offer>  - find and analyse candidates (e.g. /match python_backend_senior)
  /tokens         - show token usage and history state
  /reset          - clear conversation history
  /quit           - exit

You can also ask questions in Polish or English — the LLM will respond."""


# ── LLM query dispatch ────────────────────────────────────────────────────────

def llm_respond(user_msg: str, hm: HistoryManager) -> str:
    """Send a message to the LLM with the current history context."""
    hm.add_user(user_msg)
    messages = hm.get_messages()

    try:
        resp = completion(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0.3,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"[LLM error: {e}]"

    hm.add_assistant(answer)
    return answer


# ── Main chat loop ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CV/Job Matching Chat")
    print(f"Model: {LLM_MODEL}")
    print("=" * 60)
    print()

    if not bootstrap():
        return

    hm = HistoryManager(system_prompt=SYSTEM_PROMPT, max_tokens=1800)

    print(cmd_help())
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/help":
            print(cmd_help())
            continue

        if user_input.lower() == "/jobs":
            print(cmd_list_jobs())
            continue

        if user_input.lower() == "/cvs":
            print(cmd_list_cvs())
            continue

        if user_input.lower().startswith("/match "):
            job_name = user_input[7:].strip()
            print(cmd_match(job_name))
            continue

        if user_input.lower() == "/tokens":
            print(cmd_tokens(hm))
            continue

        if user_input.lower() == "/reset":
            hm.reset()
            print("History cleared.")
            continue

        tokens_before = hm.token_usage()["used"]
        answer = llm_respond(user_input, hm)
        tokens_after = hm.token_usage()["used"]

        print(f"\nAssistant: {answer}")
        print(f"  [tokens: {tokens_after} used, +{tokens_after - tokens_before} this turn]\n")


if __name__ == "__main__":
    main()
