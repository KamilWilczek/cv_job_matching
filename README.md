# cv_job_matching

A CV/job matching system built to learn how LLM pipelines actually work end-to-end. It takes a job offer, finds the best candidates from a database using hybrid search, and runs each one through an LLM for a structured match analysis.

The stack is fully local — PostgreSQL with pgvector for vector storage, Ollama for embeddings and the LLM, no API keys required.

## What it does

You run `main.py` and get an interactive chat where you can ask questions or use slash commands:

```
/jobs                         list available job offers
/cvs                          list candidates in the database
/match python_backend_senior  hybrid search + LLM analysis of top 3
/tokens                       show how much context window is used
/reset                        clear conversation history
```

The `/match` command runs the full pipeline: embeds the job offer with `nomic-embed-text`, finds candidates using both vector similarity and keyword search (fused via RRF), then asks the LLM to score each one and explain the match.

## How it's built

**Search** uses two signals combined with Reciprocal Rank Fusion:
- Vector search (cosine similarity on 768-dim embeddings) — catches semantic similarity
- Full-text search (`ts_rank_cd` on PostgreSQL tsvector) — catches exact keyword matches

Pure vector search alone ranks a Frontend developer above a Python Senior for a Python backend role because "REST API" and "TypeScript" are semantically close to Python backend. Adding keyword matching fixes this — the Python Senior wins because his CV contains `FastAPI`, `asyncio`, and `PostgreSQL`.

**LLM analysis** uses a few-shot RICEFACT prompt to get structured JSON output: a match score, key matches, gaps, and a recommendation. Small local models (qwen2.5:0.5b) tend to anchor scores around the few-shot example and don't distinguish candidates well — guardrails fix the format but can't fix the reasoning. Swap `LLM_MODEL` in `.env` to `claude/claude-haiku-4-5` or `openai/gpt-4o-mini` for actual quality.

**Context management** compresses old conversation turns into a summary instead of truncating them, so the chat session can run longer without losing earlier context.

## Setup

You need Docker and Ollama installed.

```bash
# pull the models
ollama pull nomic-embed-text
ollama pull qwen2.5:0.5b

# start the database
docker compose up -d

# install dependencies
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install ...  # Linux/Mac

# copy and edit env if needed
cp .env.example .env

# run
.venv\Scripts\python main.py
```

On first run it ingests the sample data automatically.

## Project structure

```
src/
  embeddings.py       get_embedding(), cosine_similarity()
  database.py         PostgreSQL schema, pgvector queries
  ingestion.py        .txt files -> embeddings -> database
  matching.py         vector / fulltext / hybrid RRF search
  guardrails.py       input validation, JSON output parsing and repair
  llm_analysis.py     RICEFACT prompt, few-shot, match scoring
  context_manager.py  conversation history with summary compression
main.py               interactive chat CLI
data/
  sample_cvs/         5 example CVs
  sample_jobs/        2 example job offers
test_*.py             one test file per module
```

## Notes

The sample data is in Polish (it's a Polish learning project). The code and comments are in English. To add your own CVs or job offers, drop `.txt` files into the relevant `data/` directory and restart — bootstrap picks them up automatically.

The database runs on port 5434 instead of the default 5432 to avoid conflicts with any existing PostgreSQL installation.
