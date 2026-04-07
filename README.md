# ReAct RAG Agent

A local ReAct RAG agent powered by Ollama, LangGraph, and ChromaDB

## Prerequisites

- Ollama installed and running
- Python 3.12
- uv
- Node.js 20+ (for React frontend)

## Quick Start

```bash
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
uv sync
```

Drop documents into `data/raw/`, then run:

```bash
uv run python -m react_rag_agent.ingest
uv run python -m react_rag_agent.main
```

## Web App (React + API)

Run backend API:

```bash
uv run react-rag-api
```

Run backend + frontend together:

```bash
uv run react-rag-dev
```

If `npm` is not installed on the machine, this command will fail fast with guidance.
In that case, install Node.js/npm or run backend-only:

```bash
uv run react-rag-api
```

If your Node version is old (for example Node 12), upgrade to Node 20+ first.
Vite 8 will not run on Node 12.

If `vite: not found` appears, install frontend dependencies first:

```bash
cd web
npm install
cd ..
uv run react-rag-dev
```

Run frontend app:

```bash
cd web
npm install
npm run dev
```

Frontend expects backend at `http://127.0.0.1:8000` by default.

## Configuration

All runtime settings are controlled through `.env`.

Additional API settings:

- `API_HOST=127.0.0.1`
- `API_PORT=8000`
- `API_WORKERS=1`
- `API_LOG_LEVEL=info`
- `CORS_ORIGINS=http://localhost:5173`
- `MAX_UPLOAD_MB=25`
- `SESSION_TTL_MINUTES=240`
- `SESSION_MAX_MESSAGES=40`
- `SESSION_MAX_COUNT=200`

## Session Memory

The API now supports session memory for conversational continuity:

- `POST /sessions` creates a new session
- `GET /sessions/{session_id}` returns stored messages
- `DELETE /sessions/{session_id}` clears a session
- `POST /chat` and `POST /chat/stream` require `session_id`

Session memory is in-memory on the backend (MVP), with TTL and bounded message count.
If the API process restarts, sessions are reset.

## Orchestration Layer

The backend includes a smart orchestration layer that routes each request to the best path:

- `calculator` route for math expressions
- `direct` route for simple conversational turns
- `agent` route for retrieval/reasoning with tools

Routing is decided by a dedicated orchestrator model (`ORCHESTRATOR_MODEL`) with a heuristic fallback.

Orchestrator settings:

- `ORCHESTRATOR_ENABLED=true`
- `ORCHESTRATOR_MODEL=glm-4.7-flash`
- `ORCHESTRATOR_TEMPERATURE=0.0`

This reduces latency and unnecessary tool loops while preserving full RAG behavior for knowledge-heavy questions.

## Production Readiness Notes

- Use dedicated Ollama models for reasoning and orchestration on H100.
- Tune context and generation limits:
  - `REASONING_NUM_CTX`, `REASONING_NUM_PREDICT`, `REASONING_KEEP_ALIVE`
  - `ORCHESTRATOR_NUM_CTX`, `ORCHESTRATOR_NUM_PREDICT`, `ORCHESTRATOR_KEEP_ALIVE`
- Built-in KV caches are enabled for production efficiency:
  - retrieval cache (`RETRIEVAL_CACHE_TTL_SECONDS`)
  - orchestrator decision cache (`ORCHESTRATOR_CACHE_TTL_SECONDS`)
  - response cache (`RESPONSE_CACHE_TTL_SECONDS`)
  - global cap via `KV_CACHE_MAX_ENTRIES`
- Streaming endpoint emits route and telemetry events for observability.

### Recommended H100 Baseline

```env
REASONING_NUM_CTX=8192
REASONING_NUM_PREDICT=768
REASONING_KEEP_ALIVE=30m
ORCHESTRATOR_NUM_CTX=2048
ORCHESTRATOR_NUM_PREDICT=128
ORCHESTRATOR_KEEP_ALIVE=30m
KV_CACHE_MAX_ENTRIES=2000
RETRIEVAL_CACHE_TTL_SECONDS=120
ORCHESTRATOR_CACHE_TTL_SECONDS=600
RESPONSE_CACHE_TTL_SECONDS=300
```
