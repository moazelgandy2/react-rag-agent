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

The backend includes a lightweight orchestration layer that routes each request to the best path:

- `calculator` route for math expressions
- `direct` route for simple conversational turns
- `agent` route for retrieval/reasoning with tools

This reduces latency and unnecessary tool loops while preserving full RAG behavior for knowledge-heavy questions.
