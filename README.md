# ReAct RAG Agent

A local ReAct RAG agent powered by Ollama, LangGraph, and ChromaDB

## Prerequisites

- Ollama installed and running
- Python 3.12
- uv

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
