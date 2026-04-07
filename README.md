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

## Configuration

All runtime settings are controlled through `.env`.
