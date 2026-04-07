import json
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from .agent import build_agent, invoke_agent, stream_agent
from .config import settings
from .ingest import run_ingestion
from .retrieval import get_vector_store, retrieve

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}

app = FastAPI(title="ReAct RAG Agent API", version="0.1.0")

origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = build_agent()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = None


def _extract_final_answer(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return str(message.content)
    return ""


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "reasoning_model": settings.reasoning_model,
        "embedding_model": settings.embedding_model,
    }


@app.get("/documents")
def list_documents() -> dict[str, Any]:
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    files = [
        path.name
        for path in sorted(data_dir.iterdir())
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]
    return {"documents": files, "count": len(files)}


@app.post("/ingest/upload")
async def upload_documents(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    skipped: list[dict[str, str]] = []
    max_bytes = settings.max_upload_mb * 1024 * 1024

    for file in files:
        safe_name = Path(file.filename or "").name
        extension = Path(safe_name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            skipped.append({"file": safe_name, "reason": "unsupported file type"})
            continue

        payload = await file.read()
        if len(payload) > max_bytes:
            skipped.append({"file": safe_name, "reason": "file exceeds size limit"})
            continue

        target = data_dir / safe_name
        target.write_bytes(payload)
        saved.append(safe_name)

    return {"saved": saved, "skipped": skipped, "saved_count": len(saved)}


@app.post("/ingest/run")
def ingest_documents() -> dict[str, Any]:
    try:
        run_ingestion()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    return {"status": "ok", "message": "Ingestion completed"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    try:
        result = invoke_agent(agent, request.message)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    return {
        "answer": _extract_final_answer(result),
    }


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    def event_stream():
        try:
            for step in stream_agent(agent, request.message):
                messages = step.get("messages", [])
                if not messages:
                    continue

                last_message = messages[-1]
                if isinstance(last_message, AIMessage) and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        payload = {
                            "type": "tool_call",
                            "name": tool_call.get("name", "unknown"),
                            "args": tool_call.get("args", {}),
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                elif isinstance(last_message, ToolMessage):
                    payload = {
                        "type": "tool_result",
                        "content": str(last_message.content)[:500],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                elif isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    payload = {"type": "final_answer", "content": str(last_message.content)}
                    yield f"data: {json.dumps(payload)}\n\n"

            yield 'data: {"type": "done"}\n\n'
        except Exception as exc:  # noqa: BLE001
            payload = {"type": "error", "content": str(exc)}
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/vector/stats")
def vector_stats() -> dict[str, Any]:
    try:
        vector_store = get_vector_store()
        raw = vector_store.get(include=["metadatas"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not read vector store: {exc}") from exc

    ids = raw.get("ids", [])
    metadatas = raw.get("metadatas", [])

    source_counter = Counter()
    for metadata in metadatas:
        source = metadata.get("source", "unknown") if isinstance(metadata, dict) else "unknown"
        source_counter[str(source)] += 1

    top_sources = [
        {"source": source, "chunks": chunks} for source, chunks in source_counter.most_common(10)
    ]

    return {
        "collection_name": settings.collection_name,
        "persist_dir": settings.chroma_persist_dir,
        "total_chunks": len(ids),
        "source_count": len(source_counter),
        "top_sources": top_sources,
    }


@app.post("/vector/search")
def vector_search(request: SearchRequest) -> dict[str, Any]:
    try:
        results = retrieve(request.query, request.top_k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Vector search failed: {exc}") from exc
    return {"results": results, "count": len(results)}


def run() -> None:
    import uvicorn

    uvicorn.run(
        "react_rag_agent.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
