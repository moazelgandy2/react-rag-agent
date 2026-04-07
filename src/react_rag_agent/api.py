import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from .agent import build_agent, invoke_agent_with_messages, stream_agent_with_messages
from .config import settings
from .ingest import run_ingestion
from .orchestrator import Route, decide_route, run_calculator_route, run_direct_reply
from .retrieval import get_vector_store, retrieve
from .session_store import session_store

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
    session_id: str = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = None


class SessionCreateResponse(BaseModel):
    session_id: str
    ttl_minutes: int
    max_messages: int


def _extract_final_answer(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return str(message.content)
    return ""


def _humanize_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^(Based on|According to|From|Using|Given)[^.]{0,180}\.\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(based on the (retrieved|provided|available) (documents|context))\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned or text


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "reasoning_model": settings.reasoning_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/sessions", response_model=SessionCreateResponse)
def create_session() -> SessionCreateResponse:
    session = session_store.create()
    return SessionCreateResponse(
        session_id=session.session_id,
        ttl_minutes=settings.session_ttl_minutes,
        max_messages=settings.session_max_messages,
    )


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    messages = session_store.list_messages(session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "messages": [{"role": role, "content": content} for role, content in messages],
        "message_count": len(messages),
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    deleted = session_store.clear(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ok", "deleted": True}


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "ReAct RAG Agent API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "frontend_dev": "http://localhost:5173",
        "note": "This is the backend API. Open the frontend URL for the GUI.",
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
    history = session_store.list_messages(request.session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found")

    message_input = [*history, ("user", request.message)]
    decision = decide_route(request.message)

    try:
        if decision.route == Route.CALCULATOR:
            answer = run_calculator_route(request.message)
        elif decision.route == Route.DIRECT:
            answer = run_direct_reply(request.message)
        else:
            result = invoke_agent_with_messages(agent, message_input)
            answer = _humanize_answer(_extract_final_answer(result))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    session_store.append_exchange(request.session_id, request.message, answer)

    return {
        "session_id": request.session_id,
        "answer": answer,
        "route": decision.route.value,
    }


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    def event_stream():
        history = session_store.list_messages(request.session_id)
        if history is None:
            payload = {"type": "error", "content": "Session not found"}
            yield f"data: {json.dumps(payload)}\n\n"
            return

        message_input = [*history, ("user", request.message)]
        decision = decide_route(request.message)
        final_answer = ""

        try:
            route_payload = {"type": "route", "route": decision.route.value}
            route_payload["source"] = decision.source
            route_payload["reason"] = decision.reason
            yield f"data: {json.dumps(route_payload)}\n\n"

            if decision.route == Route.CALCULATOR:
                final_answer = run_calculator_route(request.message)
                payload = {"type": "final_answer", "content": final_answer}
                yield f"data: {json.dumps(payload)}\n\n"
            elif decision.route == Route.DIRECT:
                final_answer = run_direct_reply(request.message)
                payload = {"type": "final_answer", "content": final_answer}
                yield f"data: {json.dumps(payload)}\n\n"
            else:
                for step in stream_agent_with_messages(agent, message_input):
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
                        final_answer = _humanize_answer(str(last_message.content))
                        payload = {"type": "final_answer", "content": final_answer}
                        yield f"data: {json.dumps(payload)}\n\n"

            session_store.append_exchange(request.session_id, request.message, final_answer)
            yield 'data: {"type": "done"}\n\n'
        except Exception as exc:  # noqa: BLE001
            payload = {"type": "error", "content": str(exc)}
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
