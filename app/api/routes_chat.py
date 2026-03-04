"""
POST /chat – SSE streaming RAG chat endpoint.

Event stream protocol
---------------------
Each event is a ``data: <json>\\n\\n`` frame.

  data: {"type": "token",     "content": "<incremental token>"}
  …
  data: {"type": "citations", "citations": [
            {"rank":1, "doc_id":"…", "chunk_id":"…", "score":0.87, "snippet":"…"},
            …
         ]}
  data: [DONE]

The ``citations`` event is always the last data event before ``[DONE]``.
It contains an empty list when the LLM returned the no-answer sentinel phrase
or when no relevant context was found.

Error handling
--------------
- Missing OPENAI_API_KEY → HTTP 503 (before any streaming begins).
- Retrieval or LLM errors occurring mid-stream are forwarded as a
  ``{"type": "error", "message": "…"}`` event followed by ``[DONE]``.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.logging import get_logger, request_id_var, tenant_id_var
from app.rag.llm_client import get_llm_provider
from app.rag.retrieval import retrieve
from app.rag.streaming import stream_rag_response

router = APIRouter()
logger = get_logger(__name__)


# ── Schemas ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    tenant_id: str
    top_k: Optional[int] = None
    # Pass 0.0 to disable minimum-score filtering
    score_threshold: Optional[float] = None
    conversation_id: Optional[str] = None


# ── Route ──────────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    req_id = str(uuid.uuid4())
    request_id_var.set(req_id)
    tenant_id_var.set(request.tenant_id)

    llm = get_llm_provider()
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM provider is not configured. "
                "Set OPENAI_API_KEY (for LLM_PROVIDER=openai) or "
                "set LLM_PROVIDER=ollama for a fully local setup."
            ),
        )

    # Retrieve context *before* opening the stream so any retrieval errors
    # produce a normal JSON error response rather than a broken SSE stream.
    chunks = await retrieve(
        query=request.query,
        tenant_id=request.tenant_id,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )

    logger.info(
        "Chat started",
        extra={
            "conversation_id": request.conversation_id,
            "n_chunks": len(chunks),
            "top_score": chunks[0].score if chunks else None,
        },
    )

    return StreamingResponse(
        stream_rag_response(
            query=request.query,
            chunks=chunks,
            llm_provider=llm,
            tenant_id=request.tenant_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx proxy buffering
            "X-Request-ID": req_id,
        },
    )
