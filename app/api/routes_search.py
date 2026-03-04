"""
POST /search – debug/inspection endpoint for retrieval only (no LLM).

Use this to validate that documents are ingested correctly and that your
query is retrieving the expected chunks before wiring up /chat.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger, tenant_id_var
from app.rag.retrieval import RetrievedChunk, retrieve

router = APIRouter()
logger = get_logger(__name__)


# ── Schemas ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    tenant_id: str
    top_k: int = settings.RETRIEVAL_TOP_K
    # Pass 0.0 to disable score filtering
    score_threshold: Optional[float] = None


class SearchResultItem(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    tenant_id: str
    results: list[SearchResultItem]
    total: int


# ── Route ──────────────────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    tenant_id_var.set(request.tenant_id)

    chunks: list[RetrievedChunk] = await retrieve(
        query=request.query,
        tenant_id=request.tenant_id,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )

    return SearchResponse(
        query=request.query,
        tenant_id=request.tenant_id,
        results=[
            SearchResultItem(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                content=c.content,
                score=c.score,
                metadata=c.metadata,
            )
            for c in chunks
        ],
        total=len(chunks),
    )
