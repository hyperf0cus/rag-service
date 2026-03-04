"""High-level retrieval: embed query → search Qdrant → return typed results."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import rag_retrieval_latency_ms, rag_retrieval_top_score
from app.rag import qdrant_store
from app.rag.embeddings import embed_query

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    content: str
    score: float
    metadata: dict


async def retrieve(
    query: str,
    tenant_id: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> list[RetrievedChunk]:
    """
    Embed *query* and return the top-k most similar chunks for *tenant_id*.

    When *score_threshold* is None the global default
    (``settings.SIMILARITY_THRESHOLD``) is used.  Pass ``0.0`` explicitly to
    disable threshold filtering entirely (useful in the eval script).
    """
    if top_k is None:
        top_k = settings.RETRIEVAL_TOP_K
    if score_threshold is None:
        score_threshold = settings.SIMILARITY_THRESHOLD

    t0 = time.monotonic()
    query_vector = await embed_query(query)
    scored_points = await qdrant_store.search(
        query_vector=query_vector,
        tenant_id=tenant_id,
        top_k=top_k,
        score_threshold=score_threshold if score_threshold > 0 else None,
    )
    latency_ms = (time.monotonic() - t0) * 1000

    rag_retrieval_latency_ms.observe(latency_ms)
    if scored_points:
        rag_retrieval_top_score.observe(scored_points[0].score)

    logger.info(
        "Retrieval done",
        extra={
            "tenant_id": tenant_id,
            "n_results": len(scored_points),
            "top_score": scored_points[0].score if scored_points else None,
            "latency_ms": round(latency_ms, 1),
        },
    )

    _skip = {"chunk_id", "doc_id", "content", "chunk_hash", "tenant_id"}
    results: list[RetrievedChunk] = []
    for sp in scored_points:
        payload = sp.payload or {}
        results.append(
            RetrievedChunk(
                chunk_id=payload.get("chunk_id", ""),
                doc_id=payload.get("doc_id", ""),
                content=payload.get("content", ""),
                score=sp.score,
                metadata={k: v for k, v in payload.items() if k not in _skip},
            )
        )

    return results
