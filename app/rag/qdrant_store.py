"""
Qdrant vector store wrapper.

Design decisions
----------------
* Single collection for all tenants; tenant isolation is enforced via a
  mandatory ``tenant_id`` payload filter on every read and at upsert time.
* Deduplication: before inserting a chunk we scroll for an existing point
  whose ``chunk_hash`` + ``tenant_id`` match.  Duplicate content is silently
  skipped so re-ingesting the same document is idempotent.
* Point IDs are random UUIDs – Qdrant requires UUID or uint64 IDs, and UUIDs
  avoid collisions across tenants without needing a distributed counter.
"""
from __future__ import annotations

import uuid
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------


def _extract_points(response) -> list[ScoredPoint]:
    """Extract ScoredPoint list from a query_points() response.

    qdrant-client >= 1.10 returns a QueryResponse whose ``.points`` attribute
    holds the list.  Older versions or future API changes may wrap them under
    ``.result.points`` or return a plain list.  This helper handles all shapes
    so callers are insulated from version differences.
    """
    if isinstance(response, list):
        return response
    if hasattr(response, "points"):
        return response.points
    if hasattr(response, "result"):
        result = response.result
        if hasattr(result, "points"):
            return result.points
        if isinstance(result, list):
            return result
    return []

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Module-level singleton; created lazily on first call to get_client().
_client: Optional[AsyncQdrantClient] = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )
    return _client


async def ensure_collection() -> None:
    """Create the collection (and payload indexes) if it does not yet exist."""
    # Import here to avoid a circular import at module level; embeddings imports
    # from core.config and core.metrics, not from qdrant_store.
    from app.rag.embeddings import get_embedding_dimension

    client = get_client()
    existing = await client.get_collections()
    names = {c.name for c in existing.collections}

    if settings.QDRANT_COLLECTION_NAME in names:
        logger.info(
            "Qdrant collection exists",
            extra={"collection": settings.QDRANT_COLLECTION_NAME},
        )
        return

    vector_size = get_embedding_dimension()
    await client.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    # Index tenant_id for fast filtering; index chunk_hash for dedup lookups.
    for field_name in ("tenant_id", "chunk_hash"):
        await client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            field_name=field_name,
            field_schema="keyword",
        )

    logger.info(
        "Qdrant collection created",
        extra={"collection": settings.QDRANT_COLLECTION_NAME},
    )


async def _chunk_exists(chunk_hash: str, tenant_id: str) -> bool:
    client = get_client()
    records, _ = await client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="chunk_hash", match=MatchValue(value=chunk_hash)),
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(records) > 0


async def upsert_chunks(
    chunks: list[dict],
) -> tuple[int, int]:
    """
    Upsert a list of chunk dicts into Qdrant, skipping duplicates.

    Each dict must contain:
        chunk_id, doc_id, content, chunk_hash, embedding, metadata, tenant_id

    Returns:
        (inserted, skipped) counts.
    """
    client = get_client()
    points: list[PointStruct] = []
    skipped = 0

    for chunk in chunks:
        exists = await _chunk_exists(chunk["chunk_hash"], chunk["tenant_id"])
        if exists:
            skipped += 1
            continue

        payload = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "content": chunk["content"],
            "chunk_hash": chunk["chunk_hash"],
            "tenant_id": chunk["tenant_id"],
            **chunk.get("metadata", {}),
        }
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk["embedding"],
                payload=payload,
            )
        )

    if points:
        await client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points,
        )

    logger.info(
        "Upsert complete",
        extra={"inserted": len(points), "skipped": skipped},
    )
    return len(points), skipped


async def search(
    query_vector: list[float],
    tenant_id: str,
    top_k: int = 5,
    score_threshold: Optional[float] = None,
) -> list[ScoredPoint]:
    """
    Search for nearest chunks, filtered to *tenant_id*.

    score_threshold=None disables the threshold filter (returns top-k regardless
    of score).  Pass a float to apply a minimum-score cutoff.
    """
    client = get_client()
    response = await client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        ),
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )
    return _extract_points(response)
