"""
Ingest endpoints.

Two variants:
  POST /ingest          – multipart/form-data with one or more file uploads.
  POST /ingest/paths    – JSON body listing server-side absolute file paths.

Both produce the same IngestResponse.  Re-ingesting the same content is
idempotent at two levels:
  1. doc_id is derived from SHA-1(file_bytes + tenant_id) so the same file
     always produces the same doc_id regardless of filename or upload path.
  2. Duplicate chunks (matched by chunk_hash + tenant_id) are silently
     skipped and reflected in ``chunks_skipped``.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger, tenant_id_var
from app.core.metrics import rag_ingest_chunks_total, rag_ingest_docs_total
from app.rag import qdrant_store
from app.rag.chunking import chunk_text
from app.rag.embeddings import embed_texts

router = APIRouter()
logger = get_logger(__name__)


# ── Schemas ────────────────────────────────────────────────────────────────────

class IngestPathsRequest(BaseModel):
    tenant_id: str
    paths: list[str]
    metadata: Optional[dict] = None


class IngestResponse(BaseModel):
    doc_ids: list[str]
    chunks_ingested: int
    chunks_skipped: int
    total_chunks: int


# ── Shared helper ──────────────────────────────────────────────────────────────

async def _ingest_text(
    text: str,
    doc_id: str,
    tenant_id: str,
    metadata: Optional[dict] = None,
) -> tuple[int, int]:
    """
    Chunk → embed → upsert one document.  Returns (inserted, skipped).
    """
    chunks = chunk_text(
        text=text,
        doc_id=doc_id,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        metadata=metadata or {},
    )
    if not chunks:
        return 0, 0

    embeddings = await embed_texts([c.content for c in chunks])

    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "content": c.content,
            "chunk_hash": c.chunk_hash,
            "embedding": emb,
            "metadata": c.metadata,
            "tenant_id": tenant_id,
        }
        for c, emb in zip(chunks, embeddings)
    ]

    inserted, skipped = await qdrant_store.upsert_chunks(chunk_dicts)
    return inserted, skipped


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest_files(
    tenant_id: str = Form(...),
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(default=None),
) -> IngestResponse:
    """
    Ingest documents via multipart/form-data file upload.

    ``metadata`` is an optional JSON-string that is merged into every chunk's
    payload (e.g. ``{"source": "confluence", "team": "engineering"}``).
    """
    tenant_id_var.set(tenant_id)

    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    meta: dict = {}
    if metadata:
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="metadata must be valid JSON.")

    doc_ids: list[str] = []
    total_inserted = 0
    total_skipped = 0

    for upload in files:
        raw = await upload.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail=f"File '{upload.filename}' is not valid UTF-8 text.",
            )

        doc_id = f"{tenant_id}_{hashlib.sha1(raw).hexdigest()[:12]}"
        file_meta = {**meta, "source_filename": upload.filename or "unknown"}

        inserted, skipped = await _ingest_text(text, doc_id, tenant_id, file_meta)
        doc_ids.append(doc_id)
        total_inserted += inserted
        total_skipped += skipped

        rag_ingest_chunks_total.labels(tenant_id=tenant_id).inc(inserted)
        rag_ingest_docs_total.labels(tenant_id=tenant_id).inc(1)

        logger.info(
            "File ingested",
            extra={
                "doc_id": doc_id,
                "source_file": upload.filename,
                "chunks_inserted": inserted,
                "chunks_skipped": skipped,
            },
        )

    return IngestResponse(
        doc_ids=doc_ids,
        chunks_ingested=total_inserted,
        chunks_skipped=total_skipped,
        total_chunks=total_inserted + total_skipped,
    )


@router.post("/ingest/paths", response_model=IngestResponse)
async def ingest_paths(request: IngestPathsRequest) -> IngestResponse:
    """
    Ingest documents by providing server-side absolute file paths.

    Useful for bulk ingest from mounted volumes or during development.
    """
    tenant_id_var.set(request.tenant_id)

    doc_ids: list[str] = []
    total_inserted = 0
    total_skipped = 0

    for path in request.paths:
        if not os.path.isfile(path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found or is not a regular file: {path}",
            )
        with open(path, "rb") as fh:
            raw = fh.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail=f"File '{path}' is not valid UTF-8 text.",
            )

        doc_id = f"{request.tenant_id}_{hashlib.sha1(raw).hexdigest()[:12]}"
        file_meta = {**(request.metadata or {}), "source_path": path}

        inserted, skipped = await _ingest_text(
            text, doc_id, request.tenant_id, file_meta
        )
        doc_ids.append(doc_id)
        total_inserted += inserted
        total_skipped += skipped

        rag_ingest_chunks_total.labels(tenant_id=request.tenant_id).inc(inserted)
        rag_ingest_docs_total.labels(tenant_id=request.tenant_id).inc(1)

        logger.info(
            "Path ingested",
            extra={
                "doc_id": doc_id,
                "path": path,
                "chunks_inserted": inserted,
                "chunks_skipped": skipped,
            },
        )

    return IngestResponse(
        doc_ids=doc_ids,
        chunks_ingested=total_inserted,
        chunks_skipped=total_skipped,
        total_chunks=total_inserted + total_skipped,
    )
