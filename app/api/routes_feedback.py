"""
POST /feedback – optional endpoint to capture user ratings.

Feedback is appended as newline-delimited JSON to FEEDBACK_LOG_PATH.
This is intentionally simple; in production, swap the file sink for a
database write or a message queue publish.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class FeedbackRequest(BaseModel):
    tenant_id: str
    query: str
    rating: int = Field(..., ge=1, le=5, description="Rating 1 (poor) – 5 (excellent)")
    conversation_id: Optional[str] = None
    comment: Optional[str] = None
    chunk_ids: Optional[list[str]] = None


@router.post("/feedback", status_code=204)
async def submit_feedback(request: FeedbackRequest) -> None:
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tenant_id": request.tenant_id,
        "conversation_id": request.conversation_id,
        "query": request.query,
        "rating": request.rating,
        "comment": request.comment,
        "chunk_ids": request.chunk_ids,
    }

    log_path = settings.FEEDBACK_LOG_PATH
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError as exc:
        # Non-fatal: log and continue; don't fail the client request.
        logger.error("Failed to persist feedback", extra={"error": str(exc)})
