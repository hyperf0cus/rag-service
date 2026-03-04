"""
FastAPI application entry-point.

Startup
-------
On startup the service attempts to connect to Qdrant and ensure the collection
exists.  A failure here is non-fatal (logged as a warning) so the service can
still start even when Qdrant is temporarily unavailable; Qdrant will be
re-initialised on the first ingest/search request.

Middleware
----------
Every request passes through ``observability_middleware`` which:
  - Assigns / propagates a ``X-Request-ID`` header.
  - Sets the ``request_id_var`` and ``tenant_id_var`` ContextVars so that all
    log lines emitted during the request are automatically annotated.
  - Records ``http_requests_total`` and ``http_request_duration_seconds``
    Prometheus metrics.
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    routes_chat,
    routes_feedback,
    routes_health,
    routes_ingest,
    routes_metrics,
    routes_search,
)
from app.core.config import settings
from app.core.logging import get_logger, request_id_var, setup_logging, tenant_id_var
from app.core.metrics import http_request_duration_seconds, http_requests_total
from app.rag import qdrant_store

setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Service starting",
        extra={"service": settings.SERVICE_NAME, "version": settings.SERVICE_VERSION},
    )
    try:
        await qdrant_store.ensure_collection()
    except Exception as exc:
        logger.warning(
            "Qdrant not reachable at startup – will retry on first request",
            extra={"error": str(exc)},
        )
    yield
    logger.info("Service stopped")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    description="Production-grade RAG service",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Observability middleware ───────────────────────────────────────────────────

@app.middleware("http")
async def observability_middleware(request: Request, call_next) -> Response:
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    tenant = request.headers.get("X-Tenant-ID", "")
    request_id_var.set(req_id)
    tenant_id_var.set(tenant)

    t0 = time.monotonic()
    response: Response = await call_next(request)
    duration = time.monotonic() - t0

    path = request.url.path
    method = request.method
    status = str(response.status_code)

    http_requests_total.labels(method=method, endpoint=path, status_code=status).inc()
    http_request_duration_seconds.labels(method=method, endpoint=path).observe(duration)

    response.headers["X-Request-ID"] = req_id

    logger.info(
        "HTTP",
        extra={
            "method": method,
            "path": path,
            "status": int(status),
            "duration_ms": round(duration * 1000, 1),
        },
    )
    return response


# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(routes_health.router, tags=["health"])
app.include_router(routes_ingest.router, tags=["ingest"])
app.include_router(routes_search.router, tags=["search"])
app.include_router(routes_chat.router, tags=["chat"])
app.include_router(routes_feedback.router, tags=["feedback"])
app.include_router(routes_metrics.router, tags=["observability"])
