"""
Smoke tests – verify the service starts and the critical endpoints behave
correctly without a live Qdrant instance or OpenAI API key.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=True)


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert isinstance(body["uptime"], float)


# ── Metrics ────────────────────────────────────────────────────────────────────

def test_metrics_endpoint_returns_prometheus_text():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    # Prometheus text format always starts with a # HELP or # TYPE comment
    assert "http_requests_total" in resp.text
    assert "rag_retrieval_latency_ms" in resp.text


# ── Search – works with local embeddings (no OPENAI_API_KEY needed) ────────────

def test_search_returns_results_with_local_embeddings():
    # Qdrant mock returns empty list; expect 200 with total=0.
    resp = client.post(
        "/search",
        json={"query": "What is the refund policy?", "tenant_id": "acme"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["results"] == []


# ── Chat – 503 because LLM still requires OPENAI_API_KEY ───────────────────────

def test_chat_returns_503_without_llm_api_key():
    resp = client.post(
        "/chat",
        json={"query": "Summarise our onboarding guide.", "tenant_id": "acme"},
    )
    assert resp.status_code == 503
    assert "OPENAI_API_KEY" in resp.json()["detail"]


# ── Ingest – succeeds with local embeddings (no OPENAI_API_KEY needed) ─────────

def test_ingest_succeeds_with_local_embeddings():
    # embed_texts and Qdrant upsert are both mocked; expect a successful response.
    resp = client.post(
        "/ingest",
        data={"tenant_id": "acme"},
        files={"files": ("hello.txt", b"Hello world. " * 30, "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "doc_ids" in body
    assert len(body["doc_ids"]) == 1
    assert body["total_chunks"] >= 1


# ── Ingest – validation: missing tenant_id ─────────────────────────────────────

def test_ingest_requires_tenant_id():
    resp = client.post(
        "/ingest",
        files={"files": ("hello.txt", b"Hello world", "text/plain")},
    )
    assert resp.status_code == 422  # FastAPI validation error


# ── Feedback ───────────────────────────────────────────────────────────────────

def test_feedback_accepts_valid_payload(tmp_path, monkeypatch):
    from app.core import config
    monkeypatch.setattr(
        config.settings, "FEEDBACK_LOG_PATH", str(tmp_path / "feedback.jsonl")
    )

    resp = client.post(
        "/feedback",
        json={
            "tenant_id": "acme",
            "query": "How do I reset my password?",
            "rating": 4,
            "comment": "Good answer",
        },
    )
    assert resp.status_code == 204


def test_feedback_rejects_out_of_range_rating():
    resp = client.post(
        "/feedback",
        json={
            "tenant_id": "acme",
            "query": "test",
            "rating": 6,  # invalid – max is 5
        },
    )
    assert resp.status_code == 422


# ── Chunking unit test (no network) ───────────────────────────────────────────

def test_chunk_text_basic():
    from app.rag.chunking import chunk_text

    doc = "This is sentence one. " * 300  # ~1800 tokens
    chunks = chunk_text(doc, doc_id="test_doc", chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert c.doc_id == "test_doc"
        assert c.content.strip()
        assert len(c.chunk_hash) == 16


def test_chunk_text_markdown():
    from app.rag.chunking import chunk_text

    md = """# Introduction
This is the intro section.

## Setup
Install the package with pip.

## Usage
Call the API endpoint.
"""
    chunks = chunk_text(md, doc_id="md_doc", chunk_size=50, chunk_overlap=5)
    # Each chunk should carry the section header
    headers_found = sum(1 for c in chunks if "#" in c.content)
    assert headers_found > 0


def test_chunk_text_empty_returns_empty():
    from app.rag.chunking import chunk_text

    assert chunk_text("", doc_id="empty") == []
    assert chunk_text("   ", doc_id="whitespace") == []


# ── Eval metrics unit tests (no network) ──────────────────────────────────────

def test_recall_at_k():
    from app.eval.metrics import recall_at_k

    relevant = {"a", "b", "c"}
    assert recall_at_k(relevant, ["a", "b", "c"], k=3) == pytest.approx(1.0)
    assert recall_at_k(relevant, ["a", "x", "y"], k=3) == pytest.approx(1 / 3)
    assert recall_at_k(relevant, ["x", "y", "z"], k=3) == pytest.approx(0.0)
    assert recall_at_k(set(), ["a", "b"], k=2) == pytest.approx(0.0)


def test_reciprocal_rank():
    from app.eval.metrics import reciprocal_rank

    assert reciprocal_rank({"a"}, ["x", "a", "y"]) == pytest.approx(0.5)
    assert reciprocal_rank({"a"}, ["a", "b"]) == pytest.approx(1.0)
    assert reciprocal_rank({"a"}, ["x", "y"]) == pytest.approx(0.0)


def test_mean_reciprocal_rank():
    from app.eval.metrics import mean_reciprocal_rank

    pairs = [
        ({"a"}, ["a", "b"]),   # RR = 1.0
        ({"b"}, ["a", "b"]),   # RR = 0.5
    ]
    assert mean_reciprocal_rank(pairs) == pytest.approx(0.75)
    assert mean_reciprocal_rank([]) == pytest.approx(0.0)
