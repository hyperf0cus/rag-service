"""
Pytest configuration and shared fixtures.

Key concerns addressed:
  1. No live Qdrant – the singleton client is replaced with a MagicMock.
  2. No embedding model / API – the provider singleton (_provider) is replaced
     with a MagicMock whose _embed_batch returns fixed vectors.  This intercepts
     all callers of embed_texts regardless of how they imported it (direct import
     or module-level call), because embed_texts calls get_provider() at runtime.
  3. No live LLM – OPENAI_API_KEY is cleared and LLM_PROVIDER is forced to
     "openai" so get_llm_provider() always returns None (→ 503).  This keeps
     /chat tests stable regardless of what the local .env specifies (e.g.
     LLM_PROVIDER=ollama).  Tests that need a working LLM should override the
     clear_openai_key fixture.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

_FAKE_DIM = 384  # matches all-MiniLM-L6-v2; used consistently across stubs


@pytest.fixture(autouse=True)
def mock_qdrant(monkeypatch):
    """Replace the Qdrant client singleton with a no-op mock."""
    from app.rag import qdrant_store

    mock_client = MagicMock()
    mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    mock_client.create_collection = AsyncMock()
    mock_client.create_payload_index = AsyncMock()
    mock_client.query_points = AsyncMock(return_value=MagicMock(points=[]))
    mock_client.upsert = AsyncMock()
    mock_client.scroll = AsyncMock(return_value=([], None))

    monkeypatch.setattr(qdrant_store, "_client", mock_client)
    return mock_client


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    """
    Stub out the embeddings provider so no model or API is invoked.

    Strategy: replace the module-level _provider singleton with a MagicMock
    whose _embed_batch is an AsyncMock returning fixed 384-dim vectors.
    embed_texts() calls get_provider() at runtime, so ALL callers (including
    routes that imported embed_texts directly at load time) transparently use
    the mock without any additional patching.
    """
    import app.rag.embeddings as emb_module

    mock_provider = MagicMock()
    mock_provider.dimension = _FAKE_DIM

    async def _fake_batch(texts: list[str]) -> list[list[float]]:
        return [[0.01] * _FAKE_DIM for _ in texts]

    mock_provider._embed_batch = _fake_batch

    monkeypatch.setattr(emb_module, "_provider", mock_provider)
    monkeypatch.setattr(emb_module, "get_embedding_dimension", lambda: _FAKE_DIM)


@pytest.fixture(autouse=True)
def clear_openai_key(monkeypatch):
    """
    Isolate tests from live LLM APIs.

    - Clears OPENAI_API_KEY so tests never hit the live OpenAI API.
    - Pins LLM_PROVIDER=openai so get_llm_provider() returns None (→ 503)
      regardless of what the local .env sets (e.g. LLM_PROVIDER=ollama).
    Tests that exercise the LLM path should override or extend this fixture.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from app.core import config
    monkeypatch.setattr(config.settings, "OPENAI_API_KEY", None)
    monkeypatch.setattr(config.settings, "LLM_PROVIDER", "openai")
