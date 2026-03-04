"""
Embedding providers with a shared in-process LRU cache.

Provider selection is controlled by EMBEDDINGS_PROVIDER (env / .env):

  local  (default)
    Uses sentence-transformers running on CPU.  No API key required.
    Model is downloaded from HuggingFace on first use (~90 MB for the default).
    Chosen model is configurable via EMBEDDINGS_LOCAL_MODEL.

  openai
    Uses the OpenAI Embeddings REST API.  Requires OPENAI_API_KEY.
    Model is configurable via OPENAI_EMBEDDING_MODEL.

The public surface is unchanged from the previous version:
  - embed_texts(texts)        → list[list[float]]
  - embed_query(text)         → list[float]
  - get_embedding_dimension() → int   (NEW – used by qdrant_store)
"""
from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import rag_cache_hit_total, rag_cache_miss_total

logger = get_logger(__name__)


# ── Shared LRU cache ───────────────────────────────────────────────────────────

class _LRUCache:
    def __init__(self, max_size: int) -> None:
        self._store: OrderedDict[str, list[float]] = OrderedDict()
        self._max = max_size

    def get(self, key: str) -> Optional[list[float]]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: list[float]) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self._max:
                self._store.popitem(last=False)
        self._store[key] = value


_cache = _LRUCache(max_size=settings.EMBEDDING_CACHE_MAX_SIZE)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── Provider interface ─────────────────────────────────────────────────────────

class EmbeddingsProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector dimension for this provider + model combination."""
        ...

    @abstractmethod
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* with no caching.  Called only by embed_texts()."""
        ...


# ── OpenAI provider ────────────────────────────────────────────────────────────

class OpenAIEmbeddingsProvider(EmbeddingsProvider):
    def __init__(self) -> None:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "EMBEDDINGS_PROVIDER=openai requires OPENAI_API_KEY to be set."
            )
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    @property
    def dimension(self) -> int:
        return settings.OPENAI_EMBEDDING_DIMENSIONS

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        batch_size = settings.EMBEDDING_BATCH_SIZE
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            logger.debug("OpenAI embed batch", extra={"batch_size": len(batch)})
            response = await self._client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=batch,
            )
            sorted_data = sorted(response.data, key=lambda d: d.index)
            all_embeddings.extend(item.embedding for item in sorted_data)
        return all_embeddings


# ── Local (sentence-transformers) provider ─────────────────────────────────────

class LocalEmbeddingsProvider(EmbeddingsProvider):
    """
    CPU-based embeddings via sentence-transformers.

    The model is loaded lazily on first use and cached for the process lifetime.
    encode() is CPU-bound, so it runs in the default ThreadPoolExecutor to avoid
    blocking the asyncio event loop.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None  # loaded lazily

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        logger.info(
            "Loading local embedding model",
            extra={"model": self._model_name},
        )
        self._model = SentenceTransformer(self._model_name)
        logger.info(
            "Local embedding model ready",
            extra={
                "model": self._model_name,
                "dim": self._model.get_sentence_embedding_dimension(),
            },
        )
        return self._model

    @property
    def dimension(self) -> int:
        return self._load().get_sentence_embedding_dimension()

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        batch_size = settings.EMBEDDING_BATCH_SIZE
        loop = asyncio.get_event_loop()
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            logger.debug("Local embed batch", extra={"batch_size": len(batch)})
            # encode() is synchronous and CPU-bound; run in thread pool.
            vecs: list[list[float]] = await loop.run_in_executor(
                None,
                lambda b=batch: model.encode(
                    b, normalize_embeddings=True, show_progress_bar=False
                ).tolist(),
            )
            all_embeddings.extend(vecs)
        return all_embeddings


# ── Singleton factory ──────────────────────────────────────────────────────────

_provider: Optional[EmbeddingsProvider] = None


def get_provider() -> EmbeddingsProvider:
    """Return the process-level provider singleton, creating it on first call."""
    global _provider
    if _provider is None:
        p = settings.EMBEDDINGS_PROVIDER.lower()
        if p == "openai":
            _provider = OpenAIEmbeddingsProvider()
        elif p == "local":
            _provider = LocalEmbeddingsProvider(settings.EMBEDDINGS_LOCAL_MODEL)
        else:
            raise ValueError(
                f"Unknown EMBEDDINGS_PROVIDER={p!r}. Valid values: 'local', 'openai'."
            )
    return _provider


def get_embedding_dimension() -> int:
    """
    Return the output vector dimension for the configured provider.

    Triggers model loading on first call (local provider only).
    Called by qdrant_store.ensure_collection() at startup so the Qdrant
    collection is created with the correct vector size.
    """
    return get_provider().dimension


# ── Public API (cache + dispatch) ──────────────────────────────────────────────

async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Return embeddings for *texts* in the same order, with LRU caching.

    Only uncached texts are forwarded to the provider.
    """
    if not texts:
        return []

    provider = get_provider()
    results: list[Optional[list[float]]] = [None] * len(texts)
    uncached_positions: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        h = _text_hash(text)
        cached = _cache.get(h)
        if cached is not None:
            results[i] = cached
            rag_cache_hit_total.inc()
        else:
            uncached_positions.append(i)
            uncached_texts.append(text)
            rag_cache_miss_total.inc()

    if uncached_texts:
        new_embeddings = await provider._embed_batch(uncached_texts)
        for pos, text, emb in zip(uncached_positions, uncached_texts, new_embeddings):
            results[pos] = emb
            _cache.set(_text_hash(text), emb)

    return [r for r in results if r is not None]  # type: ignore[misc]


async def embed_query(text: str) -> list[float]:
    """Convenience wrapper to embed a single query string."""
    return (await embed_texts([text]))[0]
