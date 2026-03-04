"""
LLM provider abstraction.

Architecture
------------
``LLMProvider`` is an abstract base class defining the contract:
  - ``stream_chat``: async generator yielding token strings one-by-one.
  - ``chat``: non-streaming convenience method returning the full response.

Providers
---------
``OpenAIProvider``  – OpenAI Chat Completions API (requires OPENAI_API_KEY).
``OllamaProvider``  – Local Ollama server via /api/chat (no API key needed).

The active provider is selected by ``get_llm_provider()`` based on
``settings.LLM_PROVIDER``.  ``get_llm_provider()`` returns ``None`` only
when the chosen provider cannot be configured (e.g. OpenAI with no key);
callers should treat ``None`` as a 503 signal.

Error taxonomy
--------------
``LLMUnavailableError`` is raised for *expected* operational failures that
should cause callers to fall back gracefully (sources-only response):
  - OpenAI: 401 AuthenticationError, 429 RateLimitError
  - Ollama: connection refused, HTTP 4xx/5xx from the Ollama server

All other exceptions propagate unchanged and are treated as internal errors.

Observability
-------------
All providers record:
  - ``rag_llm_latency_ms``: time from first token request to last token.
  - ``rag_tokens_in_total`` / ``rag_tokens_out_total``: prompt / completion
    token counts (best-effort; Ollama reports these on the final stream event).
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import rag_llm_latency_ms, rag_tokens_in_total, rag_tokens_out_total

logger = get_logger(__name__)


# ── Exceptions ─────────────────────────────────────────────────────────────────

class LLMUnavailableError(Exception):
    """
    Raised for expected provider failures (auth errors, rate limits, connection
    refused).  Callers should fall back to a sources-only response rather than
    surfacing raw provider details to end users.
    """


# ── Abstract base ──────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Yield text tokens as they arrive."""
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Return the complete response text."""
        ...


# ── OpenAI implementation ──────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model or settings.OPENAI_CHAT_MODEL

    async def stream_chat(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        max_tokens = max_tokens or settings.OPENAI_MAX_TOKENS
        temperature = temperature if temperature is not None else settings.OPENAI_TEMPERATURE

        import openai as _openai

        t0 = time.monotonic()
        tokens_out = 0
        tokens_in = 0

        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                # Final chunk with usage info (no content delta)
                if chunk.usage:
                    tokens_in = chunk.usage.prompt_tokens
                    tokens_out = chunk.usage.completion_tokens
                    continue

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if delta.content is not None:
                    tokens_out += 1  # fallback counter when usage not present
                    yield delta.content

        except _openai.AuthenticationError as exc:
            raise LLMUnavailableError(
                "OpenAI authentication failed – check OPENAI_API_KEY"
            ) from exc
        except _openai.RateLimitError as exc:
            raise LLMUnavailableError("OpenAI rate limit exceeded") from exc

        latency_ms = (time.monotonic() - t0) * 1000
        rag_llm_latency_ms.observe(latency_ms)
        if tokens_in:
            rag_tokens_in_total.inc(tokens_in)
        if tokens_out:
            rag_tokens_out_total.inc(tokens_out)

        logger.debug(
            "LLM stream complete",
            extra={
                "model": self._model,
                "latency_ms": round(latency_ms, 1),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            },
        )

    async def chat(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        max_tokens = max_tokens or settings.OPENAI_MAX_TOKENS
        temperature = temperature if temperature is not None else settings.OPENAI_TEMPERATURE

        t0 = time.monotonic()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        rag_llm_latency_ms.observe(latency_ms)

        if response.usage:
            rag_tokens_in_total.inc(response.usage.prompt_tokens)
            rag_tokens_out_total.inc(response.usage.completion_tokens)

        return response.choices[0].message.content or ""


# ── Ollama implementation ──────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """
    LLM provider backed by a local Ollama server.

    Uses the ``/api/chat`` endpoint with ``stream=true`` for token-by-token
    streaming.  Each line in the response body is a JSON object; the final
    line carries ``"done": true`` along with token-count fields.

    Connection failures and HTTP errors raise ``LLMUnavailableError``, which
    ``streaming.stream_rag_response`` catches and uses to emit a graceful
    sources-only fallback instead of an error event.
    """

    def __init__(self, base_url: str, model: str, temperature: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature

    async def stream_chat(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        import httpx

        max_tokens = max_tokens or settings.OPENAI_MAX_TOKENS
        temperature = temperature if temperature is not None else self._temperature

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        t0 = time.monotonic()
        tokens_out = 0

        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, timeout=httpx.Timeout(120.0, connect=10.0)
            ) as client:
                async with client.stream("POST", "/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            tokens_out += 1
                            yield content
                        if data.get("done"):
                            prompt_tokens = data.get("prompt_eval_count", 0)
                            completion_tokens = data.get("eval_count", 0)
                            if prompt_tokens:
                                rag_tokens_in_total.inc(prompt_tokens)
                            if completion_tokens:
                                rag_tokens_out_total.inc(completion_tokens)
                            break
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self._base_url} – "
                "is the Ollama container running and the model pulled?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMUnavailableError(
                f"Ollama returned HTTP {exc.response.status_code}"
            ) from exc

        latency_ms = (time.monotonic() - t0) * 1000
        rag_llm_latency_ms.observe(latency_ms)
        logger.debug(
            "Ollama stream complete",
            extra={
                "model": self._model,
                "latency_ms": round(latency_ms, 1),
                "tokens_out": tokens_out,
            },
        )

    async def chat(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        import httpx

        max_tokens = max_tokens or settings.OPENAI_MAX_TOKENS
        temperature = temperature if temperature is not None else self._temperature

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, timeout=httpx.Timeout(120.0, connect=10.0)
            ) as client:
                resp = await client.post("/api/chat", json=payload)
                resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self._base_url} – "
                "is the Ollama container running and the model pulled?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMUnavailableError(
                f"Ollama returned HTTP {exc.response.status_code}"
            ) from exc

        latency_ms = (time.monotonic() - t0) * 1000
        rag_llm_latency_ms.observe(latency_ms)

        data = resp.json()
        content = data.get("message", {}).get("content", "")
        if data.get("prompt_eval_count"):
            rag_tokens_in_total.inc(data["prompt_eval_count"])
        if data.get("eval_count"):
            rag_tokens_out_total.inc(data["eval_count"])

        logger.debug(
            "Ollama chat complete",
            extra={"model": self._model, "latency_ms": round(latency_ms, 1)},
        )
        return content


# ── Factory ────────────────────────────────────────────────────────────────────

def get_llm_provider() -> Optional[LLMProvider]:
    """
    Return a configured LLM provider based on ``settings.LLM_PROVIDER``.

    Returns ``None`` only when the chosen provider cannot be initialised
    (currently only ``openai`` without an API key).  Callers should check for
    ``None`` and return an informative HTTP 503.

    ``ollama`` always returns a provider instance (no API key required);
    connection / HTTP failures surface as ``LLMUnavailableError`` inside
    ``stream_chat`` and trigger a graceful sources-only fallback in the caller.
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "ollama":
        return OllamaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=settings.OLLAMA_TEMPERATURE,
        )

    if provider == "openai":
        if settings.OPENAI_API_KEY:
            return OpenAIProvider(api_key=settings.OPENAI_API_KEY)
        return None

    logger.warning(
        "Unknown LLM_PROVIDER; falling back to None",
        extra={"llm_provider": provider},
    )
    return None
