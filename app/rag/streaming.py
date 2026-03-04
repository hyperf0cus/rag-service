"""
SSE streaming helpers.

Event protocol (each event is a ``data: <json>\\n\\n`` frame):

  data: {"type": "token",     "content": "<text segment>"}
  data: {"type": "citations", "citations": [...]}
  data: {"type": "error",     "message": "<error text>"}   # internal errors only
  data: [DONE]

Token buffering
---------------
Raw LLM tokens are small (often 1-4 characters).  Emitting one SSE frame per
raw token creates unnecessary network overhead and makes client rendering
choppy.  Tokens are accumulated in a small buffer and flushed as a single
``token`` event when any of the following conditions is met:

  * The buffer contains a whitespace or punctuation character (natural
    word/sentence boundary – yields smooth word-by-word rendering).
  * The buffer has grown to ``_MAX_BUF_CHARS`` characters (12) without
    hitting a boundary – caps latency for long unbroken token sequences.
  * ``_MAX_FLUSH_INTERVAL`` seconds (80 ms) have elapsed since the last
    flush – time-based safety valve.

The full accumulated text (all raw tokens) is still kept in ``accumulated``
for no-answer sentinel detection.

Provider-error fallback
-----------------------
When ``stream_chat`` raises ``LLMUnavailableError`` (e.g. 401, 429, Ollama
unreachable) the stream does NOT emit an error event.  Instead it emits:

  data: {"type": "token", "content": "<_FALLBACK_MSG>"}
  data: {"type": "citations", "citations": [...retrieved sources...]}
  data: [DONE]

This keeps the client UX intact – the user sees a human-readable message and
the retrieved sources rather than a raw provider error string.

The ``{"type": "error"}`` event is reserved for unexpected internal failures
(bugs, serialisation errors, etc.) where there is no meaningful fallback.
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator

from app.rag.llm_client import LLMProvider, LLMUnavailableError
from app.rag.prompt import NO_ANSWER_PHRASE, build_messages
from app.rag.retrieval import RetrievedChunk

_FALLBACK_MSG = "LLM unavailable — here are the most relevant sources found."

# ── Token-buffer tuning constants ──────────────────────────────────────────────
# Flush the SSE buffer when any of these conditions is met:
_FLUSH_CHARS = frozenset(" \t\n\r.,!?;:")  # natural word/sentence boundaries
_MAX_BUF_CHARS = 12                         # max characters before forced flush
_MAX_FLUSH_INTERVAL = 0.08                  # seconds – time-based safety valve


# ── SSE formatting ─────────────────────────────────────────────────────────────

def _sse(data: dict | str) -> str:
    """Return a complete SSE frame ending with a blank line."""
    payload = json.dumps(data) if isinstance(data, dict) else data
    return f"data: {payload}\n\n"


# ── Citation builder ───────────────────────────────────────────────────────────

def _build_citation(chunk: RetrievedChunk, rank: int) -> dict:
    return {
        "rank": rank,
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "score": round(chunk.score, 4),
        "snippet": (
            chunk.content[:250] + "…"
            if len(chunk.content) > 250
            else chunk.content
        ),
    }


# ── Main streaming generator ───────────────────────────────────────────────────

async def stream_rag_response(
    query: str,
    chunks: list[RetrievedChunk],
    llm_provider: LLMProvider,
    tenant_id: str,
) -> AsyncIterator[str]:
    """
    Async generator that yields SSE-formatted strings for a single RAG turn.

    Flow:
      1. If no chunks were retrieved, immediately emit the no-answer token
         event plus an empty citations event, then [DONE].
      2. Otherwise stream LLM tokens through the buffer, accumulate the full
         response to detect the no-answer sentinel, then emit citations.
      3. On ``LLMUnavailableError`` (provider auth/rate-limit/connection
         failures): discard buffer, emit a fallback message + full citations,
         then [DONE].
      4. On any other exception: discard buffer, emit an error event, then
         [DONE].
    """
    from app.core.logging import get_logger
    from app.core.metrics import rag_no_answer_total

    logger = get_logger(__name__)

    if not chunks:
        rag_no_answer_total.labels(tenant_id=tenant_id).inc()
        yield _sse({"type": "token", "content": NO_ANSWER_PHRASE})
        yield _sse({"type": "citations", "citations": []})
        yield _sse("[DONE]")
        return

    messages = build_messages(query, chunks)
    citations = [_build_citation(c, i + 1) for i, c in enumerate(chunks)]
    accumulated: list[str] = []

    # Token buffer state
    buf: list[str] = []
    buf_len = 0
    last_flush = time.monotonic()

    try:
        async for token in llm_provider.stream_chat(messages):
            accumulated.append(token)
            buf.append(token)
            buf_len += len(token)

            now = time.monotonic()
            if (
                buf_len >= _MAX_BUF_CHARS
                or now - last_flush >= _MAX_FLUSH_INTERVAL
                or _FLUSH_CHARS.intersection(token)
            ):
                yield _sse({"type": "token", "content": "".join(buf)})
                buf.clear()
                buf_len = 0
                last_flush = now

        # Flush any remaining buffered content after the stream ends.
        if buf:
            yield _sse({"type": "token", "content": "".join(buf)})

    except LLMUnavailableError as exc:
        logger.warning(
            "LLM unavailable, falling back to sources only",
            extra={"error": str(exc)},
        )
        yield _sse({"type": "token", "content": _FALLBACK_MSG})
        yield _sse({"type": "citations", "citations": citations})
        yield _sse("[DONE]")
        return

    except Exception as exc:
        logger.error("LLM streaming error", extra={"error": str(exc)})
        yield _sse({"type": "error", "message": str(exc)})
        yield _sse("[DONE]")
        return

    full_response = "".join(accumulated)
    no_answer = NO_ANSWER_PHRASE.lower() in full_response.lower()
    if no_answer:
        rag_no_answer_total.labels(tenant_id=tenant_id).inc()

    yield _sse({"type": "citations", "citations": [] if no_answer else citations})
    yield _sse("[DONE]")
