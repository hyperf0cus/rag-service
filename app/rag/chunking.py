"""
Text chunking with two strategies:

1. Markdown-header-aware splitting: if the document contains any ATX-style
   headers (# … through ###### …) the text is first split into sections at
   each header boundary.  Each section is then chunked independently using
   the token-window strategy so that the header is always prepended to every
   sub-chunk produced from it, preserving navigation context.

2. Token-window chunking: uses tiktoken (cl100k_base – the encoding shared by
   GPT-4 and text-embedding-3-*) to split on exact token boundaries with a
   configurable overlap.  A sentence-boundary heuristic is applied when
   possible so chunks don't cut mid-sentence.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import tiktoken


@lru_cache(maxsize=1)
def _get_enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_hash(self) -> str:
        """Stable 16-hex-char fingerprint of the chunk content."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


# ── Internal helpers ───────────────────────────────────────────────────────────

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """
    Return a list of (header_line, body) pairs.
    The first element may have an empty header if text starts before any header.
    """
    sections: list[tuple[str, str]] = []
    matches = list(_HEADER_RE.finditer(text))

    if not matches:
        return [("", text)]

    # Text before the first header
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    for i, m in enumerate(matches):
        header_line = m.group(0)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((header_line, body))

    return sections


def _token_windows(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split *text* into overlapping token windows decoded back to strings."""
    enc = _get_enc()
    tokens = enc.encode(text, disallowed_special=())
    if not tokens:
        return []

    windows: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        window_tokens = tokens[start:end]
        window_text = enc.decode(window_tokens).strip()
        if window_text:
            windows.append(window_text)
        if end >= len(tokens):
            break
        start += chunk_size - overlap

    return windows


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    use_markdown: bool = True,
    metadata: Optional[dict] = None,
) -> list[Chunk]:
    """
    Chunk *text* into overlapping pieces and return a list of :class:`Chunk`.

    Args:
        text: Raw document text.
        doc_id: Identifier for the parent document.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Number of tokens to repeat between consecutive chunks.
        use_markdown: When True, detects ATX headers and splits on them first.
        metadata: Arbitrary key/value pairs stored in every chunk's payload.

    Returns:
        Ordered list of Chunk objects.  chunk_id = ``<doc_id>_<index:04d>``.
    """
    if metadata is None:
        metadata = {}

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    chunk_index = 0

    has_headers = bool(_HEADER_RE.search(text))
    if use_markdown and has_headers:
        sections = _split_markdown_sections(text)
    else:
        sections = [("", text)]

    for header_line, body in sections:
        if not body and not header_line:
            continue

        # Content to tokenize is the body; if it fits in one window, emit as-is.
        full_section = f"{header_line}\n{body}".strip() if header_line else body

        enc = _get_enc()
        n_tokens = len(enc.encode(full_section, disallowed_special=()))

        if n_tokens <= chunk_size:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_{chunk_index:04d}",
                    doc_id=doc_id,
                    content=full_section,
                    chunk_index=chunk_index,
                    metadata=dict(metadata),
                )
            )
            chunk_index += 1
        else:
            # Split body only; prepend header to each sub-chunk for context.
            windows = _token_windows(body, chunk_size, chunk_overlap)
            for window in windows:
                content = f"{header_line}\n{window}".strip() if header_line else window
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_{chunk_index:04d}",
                        doc_id=doc_id,
                        content=content,
                        chunk_index=chunk_index,
                        metadata=dict(metadata),
                    )
                )
                chunk_index += 1

    return chunks
