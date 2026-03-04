"""
Prompt construction for the RAG chat pipeline.

The system prompt enforces strict grounding: the LLM is instructed to answer
only from the supplied context and to produce a specific sentinel phrase when
context is insufficient.  The sentinel is exported so streaming.py can detect
and label no-answer events.

Source representation
---------------------
Context chunks are presented to the LLM as anonymous bracketed references
[S1], [S2], … so the model cites them cleanly without leaking internal
identifiers (doc_id, chunk_id, hashes) into the generated answer.  The real
identifiers are still returned to the client in the SSE citations event, which
is populated from the original RetrievedChunk objects – not from the text
shown to the LLM.
"""
from __future__ import annotations

from app.rag.retrieval import RetrievedChunk

# Sentinel: the LLM must produce exactly this phrase (case-insensitive) when
# the context does not contain enough information to answer the question.
NO_ANSWER_PHRASE = (
    "I don't have enough information in the provided documents to answer this question."
)

SYSTEM_PROMPT = f"""You are a precise question-answering assistant. \
Your answers MUST be strictly grounded in the context sources provided below.

Rules you must follow without exception:
1. Use ONLY the information present in the provided sources.  Do not rely on \
general knowledge or prior training.
2. If the sources do not contain sufficient information to answer, respond with \
exactly this sentence and nothing else:
   "{NO_ANSWER_PHRASE}"
3. Be concise and factually accurate.
4. Do not speculate, infer, or extrapolate beyond what the sources explicitly state.
5. When citing a source, use its bracketed label, e.g. "According to [S1], …" \
or "As stated in [S2] and [S3], …".
6. Do NOT output internal identifiers such as doc_id, chunk_id, hashes, or \
any machine-readable IDs.  Use only the [S<n>] notation."""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as anonymous bracketed sources [S1], [S2], …"""
    if not chunks:
        return "No context available."

    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[S{i}]\n{chunk.content}")

    return "\n\n---\n\n".join(parts)


def build_messages(query: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """
    Construct the OpenAI-compatible messages list for a RAG chat turn.

    The context is injected into the user message so it stays within the
    model's attended tokens and is clearly separated from the question.
    """
    context = build_context_block(chunks)
    user_content = (
        f"Sources:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
