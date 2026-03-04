"""All Prometheus metric objects used across the service."""
from prometheus_client import Counter, Histogram

# ── HTTP ───────────────────────────────────────────────────────────────────────

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ── RAG – Retrieval ────────────────────────────────────────────────────────────

rag_retrieval_latency_ms = Histogram(
    "rag_retrieval_latency_ms",
    "End-to-end retrieval latency (embed query + Qdrant search) in ms",
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)

rag_retrieval_top_score = Histogram(
    "rag_retrieval_top_score",
    "Cosine similarity of the top-ranked retrieved chunk",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── RAG – LLM ─────────────────────────────────────────────────────────────────

rag_llm_latency_ms = Histogram(
    "rag_llm_latency_ms",
    "Time-to-last-token for LLM calls in ms",
    buckets=[100, 250, 500, 1000, 2500, 5000, 10000, 30000],
)

rag_tokens_in_total = Counter(
    "rag_tokens_in_total",
    "Cumulative prompt tokens sent to the LLM",
)

rag_tokens_out_total = Counter(
    "rag_tokens_out_total",
    "Cumulative completion tokens received from the LLM",
)

# ── RAG – Cache ────────────────────────────────────────────────────────────────

rag_cache_hit_total = Counter(
    "rag_cache_hit_total",
    "Embedding cache hits (chunk already embedded in-process)",
)

rag_cache_miss_total = Counter(
    "rag_cache_miss_total",
    "Embedding cache misses (chunk must be embedded via API)",
)

# ── RAG – Quality signals ──────────────────────────────────────────────────────

rag_no_answer_total = Counter(
    "rag_no_answer_total",
    "Requests where RAG returned a no-answer response",
    ["tenant_id"],
)

# ── Ingest ─────────────────────────────────────────────────────────────────────

rag_ingest_chunks_total = Counter(
    "rag_ingest_chunks_total",
    "Total new chunks written to Qdrant",
    ["tenant_id"],
)

rag_ingest_docs_total = Counter(
    "rag_ingest_docs_total",
    "Total documents ingested",
    ["tenant_id"],
)
