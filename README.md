# rag-service

A production-grade Retrieval-Augmented Generation (RAG) service built with
**FastAPI**, **Qdrant**, and **OpenAI**.  No LangChain or LlamaIndex – every
primitive is implemented explicitly so the code is easy to read, test, and
extend.

---

## Architecture

```
┌──────────┐  POST /ingest   ┌─────────────────────────────────────────────┐
│  Client  │ ─────────────►  │  Chunk (tiktoken)  →  Embed (OpenAI)        │
│          │                 │  →  Upsert to Qdrant (dedup by chunk_hash)   │
│          │  POST /search   └─────────────────────────────────────────────┘
│          │ ─────────────►  Embed query → Qdrant ANN search (cosine)
│          │
│          │  POST /chat     ┌──────────────────────────────────────────────┐
│          │ ─────────────►  │  Retrieve chunks → Build grounded prompt     │
│          │  SSE stream ◄── │  → Stream tokens from OpenAI                 │
│          │                 │  → Emit final citations event                │
└──────────┘                 └──────────────────────────────────────────────┘
```

---

## LLM providers

| Provider | `LLM_PROVIDER` | Requires | Notes |
|----------|----------------|----------|-------|
| Ollama (default for local) | `ollama` | nothing | Fully local; pull model first |
| OpenAI | `openai` | `OPENAI_API_KEY` | Cloud; billed per token |

---

## Local LLM (Ollama + Qwen)

Run `/chat` completely locally — no cloud API keys, no usage fees.

### 1. Start the stack

```bash
cp .env.example .env          # LLM_PROVIDER=ollama is already set
docker compose up --build
```

### 2. Pull the model

The Ollama container starts immediately but has no models cached yet.
Pull Qwen 2.5 7B Instruct (≈ 5 GB):

```bash
docker exec -it rag-service-ollama-1 ollama pull qwen2.5:7b-instruct
```

> **VRAM guidance**
> | Model | Minimum VRAM | Notes |
> |-------|-------------|-------|
> | `qwen2.5:7b-instruct` | 6 GB | Default; good quality/speed balance |
> | `qwen2.5:3b-instruct` | 3 GB | Faster, lower quality |
> | `qwen2.5:14b-instruct` | 10 GB | Better reasoning |
>
> CPU inference works but is slow (~20–60 tok/s on a modern laptop).
> Uncomment the `deploy.resources` GPU block in `docker-compose.yml` to
> enable NVIDIA GPU acceleration.

### 3. Ingest and chat

```bash
# Ingest a document (uses local sentence-transformers embeddings)
curl -X POST http://localhost:8010/ingest \
  -F "tenant_id=acme" \
  -F "files=@docs/handbook.txt"

# Stream a chat response (no API key needed)
curl -N -X POST http://localhost:8010/chat \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "acme", "query": "What is the onboarding process?"}'
```

---

## Embeddings providers

| Provider | `EMBEDDINGS_PROVIDER` | Requires | Vector dim |
|----------|-----------------------|----------|------------|
| sentence-transformers (default) | `local` | nothing | 384 (all-MiniLM-L6-v2) |
| OpenAI Embeddings API | `openai` | `OPENAI_API_KEY` | 1536 (text-embedding-3-small) |

**`/ingest` and `/search` work without any API key when `EMBEDDINGS_PROVIDER=local`.**
Only `/chat` (the LLM step) still requires `OPENAI_API_KEY`.

> **Note:** the local model (~90 MB) is downloaded from HuggingFace on first
> use.  Set `HF_HOME` to a Docker volume to avoid re-downloading on every
> container start.

---

## Quick Start

### Prerequisites

- Docker ≥ 24 and Docker Compose ≥ 2.20
- An OpenAI API key (**optional** – only needed for `/chat`)

```bash
# 1. Clone / enter the repo
cd rag-service

# 2. Configure environment
cp .env.example .env
# For ingest + search only, no edits needed (EMBEDDINGS_PROVIDER=local by default).
# For /chat, also set: OPENAI_API_KEY=sk-...

# 3. Start everything
docker compose up --build
```

Services:
| Service    | URL                      |
|------------|--------------------------|
| API        | http://localhost:8000    |
| Qdrant UI  | http://localhost:6333/dashboard |
| Prometheus | http://localhost:9090    |

Check the API is running:
```bash
curl http://localhost:8000/health
# {"status":"ok","version":"0.1.0","uptime":3.14}
```

---

## Local Development (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Start Qdrant separately
docker run -p 6333:6333 qdrant/qdrant:v1.10.0

# Run the API with auto-reload
uvicorn app.main:app --reload --port 8000
```

---

## Ingest Documents

### Via file upload (multipart/form-data)

```bash
curl -X POST http://localhost:8000/ingest \
  -F "tenant_id=acme" \
  -F 'files=@docs/product_manual.txt' \
  -F 'files=@docs/faq.md' \
  -F 'metadata={"source":"confluence","team":"support"}'
```

Response:
```json
{
  "doc_ids": ["acme_a1b2c3d4e5f6", "acme_7g8h9i0j1k2l"],
  "chunks_ingested": 47,
  "chunks_skipped": 0,
  "total_chunks": 47
}
```

### Via server-side paths (JSON body)

Useful when documents are already on the server (e.g. mounted volume):

```bash
curl -X POST http://localhost:8000/ingest/paths \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme",
    "paths": ["/app/data/docs/manual.txt"],
    "metadata": {"version": "2.1"}
  }'
```

---

## Debug Retrieval

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme",
    "query": "What is the refund policy?",
    "top_k": 5
  }'
```

Response includes ranked chunks with `score`, `chunk_id`, `doc_id`, and full `content`.

---

## Chat (SSE Streaming)

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme",
    "query": "What is the refund policy for digital products?",
    "top_k": 5
  }'
```

Example event stream:
```
data: {"type": "token", "content": "Digital"}
data: {"type": "token", "content": " products"}
data: {"type": "token", "content": " are"}
...
data: {"type": "citations", "citations": [
  {"rank": 1, "doc_id": "acme_a1b2", "chunk_id": "acme_a1b2_0003",
   "score": 0.8821, "snippet": "Digital products are non-refundable once downloaded…"}
]}
data: [DONE]
```

When the context does not contain an answer:
```
data: {"type": "token", "content": "I don't have enough information …"}
data: {"type": "citations", "citations": []}
data: [DONE]
```

---

## Feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme",
    "query": "What is the refund policy?",
    "rating": 5,
    "comment": "Perfect answer",
    "conversation_id": "conv_xyz"
  }'
```

Feedback is appended to `data/feedback.jsonl`.

---

## Observability

### Prometheus metrics

```bash
curl http://localhost:8000/metrics
```

Key metrics:

| Metric | Description |
|--------|-------------|
| `http_requests_total` | Request count by method/endpoint/status |
| `http_request_duration_seconds` | Request latency histogram |
| `rag_retrieval_latency_ms` | Embed + Qdrant search latency |
| `rag_llm_latency_ms` | LLM time-to-last-token |
| `rag_tokens_in_total` | Prompt tokens sent to LLM |
| `rag_tokens_out_total` | Completion tokens received |
| `rag_cache_hit_total` | Embedding cache hits |
| `rag_cache_miss_total` | Embedding cache misses |
| `rag_no_answer_total` | No-answer responses |
| `rag_retrieval_top_score` | Top cosine score histogram |
| `rag_ingest_chunks_total` | Chunks upserted |

### Structured JSON logs

Every log line is a JSON object:
```json
{
  "ts": "2025-01-15T10:23:45.123Z",
  "level": "INFO",
  "logger": "app.rag.retrieval",
  "msg": "Retrieval done",
  "request_id": "a3f1…",
  "tenant_id": "acme",
  "n_results": 5,
  "top_score": 0.8821,
  "latency_ms": 142.3
}
```

---

## Evaluation

The evaluation script measures retrieval quality against a golden dataset
(no LLM call required – pure retrieval evaluation).

### Golden dataset format (`data/eval/golden.jsonl`)

```jsonl
{"query": "What is the refund policy?", "relevant_chunk_ids": ["doc_001_0002"], "relevant_doc_ids": ["doc_001"]}
{"query": "How do I reset my password?", "relevant_doc_ids": ["doc_002"]}
```

- `relevant_chunk_ids` takes precedence over `relevant_doc_ids` when both are present.
- Either field may be omitted; the other is used as fallback.

### Run evaluation

```bash
python -m app.eval.run \
    --tenant acme \
    --dataset data/eval/golden.jsonl \
    --k 3 5 10
```

Example output:
```
Evaluating 50 queries | tenant=acme | k=[3, 5, 10]

  [1/50] R@3=1.000  R@5=1.000  R@10=1.000  RR=1.000  | What is the refund policy?
  [2/50] R@3=0.500  R@5=1.000  R@10=1.000  RR=0.333  | How do I reset my password?
  …

────────────────────────────────────────────────
  METRIC                     VALUE
────────────────────────────────────────────────
  Recall@3                  0.7640
  Recall@5                  0.8420
  Recall@10                 0.9100
  HitRate@3                 0.8200
  HitRate@5                 0.8800
  HitRate@10                0.9400
  MRR                       0.7213
  Queries evaluated             50
────────────────────────────────────────────────

  Report saved → reports/eval_acme_1736940000.json
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```

Tests run without a live Qdrant instance or OpenAI API key (both are mocked).

---

## Configuration Reference

All configuration is via environment variables (or `.env`).

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDINGS_PROVIDER` | `local` | `local` or `openai` |
| `EMBEDDINGS_LOCAL_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model ID (local provider) |
| `OPENAI_API_KEY` | — | Required for `/chat`; also required when `EMBEDDINGS_PROVIDER=openai` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (openai provider) |
| `OPENAI_EMBEDDING_DIMENSIONS` | `1536` | Must match the model's native dimension (openai provider) |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat completion model |
| `OPENAI_MAX_TOKENS` | `1024` | Max tokens in LLM response |
| `OPENAI_TEMPERATURE` | `0.0` | LLM temperature (0 = deterministic) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP endpoint |
| `QDRANT_API_KEY` | — | Optional Qdrant API key |
| `QDRANT_COLLECTION_NAME` | `rag_chunks` | Collection for all tenants |
| `CHUNK_SIZE` | `512` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap tokens between chunks |
| `RETRIEVAL_TOP_K` | `5` | Default number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.30` | Minimum cosine score (0 = no filter) |
| `EMBEDDING_BATCH_SIZE` | `64` | Texts per OpenAI embeddings API call |
| `EMBEDDING_CACHE_MAX_SIZE` | `10000` | In-process LRU cache capacity |
| `FEEDBACK_LOG_PATH` | `data/feedback.jsonl` | Feedback JSONL file path |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Design Trade-offs & Notes

### Multi-tenancy
All tenants share a single Qdrant collection.  Tenant isolation is enforced by
a mandatory `tenant_id` payload filter on every query.  This keeps operations
simple and avoids per-tenant collection provisioning overhead.  The trade-off
is that a single tenant with millions of vectors can slow searches for other
tenants; for extreme multi-tenancy consider sharding by tenant.

### Chunking strategy
Documents are chunked using tiktoken (cl100k\_base) for accurate token counts.
Markdown ATX headers (`# … ######`) are detected and used as primary split
points so each chunk retains its section heading for context.  A configurable
token overlap prevents information loss at boundaries.

### Embedding cache
The in-process LRU cache avoids redundant API calls when the same content is
re-ingested.  It is intentionally process-local (no Redis) so each replica
is stateless.  Cache hit rate is exposed via `rag_cache_hit_total`.

### LLM provider interface
`LLMProvider` is an abstract base class.  `OpenAIProvider` and
`OllamaProvider` are the two built-in implementations.
`OllamaProvider` uses the `/api/chat` endpoint with line-by-line JSON
streaming; if Ollama is unreachable it raises `RuntimeError`, which
`stream_rag_response` catches and forwards as an SSE `{"type":"error"}`
event so the client always receives a terminated stream.
To add Anthropic Claude: subclass `LLMProvider`, implement `stream_chat`
and `chat`, and extend `get_llm_provider` to handle `LLM_PROVIDER=anthropic`.

### Similarity threshold
`SIMILARITY_THRESHOLD=0.30` is a conservative default.  If your embeddings
cluster tightly you may want to raise it.  Setting it to `0.0` disables
filtering.  The eval script defaults to `0.0` so retrieval is evaluated at
full recall.

### Deduplication and deterministic doc IDs
`doc_id` is derived from `SHA-1(file_bytes)[:12]` prefixed by `tenant_id`,
so re-ingesting the same file bytes always returns the same `doc_id`
regardless of filename or server path.  Copying a file to a different folder
and re-ingesting it still deduplicates against the original.
Chunk-level deduplication by `(chunk_hash, tenant_id)` remains unchanged —
re-ingesting an unchanged file inserts zero new chunks.

### Qdrant Query Points API
We use `AsyncQdrantClient.query_points()` instead of the deprecated
`search()` method.  The response is normalized by `_extract_points()` in
`qdrant_store.py` which handles shape differences across qdrant-client
versions (`response.points`, `response.result.points`, or a plain list).
`query_points` is also the foundation for future hybrid / multi-stage
queries (sparse + dense, re-ranking) that the old `search()` path did not
support.
