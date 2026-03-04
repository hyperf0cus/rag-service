# rag-service

A production-ready Retrieval-Augmented Generation service built with **FastAPI** and **Qdrant** — no LangChain or LlamaIndex. Every primitive (chunking, embedding, retrieval, prompting, streaming) is implemented explicitly. Documents are ingested with tenant isolation and content-hash deduplication; queries stream back via SSE with bracketed `[S1]/[S2]` source references in the generated text and a structured `citations` event carrying `doc_id`, `chunk_id`, score, and a snippet. Embeddings default to a local sentence-transformers model (no API key required), and `/chat` supports both a local Ollama/Qwen LLM and OpenAI — if the LLM is unavailable the service falls back gracefully to a sources-only response.

---

## Features

- **Two ingest paths:** multipart file upload (`POST /ingest`) and server-side paths (`POST /ingest/paths`).
- **Deterministic doc IDs:** `doc_id = {tenant_id}_{sha1(file_bytes)[:12]}` — re-ingesting the same bytes returns the same ID regardless of filename or path.
- **Idempotent ingest:** chunk-level deduplication by `(chunk_hash, tenant_id)`; re-ingesting an unchanged file inserts zero new chunks.
- **Retrieval debug endpoint:** `POST /search` returns ranked chunks with score, chunk\_id, doc\_id, and content — no LLM call.
- **SSE streaming chat (`POST /chat`):** tokens buffered into word-sized frames; answer text uses `[S1]`/`[S2]` references; stream always ends with a `citations` event (`doc_id`, `chunk_id`, score, snippet) then `[DONE]`.
- **LLM fallback:** if the LLM is unreachable (auth error, rate limit, connection refused) the stream emits a human-readable message and real citations instead of a raw error.
- **Prometheus metrics** at `GET /metrics` — request counts/latencies, retrieval/LLM latencies, token totals, cache hit rate, no-answer events.
- **Structured JSON logs** with per-request `request_id` and `tenant_id` propagated via context vars.
- **Evaluation CLI:** recall\@k, hit-rate\@k, MRR against a golden JSONL dataset; reports written to `reports/`.
- **Fully local option:** `EMBEDDINGS_PROVIDER=local` + `LLM_PROVIDER=ollama` — zero API keys needed end-to-end.

---

## Quickstart (Docker Compose)

**Prerequisites:** Docker ≥ 24, Docker Compose ≥ 2.20. No API key required for the default local stack.

```bash
cp .env.example .env          # defaults: EMBEDDINGS_PROVIDER=local, LLM_PROVIDER=ollama
docker compose up --build
```

After the stack is healthy, **pull the Ollama model** (≈ 5 GB, one-time):

```bash
docker exec -it rag-service-ollama-1 ollama pull qwen2.5:7b-instruct
```

> CPU inference works without a GPU; expect ≈ 5–20 tok/s depending on hardware.
> For NVIDIA GPU acceleration, uncomment the `deploy.resources` block in `docker-compose.yml`.

Running services:

| Service     | Host URL                          |
|-------------|-----------------------------------|
| API         | http://localhost:8010             |
| Qdrant UI   | http://localhost:6333/dashboard   |
| Prometheus  | http://localhost:9090             |

```bash
curl http://localhost:8010/health
# {"status":"ok","version":"0.1.0","uptime":2.31}
```

---

## Demo in 60 seconds

Run all commands after `docker compose up --build` and the model pull above.

```bash
# 1. Create demo documents
mkdir -p data/demo_docs

cat > data/demo_docs/policy.txt << 'EOF'
Refund policy: full refunds are available within 30 days of purchase.
Digital products are non-refundable once downloaded.
EOF

cat > data/demo_docs/faq.txt << 'EOF'
To reset your password go to Settings > Security > Change Password.
Contact support@example.com for account-related issues.
EOF

# 2. Ingest both files
curl -s -X POST http://localhost:8010/ingest \
  -F "tenant_id=demo" \
  -F "files=@data/demo_docs/policy.txt" \
  -F "files=@data/demo_docs/faq.txt" | python3 -m json.tool
# {
#   "doc_ids": ["demo_3a9f12b84c01", "demo_7e2a4108cd99"],
#   "chunks_ingested": 2, "chunks_skipped": 0, "total_chunks": 2
# }

# 3. Debug retrieval (no LLM)
curl -s -X POST http://localhost:8010/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Can I get a refund?", "tenant_id": "demo", "top_k": 3}' \
  | python3 -m json.tool

# 4. Chat via SSE (streams to terminal)
curl -N -X POST http://localhost:8010/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "tenant_id": "demo"}'
```

Expected SSE output shape:

```
data: {"type": "token", "content": "According to [S1], full refunds are available"}
data: {"type": "token", "content": " within 30 days of purchase."}
...
data: {"type": "citations", "citations": [
  {"rank": 1, "doc_id": "demo_3a9f12b84c01", "chunk_id": "demo_3a9f12b84c01_0000",
   "score": 0.9241, "snippet": "Refund policy: full refunds are available…"}
]}
data: [DONE]
```

When no relevant sources exist:

```
data: {"type": "token", "content": "I don't have enough information in the provided documents to answer this question."}
data: {"type": "citations", "citations": []}
data: [DONE]
```

```bash
# 5. Check metrics
curl -s http://localhost:8010/metrics | grep -E "^rag_retrieval_latency|^rag_ingest_chunks"
```

---

## Local Development (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,local]"    # [local] installs sentence-transformers

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant:v1.10.0

# Optional: start Ollama
docker run -p 11434:11434 ollama/ollama:latest

uvicorn app.main:app --reload --port 8000
```

Set `OLLAMA_BASE_URL=http://localhost:11434` in `.env` when running outside Docker Compose.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```

Tests run without a live Qdrant instance, embedding model, or LLM — all three are replaced by mocks in `tests/conftest.py`.

---

## Configuration Reference

All settings are read from environment variables (or `.env`).

| Variable | Default | Description |
|----------|---------|-------------|
| **Embeddings** | | |
| `EMBEDDINGS_PROVIDER` | `local` | `local` (sentence-transformers) or `openai` |
| `EMBEDDINGS_LOCAL_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model ID (384-dim, ≈ 90 MB) |
| **LLM** | | |
| `LLM_PROVIDER` | `openai` | `ollama` (local, no key) or `openai` (requires key) |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama server URL (`http://localhost:11434` outside Docker) |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Model to use; must be pulled first |
| `OLLAMA_TEMPERATURE` | `0.2` | Sampling temperature for Ollama |
| `OPENAI_API_KEY` | — | Required only when `LLM_PROVIDER=openai` or `EMBEDDINGS_PROVIDER=openai` |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `OPENAI_EMBEDDING_DIMENSIONS` | `1536` | Must match the model's native output dimension |
| `OPENAI_MAX_TOKENS` | `1024` | Max completion tokens |
| `OPENAI_TEMPERATURE` | `0.0` | OpenAI sampling temperature |
| **Qdrant** | | |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP endpoint |
| `QDRANT_API_KEY` | — | Optional; leave blank for unauthenticated Qdrant |
| `QDRANT_COLLECTION_NAME` | `rag_chunks` | All tenants share this collection |
| **RAG** | | |
| `CHUNK_SIZE` | `512` | Target chunk size in tokens (tiktoken cl100k\_base) |
| `CHUNK_OVERLAP` | `64` | Token overlap between consecutive chunks |
| `RETRIEVAL_TOP_K` | `5` | Default number of chunks returned by `/search` and `/chat` |
| `SIMILARITY_THRESHOLD` | `0.30` | Minimum cosine score; set `0.0` to disable |
| `EMBEDDING_BATCH_SIZE` | `64` | Texts embedded per model/API call |
| `EMBEDDING_CACHE_MAX_SIZE` | `10000` | In-process LRU cache capacity |
| **Service** | | |
| `FEEDBACK_LOG_PATH` | `data/feedback.jsonl` | Feedback append log |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Evaluation CLI

Measures retrieval quality against a golden dataset (no LLM call required).

**Golden dataset** (`data/eval/golden.jsonl`):

```jsonl
{"query": "What is the refund policy?", "relevant_chunk_ids": ["demo_3a9f12b84c01_0000"], "relevant_doc_ids": ["demo_3a9f12b84c01"]}
{"query": "How do I reset my password?", "relevant_doc_ids": ["demo_7e2a4108cd99"]}
```

`relevant_chunk_ids` takes precedence when present; `relevant_doc_ids` is used as fallback.

```bash
python -m app.eval.run \
  --tenant demo \
  --dataset data/eval/golden.jsonl \
  --k 3 5 10 \
  --output-dir reports
```

Reports are saved to `reports/eval_{tenant}_{unix_timestamp}.json`.

---

## Troubleshooting

**Qdrant vector dimension mismatch**
`Collection was created with dimension X, got Y` — this happens when you switch embedding providers after a collection already exists.

```bash
# Option A: use a fresh collection name
echo "QDRANT_COLLECTION_NAME=rag_chunks_v2" >> .env

# Option B: wipe the volume and restart
docker compose down -v && docker compose up -d
```

**Port already in use**
Edit the host-side port mapping in `docker-compose.yml` (e.g. `"8020:8000"`) and restart.

**Ollama model not found / LLM unavailable**
If `/chat` returns `"LLM unavailable — here are the most relevant sources found."` in the token stream:
- The Ollama model was not pulled yet: `docker exec -it rag-service-ollama-1 ollama pull qwen2.5:7b-instruct`
- Or the Ollama container is still starting; wait a few seconds and retry.
- The `citations` event in the response still contains the retrieved sources.

**OpenAI rate limit or auth error**
Same graceful fallback as above — the response stream returns sources even when the LLM call fails. Fix the API key in `.env` and restart the API container.

---

## Architecture

The service is stateless (no session storage) and multi-tenant. All tenants share a single Qdrant collection; isolation is enforced by a mandatory `tenant_id` payload filter on every query. Documents are chunked with tiktoken (`cl100k_base`) and a markdown-header-aware splitter so each chunk retains its section heading. The embedding cache (`LRU`, per-process) avoids redundant model/API calls on re-ingest. The LLM layer is abstracted behind `LLMProvider`; adding a new backend means subclassing it and registering it in `get_llm_provider()`.

**Ingest pipeline:**
1. Receive file bytes → compute `doc_id = {tenant_id}_{sha1[:12]}`
2. Chunk text (token-window + markdown sections)
3. Embed chunks (local sentence-transformers or OpenAI)
4. Dedup by `(chunk_hash, tenant_id)` → upsert new chunks to Qdrant

**Query pipeline:**
1. Embed query → `query_points()` against Qdrant with tenant filter
2. Format retrieved chunks as anonymous `[S1] … [S2] …` sources
3. Stream LLM response token-by-token (buffered into word/phrase frames)
4. Emit `citations` event with full metadata → emit `[DONE]`

---

## Feedback

```bash
curl -X POST http://localhost:8010/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "query": "What is the refund policy?",
    "rating": 5,
    "comment": "Correct and concise."
  }'
# HTTP 204 No Content
```

Feedback is appended to `FEEDBACK_LOG_PATH` (default `data/feedback.jsonl`).

---

## License

No LICENSE file is present in this repository. Add one before making it public (e.g. `MIT`, `Apache-2.0`).
