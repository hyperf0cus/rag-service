# ── Stage 1: install dependencies ─────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN pip install --no-cache-dir --upgrade pip

# CPU-only PyTorch avoids pulling the ~2 GB CUDA build (~250 MB instead).
RUN pip install --no-cache-dir \
        "torch>=2.0.0" \
        --index-url https://download.pytorch.org/whl/cpu

# Application dependencies.
# sentence-transformers is included for the default EMBEDDINGS_PROVIDER=local.
# Model weights (~90 MB for all-MiniLM-L6-v2) are fetched at first run;
# mount a volume at $HF_HOME to cache them across restarts.
RUN pip install --no-cache-dir \
        "fastapi>=0.111.0" \
        "uvicorn[standard]>=0.30.0" \
        "pydantic>=2.7.0" \
        "pydantic-settings>=2.3.0" \
        "openai>=1.40.0" \
        "qdrant-client>=1.10.0" \
        "prometheus-client>=0.20.0" \
        "python-multipart>=0.0.9" \
        "httpx>=0.27.0" \
        "tiktoken>=0.7.0" \
        "aiofiles>=23.2.1" \
        "rich>=13.7.0" \
        "sentence-transformers>=3.0.0"

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY app/ ./app/

RUN mkdir -p data/docs data/eval reports

EXPOSE 8000

# --no-access-log: uvicorn request logs suppressed; our middleware emits JSON logs.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
