from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Service ────────────────────────────────────────────────────────────────
    SERVICE_NAME: str = "rag-service"
    SERVICE_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    # ── OpenAI ─────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    # Native output dimension – only used when EMBEDDINGS_PROVIDER=openai.
    # text-embedding-3-small: 1536  |  text-embedding-3-large: 3072
    OPENAI_EMBEDDING_DIMENSIONS: int = 1536
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_TOKENS: int = 1024
    OPENAI_TEMPERATURE: float = 0.0

    # ── LLM provider ──────────────────────────────────────────────────────────
    # "openai"   – OpenAI Chat API, requires OPENAI_API_KEY (default).
    # "ollama"   – Local Ollama server, no API key required.
    # "anthropic"– Placeholder; implement AnthropicProvider when needed.
    LLM_PROVIDER: str = "openai"

    # ── Ollama ─────────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct"
    OLLAMA_TEMPERATURE: float = 0.2

    # ── Embeddings provider ────────────────────────────────────────────────────
    # "local"  – sentence-transformers on CPU, no API key required (default).
    # "openai" – OpenAI Embeddings API, requires OPENAI_API_KEY.
    EMBEDDINGS_PROVIDER: str = "local"
    # HuggingFace model ID used when EMBEDDINGS_PROVIDER=local.
    # all-MiniLM-L6-v2 → 384-dim, ~90 MB download on first run.
    EMBEDDINGS_LOCAL_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Qdrant ─────────────────────────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "rag_chunks"

    # ── RAG ────────────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512       # target tokens per chunk
    CHUNK_OVERLAP: int = 64     # overlap tokens between consecutive chunks
    RETRIEVAL_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.30
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_CACHE_MAX_SIZE: int = 10_000

    # ── Feedback ───────────────────────────────────────────────────────────────
    FEEDBACK_LOG_PATH: str = "data/feedback.jsonl"


settings = Settings()
