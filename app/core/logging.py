"""Structured JSON logging with per-request context via ContextVars."""
from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar

# These vars are set per-request by the observability middleware.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="")


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    # Fields already represented at the top level – skip in extras.
    _SKIP = frozenset(
        {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "id", "levelname", "levelno", "lineno", "module",
            "msecs", "message", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
        ms = int(record.msecs)
        doc: dict = {
            "ts": f"{ts}.{ms:03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(""),
            "tenant_id": tenant_id_var.get(""),
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        # Attach any extra= kwargs the caller provided
        for key, value in record.__dict__.items():
            if key not in self._SKIP and not key.startswith("_"):
                doc[key] = value
        return json.dumps(doc, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Quiet down noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
