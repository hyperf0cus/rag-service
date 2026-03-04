import time

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()

_start_time = time.time()


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": settings.SERVICE_VERSION,
        "uptime": round(time.time() - _start_time, 2),
    }
