from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get("/metrics")
async def metrics() -> Response:
    """Expose all registered Prometheus metrics in text/plain exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
