"""Solace FastAPI application entry point.

Layer 0: health endpoint only.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import FastAPI

from app.config import settings

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)
settings.log_configuration()

app = FastAPI(
    title="Solace",
    description="Session-based multi-agent mental-health counselling system.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    """Return service health status.

    Returns:
        JSON with ``status`` key set to ``"ok"`` and current UTC timestamp.
    """
    return {
        "status": "ok",
        "environment": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
