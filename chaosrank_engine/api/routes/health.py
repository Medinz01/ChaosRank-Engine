"""Engine health check and status reporting.
Provides a simple endpoint for monitoring service availability and version
information.
"""
from __future__ import annotations
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "engine": "chaosrank-engine", "version": "0.1.0"}
