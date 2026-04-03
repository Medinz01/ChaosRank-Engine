"""API key authentication for the ChaosRank Engine.
Validates keys against the CHAOSRANK_API_KEYS environment variable.
"""
from __future__ import annotations

import os
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

_API_KEY_HEADER = APIKeyHeader(name="X-ChaosRank-Key", auto_error=False)


def _load_keys() -> set[str]:
    raw = os.environ.get("CHAOSRANK_API_KEYS", "")
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    if not keys:
        import logging

        logging.getLogger(__name__).warning(
            "CHAOSRANK_API_KEYS is not set — engine is running without auth. "
            "Set this env var in production."
        )
    return keys


_VALID_KEYS: set[str] = _load_keys()


async def require_api_key(api_key: str | None = Security(_API_KEY_HEADER)) -> str:
    """FastAPI dependency — raises 401 if key is missing or invalid."""
    if not _VALID_KEYS:
        return "dev"
    if not api_key or api_key not in _VALID_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Set X-ChaosRank-Key header.",
        )
    return api_key
