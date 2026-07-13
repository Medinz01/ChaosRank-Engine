"""FastAPI application for the ChaosRank Engine.
Assembles and mounts all core logic routes under the /v1 prefix.
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from chaosrank_engine.api.routes import rank, adaptive, orchestration, federation, health, webhooks

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

_ENV = os.environ.get("ENV", "development")
_is_prod = _ENV == "production"

# Maximum request body size (10 MB). Prevents memory exhaustion from oversized payloads.
_MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", 10 * 1024 * 1024))



app = FastAPI(
    title="ChaosRank Engine",
    description=(
        "Private risk-scoring engine for ChaosRank. "
        "Accepts serialized dependency graphs and incident history; "
        "returns ranked services with chaos experiment recommendations."
    ),
    version="0.1.0",
    docs_url=None if _is_prod else "/docs",
    redoc_url=None if _is_prod else "/redoc",
)



# CORS: deny-by-default. Set CORS_ORIGINS env var to a comma-separated allowlist.
_cors_origins_raw = os.environ.get("CORS_ORIGINS", "http://localhost:5173" if not _is_prod else "")
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type"],
    )


@app.middleware("http")
async def _limit_request_body(request: Request, call_next):
    """Reject requests with bodies exceeding MAX_BODY_BYTES."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_BODY_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request body too large. Maximum: {_MAX_BODY_BYTES} bytes."},
        )
    return await call_next(request)



# Mount all routers under /v1
for _router in (
    rank.router,
    adaptive.router,
    orchestration.router,
    federation.router,
    health.router,
    webhooks.router,
):
    app.include_router(_router, prefix="/v1")

# AWS Lambda entry point — Mangum wraps the ASGI app for API Gateway events
from mangum import Mangum  # noqa: E402
handler = Mangum(app, lifespan="off")


def run() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "chaosrank_engine.api.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    run()

