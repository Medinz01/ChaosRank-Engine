"""FastAPI application for the ChaosRank Engine.
Assembles and mounts all core logic routes under the /v1 prefix.
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chaosrank_engine.api.routes import rank, adaptive, orchestration, federation, health

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

app = FastAPI(
    title="ChaosRank Engine",
    description=(
        "Private risk-scoring engine for ChaosRank. "
        "Accepts serialized dependency graphs and incident history; "
        "returns ranked services with chaos experiment recommendations."
    ),
    version="0.1.0",
    docs_url="/docs",  # disable in prod: set to None
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["X-ChaosRank-Key", "Content-Type"],
)

# Mount all routers under /v1
for _router in (
    rank.router,
    adaptive.router,
    orchestration.router,
    federation.router,
    health.router,
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

