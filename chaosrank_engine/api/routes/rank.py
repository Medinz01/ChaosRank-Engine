"""Core risk ranking endpoint.
Accepts graph data and incident history to return prioritized services.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import networkx as nx
from fastapi import APIRouter, Depends, HTTPException, status

from chaosrank_engine.api.auth import require_api_key
from chaosrank_engine.api.models import (
    RankRequest,
    RankResponse,
    RankMetadata,
    RankedService,
)
from chaosrank_engine.graph.blast_radius import compute_blast_radius
from chaosrank_engine.parser.incidents import Incident, ServiceIncidents
from chaosrank_engine.scorer.ranker import rank_services

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_graph(req: RankRequest) -> nx.DiGraph:
    """Reconstruct a NetworkX graph from the API request payload."""
    G = nx.DiGraph()
    for e in req.graph.edges:
        G.add_edge(
            e.source,
            e.target,
            weight=e.weight,
            edge_type=e.edge_type,
            channel=e.channel,
            topic=e.topic,
        )
    return G


def _build_incidents(req: RankRequest) -> dict[str, ServiceIncidents]:
    """Parse incident payloads into the internal ServiceIncidents model."""
    result: dict[str, ServiceIncidents] = {}
    for svc, payloads in req.incidents.items():
        si = ServiceIncidents(service=svc)
        for p in payloads:
            si.incidents.append(
                Incident(
                    timestamp=datetime.fromisoformat(p.timestamp),
                    service=svc,
                    type=p.type,
                    severity=p.severity,
                    request_volume=p.request_volume,
                )
            )
        result[svc] = si
    return result


@router.post("/rank", response_model=RankResponse)
async def rank(
    req: RankRequest,
    _: str = Depends(require_api_key),
) -> RankResponse:
    """Compute risk scores and ranking for the provided system state."""
    cfg = req.config

    if abs(cfg.alpha + cfg.beta - 1.0) > 1e-6:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"config.alpha + config.beta must equal 1.0, got {cfg.alpha + cfg.beta:.6f}",
        )

    G = _build_graph(req)
    if G.number_of_nodes() == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="graph.edges is empty — cannot score an empty graph.",
        )

    incidents = _build_incidents(req)

    try:
        blast = compute_blast_radius(
            G,
            w_pr=cfg.w_pr,
            w_od=cfg.w_od,
            use_betweenness=cfg.use_betweenness,
            w_bc=cfg.w_bc,
            async_weight_factor=cfg.async_weight_factor,
            async_deps_provided=cfg.async_deps_provided,
        )
    except Exception as exc:
        logger.exception("blast_radius computation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        ranked_raw = rank_services(
            blast_radius=blast,
            service_incidents=incidents,
            alpha=cfg.alpha,
            beta=cfg.beta,
            decay_lambda=cfg.decay_lambda,
            base_window=cfg.base_window,
            severity_weights=cfg.severity_weights,
        )
    except Exception as exc:
        logger.exception("rank_services failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if cfg.top_n > 0:
        ranked_raw = ranked_raw[: cfg.top_n]

    ranked = [RankedService(**r) for r in ranked_raw]

    return RankResponse(
        ranked=ranked,
        metadata=RankMetadata(
            graph_nodes=G.number_of_nodes(),
            graph_edges=G.number_of_edges(),
            scored_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
