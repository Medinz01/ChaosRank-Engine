"""API routes for Reinforcement Learning-based adaptive ranking.
Provides endpoints for scoring with dynamic weights and recording experiment
outcomes to calibrate the ranking model.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import networkx as nx
from fastapi import APIRouter, Depends, HTTPException

from chaosrank_engine.api.auth import require_api_key
from chaosrank_engine.api.models import (
    AdaptiveRankRequest,
    AdaptiveRankResponse,
    AdaptiveRankedService,
    OutcomeRequest,
    OutcomeResponse,
    RankMetadata,
)
from chaosrank_engine.adaptive.adaptive_ranker import AdaptiveRanker
from chaosrank_engine.adaptive.outcome_store import OutcomeType
from chaosrank_engine.parser.incidents import Incident, ServiceIncidents

logger = logging.getLogger(__name__)
router = APIRouter()

# Global AdaptiveRanker instance — state persists across requests.
# In a multi-worker deployment, use a shared Redis-backed store instead.
_RANKER = AdaptiveRanker()


def _build_graph(edges: list) -> nx.DiGraph:
    """Helper to convert API edge specs into a NetworkX graph."""
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(
            e.source,
            e.target,
            weight=e.weight,
            edge_type=e.edge_type,
            channel=e.channel,
            topic=e.topic,
        )
    return G


def _build_incidents(incidents_map: dict) -> dict[str, ServiceIncidents]:
    """Helper to convert API incident payloads into ServiceIncidents models."""
    result = {}
    for svc, payloads in incidents_map.items():
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


@router.post("/adaptive/rank", response_model=AdaptiveRankResponse)
async def adaptive_rank(
    req: AdaptiveRankRequest,
    _: str = Depends(require_api_key),
) -> AdaptiveRankResponse:
    """Perform risk ranking using live alpha/beta weights and adaptive logic."""
    G = _build_graph(req.graph.edges)
    if G.number_of_nodes() == 0:
        raise HTTPException(status_code=422, detail="Empty graph.")

    incidents = _build_incidents(req.incidents)
    last_observed = None
    if req.last_observed:
        last_observed = datetime.fromisoformat(req.last_observed)

    try:
        ranked_raw = _RANKER.rank(
            blast_radius=__import__(
                "chaosrank_engine.graph.blast_radius",
                fromlist=["compute_blast_radius"],
            ).compute_blast_radius(
                G,
                w_pr=req.config.w_pr,
                w_od=req.config.w_od,
                async_weight_factor=req.config.async_weight_factor,
            ),
            service_incidents=incidents,
            G=G,
            decay_lambda=req.config.decay_lambda,
            base_window=req.config.base_window,
            last_observed=last_observed,
        )
    except Exception as exc:
        logger.exception("adaptive rank failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if req.config.top_n > 0:
        ranked_raw = ranked_raw[: req.config.top_n]

    ranked = [AdaptiveRankedService(**r) for r in ranked_raw]
    state = _RANKER.updater.state()

    return AdaptiveRankResponse(
        ranked=ranked,
        alpha=state.alpha,
        beta=state.beta,
        update_count=state.update_count,
        metadata=RankMetadata(
            graph_nodes=G.number_of_nodes(),
            graph_edges=G.number_of_edges(),
            scored_at=datetime.now(timezone.utc).isoformat(),
        ),
    )


@router.post("/adaptive/outcome", response_model=OutcomeResponse)
async def record_outcome(
    req: OutcomeRequest,
    _: str = Depends(require_api_key),
) -> OutcomeResponse:
    """Record an experiment outcome and update the global adaptive weights."""
    try:
        outcome = OutcomeType(req.outcome)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid outcome: {req.outcome!r}. "
            f"Valid values: WEAKNESS_CONFIRMED, WEAKNESS_NOT_FOUND, INCONCLUSIVE",
        )

    ranked_row = {
        "service": req.service,
        "risk": req.risk_score,
        "blast_radius": req.blast_radius,
        "fragility": req.fragility,
        "rank": req.rank_at_time,
        "alpha_used": req.alpha_used,
        "beta_used": req.beta_used,
    }

    try:
        _RANKER.record_outcome(
            ranked_row=ranked_row,
            outcome=outcome,
            graph_state_hash=req.graph_state_hash,
            notes=req.notes,
        )
    except Exception as exc:
        logger.exception("record_outcome failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return OutcomeResponse(
        new_alpha=_RANKER.alpha,
        new_beta=_RANKER.beta,
        message=f"Outcome recorded. Weights updated: α={_RANKER.alpha:.4f} β={_RANKER.beta:.4f}",
    )


@router.get("/adaptive/summary")
async def adaptive_summary(_: str = Depends(require_api_key)) -> dict:
    return _RANKER.summary()
