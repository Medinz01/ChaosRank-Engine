"""API routes for multi-domain, federated risk ranking.
Provides endpoints for consolidating independent domain graphs into a unified
dependency model for cross-domain scoring.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import networkx as nx
from fastapi import APIRouter, Depends, HTTPException

from chaosrank_engine.api.auth import require_api_key
from chaosrank_engine.api.models import (
    FederationRankRequest,
    FederationRankResponse,
    RankMetadata,
    RankedService,
)
from chaosrank_engine.graph.blast_radius import compute_blast_radius
from chaosrank_engine.parser.incidents import Incident, ServiceIncidents
from chaosrank_engine.scorer.ranker import rank_services

logger = logging.getLogger(__name__)
router = APIRouter()


def _qualify(component_id: str, domain_id: str) -> str:
    """Produce a domain-qualified node ID: '{domain_id}/{component_id}'."""
    return f"{domain_id}/{component_id}"


@router.post("/federation/rank", response_model=FederationRankResponse)
async def federation_rank(
    req: FederationRankRequest,
    _: str = Depends(require_api_key),
) -> FederationRankResponse:
    """Merge multiple domain graphs and perform federated risk ranking."""
    if not req.domains:
        raise HTTPException(status_code=422, detail="No domains provided.")

    cfg = req.config
    warnings: list[str] = []
    G_fed = nx.DiGraph()
    domain_node_map: dict[str, list[str]] = {}
    service_incidents: dict[str, ServiceIncidents] = {}

    # Build per-domain sub-graphs and merge
    for domain in req.domains:
        domain_nodes: list[str] = []
        for e in domain.edges:
            qsrc = _qualify(e.source, domain.domain_id)
            qtgt = _qualify(e.target, domain.domain_id)
            G_fed.add_node(qsrc, domain_id=domain.domain_id)
            G_fed.add_node(qtgt, domain_id=domain.domain_id)
            G_fed.add_edge(
                qsrc, qtgt, weight=e.weight, edge_type=e.edge_type, channel=e.channel, topic=e.topic
            )
            if qsrc not in domain_nodes:
                domain_nodes.append(qsrc)
            if qtgt not in domain_nodes:
                domain_nodes.append(qtgt)
        domain_node_map[domain.domain_id] = domain_nodes

        # Collect incidents
        for di in domain.incidents:
            qid = _qualify(di.component_id, domain.domain_id)
            if qid not in service_incidents:
                service_incidents[qid] = ServiceIncidents(service=qid)
            service_incidents[qid].incidents.append(
                Incident(
                    timestamp=datetime.fromisoformat(di.timestamp),
                    service=qid,
                    type=di.type,
                    severity=di.severity,
                    request_volume=di.request_volume,
                )
            )

    # Add inter-domain edges
    for ide in req.inter_domain_edges:
        src = _qualify(ide.source_component, ide.source_domain)
        tgt = _qualify(ide.target_component, ide.target_domain)
        if src not in G_fed:
            warnings.append(f"Inter-domain edge skipped: '{src}' not in graph")
            continue
        if tgt not in G_fed:
            warnings.append(f"Inter-domain edge skipped: '{tgt}' not in graph")
            continue
        G_fed.add_edge(
            src,
            tgt,
            weight=ide.weight,
            edge_type=ide.edge_type,
            channel=ide.channel,
            topic=ide.topic,
            inter_domain=True,
        )

    if G_fed.number_of_nodes() == 0:
        raise HTTPException(status_code=422, detail="Federated graph is empty.")

    try:
        blast = compute_blast_radius(
            G_fed,
            w_pr=cfg.w_pr,
            w_od=cfg.w_od,
            async_weight_factor=cfg.async_weight_factor,
        )
        ranked_raw = rank_services(
            blast_radius=blast,
            service_incidents=service_incidents,
            alpha=cfg.alpha,
            beta=cfg.beta,
            decay_lambda=cfg.decay_lambda,
        )
    except Exception as exc:
        logger.exception("Federation rank failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if cfg.top_n > 0:
        ranked_raw = ranked_raw[: cfg.top_n]

    return FederationRankResponse(
        ranked=[RankedService(**r) for r in ranked_raw],
        domain_node_map=domain_node_map,
        warnings=warnings,
        metadata=RankMetadata(
            graph_nodes=G_fed.number_of_nodes(),
            graph_edges=G_fed.number_of_edges(),
            scored_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
