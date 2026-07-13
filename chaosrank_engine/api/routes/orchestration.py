"""API routes for multi-agent graph orchestration.
Provides endpoints for consolidating local graph snapshots from regional agents
into a unified canonical dependency model.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from chaosrank_engine.api.models import (
    MergeRequest,
    MergeResponse,
    EdgePayload,
    GraphPayload,
)
from chaosrank_engine.orchestration.agent import EdgeObservation, LocalGraphSnapshot
from chaosrank_engine.orchestration.merger import CentralMerger

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/orchestration/merge", response_model=MergeResponse)
async def merge_snapshots(
    request: Request,
    req: MergeRequest,
) -> MergeResponse:
    """Consolidate multiple agent snapshots into a unified canonical graph."""
    if not req.snapshots:
        raise HTTPException(status_code=422, detail="No snapshots provided.")

    merger = CentralMerger(
        min_agents=1,
        single_agent_warn=True,
        min_call_frequency=req.min_call_frequency,
    )

    for snap in req.snapshots:
        observations = [
            EdgeObservation(
                source=e.source,
                target=e.target,
                weight=e.weight,
                confidence=e.confidence,
                edge_type=e.edge_type,
                channel=e.channel,
                topic=e.topic,
            )
            for e in snap.edges
        ]
        snapshot = LocalGraphSnapshot(
            agent_id=snap.agent_id,
            observed_at=datetime.fromisoformat(snap.observed_at),
            total_spans=snap.total_spans,
            edges=observations,
            scope_metadata=snap.scope_metadata,
        )
        merger.ingest(snapshot)

    try:
        result = merger.merge()
    except Exception as exc:
        logger.exception("Merge failed")
        raise HTTPException(status_code=500, detail="Internal merge error. See server logs.") from exc

    # Serialize canonical graph back to EdgePayload list
    canonical_edges = [
        EdgePayload(
            source=e.source,
            target=e.target,
            weight=e.canonical_weight,
            edge_type=e.edge_type,
            channel=e.channel,
            topic=e.topic,
        )
        for e in result.canonical_edges
    ]

    return MergeResponse(
        graph=GraphPayload(edges=canonical_edges),
        agent_count=result.agent_count,
        edge_count=len(canonical_edges),
        single_agent_edges=result.single_agent_edge_count,
        warnings=result.warnings,
        merged_at=datetime.now(timezone.utc).isoformat(),
    )
