"""API request/response schemas for the ChaosRank Engine.
Defines the public contract between the private engine and the external SDK/CLI.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# Shared primitives


class EdgePayload(BaseModel):
    source: str
    target: str
    weight: float = Field(default=1.0, ge=0)
    edge_type: str = "sync"  # sync | async
    channel: str | None = None
    topic: str | None = None


class IncidentPayload(BaseModel):
    timestamp: str  # ISO 8601
    type: str  # latency | error | timeout
    severity: str  # critical | high | medium | low
    request_volume: float | None = None


class GraphPayload(BaseModel):
    edges: list[EdgePayload]


# /v1/rank


class RankConfig(BaseModel):
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    beta: float = Field(default=0.4, ge=0.0, le=1.0)
    decay_lambda: float = Field(default=0.10, gt=0)
    base_window: float = Field(default=5.0, gt=0)
    use_betweenness: bool = False
    w_pr: float = Field(default=0.5, ge=0)
    w_od: float = Field(default=0.5, ge=0)
    w_bc: float | None = None
    async_weight_factor: float = Field(default=0.5, gt=0, le=1.0)
    async_deps_provided: bool = False
    severity_weights: dict[str, float] | None = None
    top_n: int = Field(default=0, ge=0)  # 0 = return all


class RankRequest(BaseModel):
    graph: GraphPayload
    incidents: dict[str, list[IncidentPayload]] = Field(default_factory=dict)
    config: RankConfig = Field(default_factory=RankConfig)


class RankedService(BaseModel):
    rank: int
    service: str
    risk: float
    blast_radius: float
    fragility: float
    suggested_fault: str
    confidence: str


class RankMetadata(BaseModel):
    graph_nodes: int
    graph_edges: int
    scored_at: str
    engine_version: str = "0.1.0"


class RankResponse(BaseModel):
    ranked: list[RankedService]
    metadata: RankMetadata


# /v1/adaptive/rank


class AdaptiveRankRequest(BaseModel):
    graph: GraphPayload
    incidents: dict[str, list[IncidentPayload]] = Field(default_factory=dict)
    config: RankConfig = Field(default_factory=RankConfig)
    session_id: str | None = None  # for stateful outcome tracking
    last_observed: str | None = None  # ISO 8601; for CI age component


class AdaptiveRankedService(RankedService):
    alpha_used: float
    beta_used: float
    ci_lower: float | None
    ci_upper: float | None
    ci_width: float | None
    low_confidence: bool | None
    confidence_note: str


class AdaptiveRankResponse(BaseModel):
    ranked: list[AdaptiveRankedService]
    alpha: float
    beta: float
    update_count: int
    metadata: RankMetadata


# /v1/adaptive/outcome


class OutcomeRequest(BaseModel):
    session_id: str | None = None
    service: str
    outcome: str  # WEAKNESS_CONFIRMED | WEAKNESS_NOT_FOUND | INCONCLUSIVE
    risk_score: float
    blast_radius: float
    fragility: float
    alpha_used: float
    beta_used: float
    rank_at_time: int
    graph_state_hash: str | None = None
    notes: str | None = None


class OutcomeResponse(BaseModel):
    new_alpha: float
    new_beta: float
    message: str


# /v1/orchestration/merge


class EdgeObservationPayload(BaseModel):
    source: str
    target: str
    weight: float
    confidence: float
    edge_type: str = "sync"
    channel: str | None = None
    topic: str | None = None


class AgentSnapshotPayload(BaseModel):
    agent_id: str
    observed_at: str  # ISO 8601
    total_spans: int
    edges: list[EdgeObservationPayload]
    scope_metadata: dict[str, Any] = Field(default_factory=dict)


class MergeRequest(BaseModel):
    snapshots: list[AgentSnapshotPayload]
    min_call_frequency: int = 10


class MergeResponse(BaseModel):
    graph: GraphPayload  # canonical merged graph
    agent_count: int
    edge_count: int
    single_agent_edges: int
    warnings: list[str]
    merged_at: str


# /v1/federation/rank


class DomainIncidentPayload(BaseModel):
    component_id: str
    timestamp: str
    type: str
    severity: str
    request_volume: float | None = None


class DomainEdgePayload(BaseModel):
    source: str
    target: str
    weight: float = 1.0
    edge_type: str = "sync"
    channel: str | None = None
    topic: str | None = None


class DomainPayload(BaseModel):
    domain_id: str
    edges: list[DomainEdgePayload]
    incidents: list[DomainIncidentPayload] = Field(default_factory=list)
    components: list[dict[str, Any]] = Field(default_factory=list)


class InterDomainEdgePayload(BaseModel):
    source_domain: str
    source_component: str
    target_domain: str
    target_component: str
    weight: float = 1.0
    edge_type: str = "sync"
    channel: str | None = None
    topic: str | None = None


class FederationRankRequest(BaseModel):
    domains: list[DomainPayload]
    inter_domain_edges: list[InterDomainEdgePayload] = Field(default_factory=list)
    config: RankConfig = Field(default_factory=RankConfig)
    window_days: int = 30


class FederationRankResponse(BaseModel):
    ranked: list[RankedService]
    domain_node_map: dict[str, list[str]]
    warnings: list[str]
    metadata: RankMetadata


# Errors


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
