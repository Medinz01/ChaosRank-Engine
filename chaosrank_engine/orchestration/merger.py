"""Weighted confidence merge protocol for federated graphs.
Consolidates local graph snapshots from multiple regional agents into a unified
canonical dependency model using confidence-weighted conflict resolution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import networkx as nx

from chaosrank_engine.orchestration.agent import EdgeObservation, LocalGraphSnapshot

logger = logging.getLogger(__name__)


@dataclass
class CanonicalEdge:
    """Represents a merged edge in the aggregated canonical graph."""

    source: str
    target: str
    canonical_weight: float
    contributing_agents: list[str]
    total_confidence: float
    single_agent: bool
    edge_type: str = "sync"
    channel: str | None = None
    topic: str | None = None


@dataclass
class MergeResult:
    """Container for the results of a single graph consolidation run."""

    graph: nx.DiGraph
    canonical_edges: list[CanonicalEdge]
    agent_count: int
    snapshot_times: dict[str, datetime]
    warnings: list[str] = field(default_factory=list)
    merged_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def single_agent_edge_count(self) -> int:
        return sum(1 for e in self.canonical_edges if e.single_agent)

    @property
    def corroborated_edge_count(self) -> int:
        return sum(1 for e in self.canonical_edges if not e.single_agent)


class CentralMerger:
    """Consolidates discrete agent snapshots into a unified dependency graph."""

    def __init__(
        self,
        min_agents: int = 1,
        single_agent_warn: bool = True,
        min_call_frequency: int = 10,
    ) -> None:
        self.min_agents = min_agents
        self.single_agent_warn = single_agent_warn
        self.min_call_frequency = min_call_frequency
        self._snapshots: dict[str, LocalGraphSnapshot] = {}

    # Snapshot management

    def ingest(self, snapshot: LocalGraphSnapshot) -> None:
        """
        Ingest a snapshot from a regional agent.

        If a snapshot from the same agent_id already exists, it is replaced
        (latest observation wins within a merge window).
        """
        self._snapshots[snapshot.agent_id] = snapshot
        logger.info(
            "Merger: ingested snapshot from agent '%s' (%d edges, %d spans, at %s)",
            snapshot.agent_id,
            len(snapshot.edges),
            snapshot.total_spans,
            snapshot.observed_at.isoformat(),
        )

    def clear(self) -> None:
        """Clear all ingested snapshots. Call between merge windows."""
        self._snapshots.clear()

    def ready(self) -> bool:
        """True if minimum agent count for a merge has been reached."""
        return len(self._snapshots) >= self.min_agents

    @property
    def agent_count(self) -> int:
        return len(self._snapshots)

    # Merge operations

    def merge(self) -> MergeResult:
        """
        Merge all ingested snapshots into a canonical graph.

        Returns
        -------
        MergeResult with canonical graph and full edge provenance
        """
        if not self._snapshots:
            raise ValueError("No snapshots ingested — cannot merge")

        warnings: list[str] = []
        edge_pool: dict[tuple[str, str], list[EdgeObservation]] = {}
        agent_ids: list[str] = []
        snapshot_times: dict[str, datetime] = {}

        # Collect all edge observations across agents, grouped by (source, target)
        for agent_id, snapshot in self._snapshots.items():
            agent_ids.append(agent_id)
            snapshot_times[agent_id] = snapshot.observed_at
            for obs in snapshot.edges:
                key = (obs.source, obs.target)
                edge_pool.setdefault(key, []).append(obs)

        # Apply weighted confidence merge protocol
        canonical_edges = []
        for (source, target), observations in edge_pool.items():
            canonical = self._merge_edge(source, target, observations, warnings)
            if canonical is None:
                continue
            canonical_edges.append(canonical)

        # Build nx.DiGraph from canonical edges
        G = nx.DiGraph()
        for edge in canonical_edges:
            G.add_edge(
                edge.source,
                edge.target,
                weight=edge.canonical_weight,
                edge_type=edge.edge_type,
                channel=edge.channel,
                topic=edge.topic,
                total_confidence=edge.total_confidence,
                contributing_agents=edge.contributing_agents,
                single_agent=edge.single_agent,
            )

        logger.info(
            "Merger: merged %d agents → %d nodes, %d edges (%d corroborated, %d single-agent)",
            len(agent_ids),
            G.number_of_nodes(),
            G.number_of_edges(),
            sum(1 for e in canonical_edges if not e.single_agent),
            sum(1 for e in canonical_edges if e.single_agent),
        )

        return MergeResult(
            graph=G,
            canonical_edges=canonical_edges,
            agent_count=len(agent_ids),
            snapshot_times=snapshot_times,
            warnings=warnings,
        )

    # Internal merge protocol

    def _merge_edge(
        self,
        source: str,
        target: str,
        observations: list[EdgeObservation],
        warnings: list[str],
    ) -> CanonicalEdge | None:
        """
        Apply weighted confidence merge to a set of observations for one edge.

        canonical_weight(u, v) =
          Σ [weight_i × confidence_i] / Σ confidence_i

        This is a confidence-weighted average. An agent observing 10,000 spans
        with 200 on this edge (confidence=0.02) has more influence than an agent
        observing 100 spans with 2 on this edge (confidence=0.02 — same ratio
        but lower absolute count). The absolute count is captured in weight_i.

        Returns None if the canonical weight falls below min_call_frequency.
        """
        total_confidence = sum(obs.confidence for obs in observations)
        [obs.source for obs in observations]

        if total_confidence <= 0:
            return None

        # Confidence-weighted average weight
        canonical_weight = (
            sum(obs.weight * obs.confidence for obs in observations) / total_confidence
        )

        # Apply min_call_frequency filter on canonical weight
        if canonical_weight < self.min_call_frequency:
            return None

        # Majority vote on edge_type (sync wins ties)
        async_count = sum(1 for obs in observations if obs.edge_type == "async")
        edge_type = "async" if async_count > len(observations) / 2 else "sync"

        # Channel and topic from first async observation
        channel = next((obs.channel for obs in observations if obs.channel), None)
        topic = next((obs.topic for obs in observations if obs.topic), None)

        single_agent = len(observations) == 1
        if single_agent and self.single_agent_warn:
            warnings.append(
                f"Edge ({source} → {target}) observed by only one agent "
                f"('{observations[0].source}'). Consider corroborating with "
                f"additional agents for higher confidence."
            )

        contributing = [obs.source for obs in observations]

        return CanonicalEdge(
            source=source,
            target=target,
            canonical_weight=round(canonical_weight, 4),
            contributing_agents=contributing,
            total_confidence=round(total_confidence, 6),
            single_agent=single_agent,
            edge_type=edge_type,
            channel=channel,
            topic=topic,
        )
