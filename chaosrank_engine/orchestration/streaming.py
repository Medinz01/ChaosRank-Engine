"""Delta-triggered partial centrality recomputation for ChaosRank.
Optimizes scoring performance by tracking neighborhood changes and only
recomputing blast radius scores for affected nodes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import networkx as nx

from chaosrank_engine.graph.blast_radius import compute_blast_radius
from chaosrank_engine.orchestration.incremental import IncrementalUpdateResult

logger = logging.getLogger(__name__)

# Fraction of edges changed that triggers a full recomputation
FULL_RECOMPUTE_THRESHOLD = 0.30

# Minimum absolute weight delta to consider an edge "significantly changed"
MIN_DELTA_FOR_TRIGGER = 5.0


class RecomputeStrategy(str, Enum):
    DELTA = "delta"  # recompute only if delta threshold exceeded
    NEIGHBORHOOD = "neighborhood"  # recompute only affected node neighborhoods
    FULL = "full"  # recompute all nodes


@dataclass
class StreamingRankResult:
    """Container for the results of an incremental scoring update."""

    scores: dict[str, float]
    updated_nodes: list[str]
    strategy_used: RecomputeStrategy
    full_graph_size: int
    recomputed_at: datetime = field(default_factory=datetime.utcnow)
    change_magnitude: float = 0.0


class StreamingScorer:
    """Maintains an incrementally updated cache of blast radius scores."""

    def __init__(
        self,
        w_pr: float = 0.5,
        w_od: float = 0.5,
        async_weight_factor: float = 0.5,
        full_recompute_threshold: float = FULL_RECOMPUTE_THRESHOLD,
        min_delta: float = MIN_DELTA_FOR_TRIGGER,
    ) -> None:
        self.w_pr = w_pr
        self.w_od = w_od
        self.async_weight_factor = async_weight_factor
        self.full_recompute_threshold = full_recompute_threshold
        self.min_delta = min_delta

        self._cached_scores: dict[str, float] = {}
        self._cached_graph: nx.DiGraph | None = None

    # Update API

    def update(
        self,
        G: nx.DiGraph,
        update: IncrementalUpdateResult,
    ) -> StreamingRankResult:
        """
        Update blast radius scores based on incremental graph changes.

        Auto-selects recompute strategy based on change magnitude:
          - Large changes (>30% edges) → FULL recomputation
          - Topology changes (edges added/pruned) → NEIGHBORHOOD recomputation
          - Weight-only changes → DELTA-triggered or skip

        Parameters
        ----------
        G:      Current graph from IncrementalGraphState.apply()
        update: IncrementalUpdateResult describing what changed

        Returns
        -------
        StreamingRankResult with current scores
        """
        if G.number_of_nodes() == 0:
            self._cached_scores = {}
            self._cached_graph = G
            return StreamingRankResult(
                scores={},
                updated_nodes=[],
                strategy_used=RecomputeStrategy.FULL,
                full_graph_size=0,
            )

        total_edges = G.number_of_edges()
        topology_changes = len(update.edges_added) + len(update.edges_pruned)
        change_magnitude = topology_changes / max(total_edges, 1)

        strategy = self._select_strategy(G, update, change_magnitude)

        if strategy == RecomputeStrategy.FULL:
            return self._full_recompute(G, change_magnitude)

        elif strategy == RecomputeStrategy.NEIGHBORHOOD:
            return self._neighborhood_recompute(G, update, change_magnitude)

        else:  # DELTA
            return self._delta_recompute(G, update, change_magnitude)

    def force_full_recompute(self, G: nx.DiGraph) -> StreamingRankResult:
        """Force a full recomputation regardless of change magnitude."""
        return self._full_recompute(G, change_magnitude=1.0)

    @property
    def cached_scores(self) -> dict[str, float]:
        """Current cached blast radius scores."""
        return dict(self._cached_scores)

    # Strategy selection

    def _select_strategy(
        self,
        G: nx.DiGraph,
        update: IncrementalUpdateResult,
        change_magnitude: float,
    ) -> RecomputeStrategy:
        """Auto-select recompute strategy based on change characteristics."""

        # No cached scores yet — must do full
        if not self._cached_scores:
            return RecomputeStrategy.FULL

        # Large topology change — full recomputation
        if change_magnitude >= self.full_recompute_threshold:
            logger.debug(
                "Strategy: FULL (change_magnitude=%.2f >= threshold=%.2f)",
                change_magnitude,
                self.full_recompute_threshold,
            )
            return RecomputeStrategy.FULL

        # Any topology changes (new nodes/edges or pruned) → neighborhood
        if update.edges_added or update.edges_pruned:
            logger.debug(
                "Strategy: NEIGHBORHOOD (+%d added, -%d pruned)",
                len(update.edges_added),
                len(update.edges_pruned),
            )
            return RecomputeStrategy.NEIGHBORHOOD

        # Weight-only changes → delta
        logger.debug(
            "Strategy: DELTA (%d edges updated, %d attenuated)",
            len(update.edges_updated),
            len(update.edges_attenuated),
        )
        return RecomputeStrategy.DELTA

    # Recompute implementations

    def _full_recompute(
        self,
        G: nx.DiGraph,
        change_magnitude: float,
    ) -> StreamingRankResult:
        """Full blast radius recomputation using compute_blast_radius()."""
        scores = compute_blast_radius(
            G,
            w_pr=self.w_pr,
            w_od=self.w_od,
            async_weight_factor=self.async_weight_factor,
        )
        self._cached_scores = scores
        self._cached_graph = G

        logger.info(
            "StreamingScorer: FULL recompute — %d nodes scored",
            len(scores),
        )
        return StreamingRankResult(
            scores=scores,
            updated_nodes=list(scores.keys()),
            strategy_used=RecomputeStrategy.FULL,
            full_graph_size=G.number_of_nodes(),
            change_magnitude=change_magnitude,
        )

    def _neighborhood_recompute(
        self,
        G: nx.DiGraph,
        update: IncrementalUpdateResult,
        change_magnitude: float,
    ) -> StreamingRankResult:
        """
        Recompute scores only for nodes whose neighborhoods changed.

        Affected nodes: nodes that are source or target of any added or
        pruned edge, plus their immediate in-degree neighbors (since their
        PageRank flows are affected).

        For small graphs or heavily connected topologies, this may approach
        a full recompute. In those cases, fall back to full.
        """
        affected = self._affected_nodes(G, update)

        # If affected set is large relative to total, just do full
        if len(affected) / max(G.number_of_nodes(), 1) > 0.5:
            return self._full_recompute(G, change_magnitude)

        # Recompute full scores (PageRank is global so we can't do partial)
        # but only update cached scores for affected nodes
        full_scores = compute_blast_radius(
            G,
            w_pr=self.w_pr,
            w_od=self.w_od,
            async_weight_factor=self.async_weight_factor,
        )

        updated_nodes = []
        for node in affected:
            if node in full_scores:
                self._cached_scores[node] = full_scores[node]
                updated_nodes.append(node)

        # Remove pruned nodes from cache
        for source, target in update.edges_pruned:
            for node in (source, target):
                if node not in G and node in self._cached_scores:
                    del self._cached_scores[node]

        self._cached_graph = G

        logger.info(
            "StreamingScorer: NEIGHBORHOOD recompute — %d/%d nodes updated",
            len(updated_nodes),
            G.number_of_nodes(),
        )
        return StreamingRankResult(
            scores=dict(self._cached_scores),
            updated_nodes=updated_nodes,
            strategy_used=RecomputeStrategy.NEIGHBORHOOD,
            full_graph_size=G.number_of_nodes(),
            change_magnitude=change_magnitude,
        )

    def _delta_recompute(
        self,
        G: nx.DiGraph,
        update: IncrementalUpdateResult,
        change_magnitude: float,
    ) -> StreamingRankResult:
        """
        Weight-only changes: recompute only if delta exceeds threshold.

        If total weight change is minor (traffic fluctuation within EMA),
        return cached scores unchanged to avoid unnecessary computation.
        """
        if not self._cached_graph:
            return self._full_recompute(G, change_magnitude)

        # Compute total absolute weight delta across updated edges
        total_delta = 0.0
        for source, target in update.edges_updated:
            if self._cached_graph.has_edge(source, target) and G.has_edge(source, target):
                old_w = self._cached_graph[source][target].get("weight", 0)
                new_w = G[source][target].get("weight", 0)
                total_delta += abs(new_w - old_w)

        if total_delta < self.min_delta:
            logger.debug(
                "StreamingScorer: DELTA skip — total_delta=%.2f < threshold=%.2f",
                total_delta,
                self.min_delta,
            )
            return StreamingRankResult(
                scores=dict(self._cached_scores),
                updated_nodes=[],
                strategy_used=RecomputeStrategy.DELTA,
                full_graph_size=G.number_of_nodes(),
                change_magnitude=change_magnitude,
            )

        # Delta is significant — do a full recompute
        return self._full_recompute(G, change_magnitude)

    # Internal helpers

    def _affected_nodes(
        self,
        G: nx.DiGraph,
        update: IncrementalUpdateResult,
    ) -> set[str]:
        """
        Compute the set of nodes affected by topology changes.

        Includes: nodes on added/pruned edges + their in-degree neighbors.
        """
        directly_affected: set[str] = set()
        for source, target in update.edges_added + update.edges_pruned:
            directly_affected.add(source)
            directly_affected.add(target)

        # Also include in-degree neighbors (their PageRank flows changed)
        neighbors: set[str] = set()
        for node in directly_affected:
            if node in G:
                for pred in G.predecessors(node):
                    neighbors.add(pred)

        return directly_affected | neighbors
