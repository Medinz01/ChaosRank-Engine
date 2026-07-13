"""Sliding window graph state with EMA weight maintenance and staleness attenuation.
Maintains a live, incrementally-updated dependency model as new observations
arrive from regional agents.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import networkx as nx

from chaosrank_engine.orchestration.merger import MergeResult

logger = logging.getLogger(__name__)

DEFAULT_EMA_ALPHA = 0.3

DEFAULT_STALENESS_FACTOR = 0.85

DEFAULT_MIN_WEIGHT = 1.0

MAX_STALE_DAYS = 30.0


@dataclass
class EdgeState:
    """Represents the live state for a single graph edge."""

    source: str
    target: str
    weight: float
    edge_type: str
    last_observed: datetime
    first_observed: datetime
    update_count: int = 1
    channel: str | None = None
    topic: str | None = None

    @property
    def age_days(self) -> float:
        return (datetime.now(timezone.utc) - self.first_observed).total_seconds() / 86400

    @property
    def stale_days(self) -> float:
        return (datetime.now(timezone.utc) - self.last_observed).total_seconds() / 86400


@dataclass
class IncrementalUpdateResult:
    """Container for the results of a single incremental state update."""

    graph: nx.DiGraph
    edges_added: list[tuple[str, str]]
    edges_updated: list[tuple[str, str]]
    edges_pruned: list[tuple[str, str]]
    edges_attenuated: list[tuple[str, str]]
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class IncrementalGraphState:
    """Maintains a live, incrementally-updated dependency graph."""

    def __init__(
        self,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        staleness_factor: float = DEFAULT_STALENESS_FACTOR,
        min_weight: float = DEFAULT_MIN_WEIGHT,
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if not 0.0 < staleness_factor <= 1.0:
            raise ValueError(f"staleness_factor must be in (0, 1], got {staleness_factor}")

        self.ema_alpha = ema_alpha
        self.staleness_factor = staleness_factor
        self.min_weight = min_weight

        self._edge_states: dict[tuple[str, str], EdgeState] = {}
        self._last_update: datetime | None = None


    def apply(self, merge_result: MergeResult) -> IncrementalUpdateResult:
        """
        Apply a MergeResult to the current graph state.

        Steps:
          1. For each edge in merge_result: EMA update or add
          2. For edges NOT in merge_result: apply staleness attenuation
          3. Prune edges below min_weight or past MAX_STALE_DAYS

        Returns IncrementalUpdateResult with the updated graph.
        """
        now = datetime.now(timezone.utc)
        observed_edges = {(e.source, e.target) for e in merge_result.canonical_edges}

        edges_added: list[tuple[str, str]] = []
        edges_updated: list[tuple[str, str]] = []
        edges_attenuated: list[tuple[str, str]] = []
        edges_pruned: list[tuple[str, str]] = []

        for edge in merge_result.canonical_edges:
            key = (edge.source, edge.target)

            if key in self._edge_states:
                old_state = self._edge_states[key]
                new_weight = (
                    self.ema_alpha * edge.canonical_weight
                    + (1.0 - self.ema_alpha) * old_state.weight
                )
                self._edge_states[key] = EdgeState(
                    source=edge.source,
                    target=edge.target,
                    weight=new_weight,
                    edge_type=edge.edge_type,
                    last_observed=now,
                    first_observed=old_state.first_observed,
                    update_count=old_state.update_count + 1,
                    channel=edge.channel,
                    topic=edge.topic,
                )
                edges_updated.append(key)
            else:
                self._edge_states[key] = EdgeState(
                    source=edge.source,
                    target=edge.target,
                    weight=edge.canonical_weight,
                    edge_type=edge.edge_type,
                    last_observed=now,
                    first_observed=now,
                    update_count=1,
                    channel=edge.channel,
                    topic=edge.topic,
                )
                edges_added.append(key)

        stale_keys = set(self._edge_states.keys()) - observed_edges
        for key in stale_keys:
            state = self._edge_states[key]
            stale_days = state.stale_days

            attenuation = self.staleness_factor**stale_days
            new_weight = state.weight * attenuation

            if new_weight < self.min_weight or stale_days > MAX_STALE_DAYS:
                edges_pruned.append(key)
                del self._edge_states[key]
                logger.debug(
                    "Pruned edge (%s → %s): weight=%.2f after %.1f stale days",
                    key[0],
                    key[1],
                    new_weight,
                    stale_days,
                )
            else:
                self._edge_states[key] = EdgeState(
                    source=state.source,
                    target=state.target,
                    weight=new_weight,
                    edge_type=state.edge_type,
                    last_observed=state.last_observed,
                    first_observed=state.first_observed,
                    update_count=state.update_count,
                    channel=state.channel,
                    topic=state.topic,
                )
                edges_attenuated.append(key)

        self._last_update = now

        G = self._build_graph()

        if edges_pruned:
            logger.info(
                "Incremental update: +%d added, ~%d updated, -%d pruned, ↓%d attenuated",
                len(edges_added),
                len(edges_updated),
                len(edges_pruned),
                len(edges_attenuated),
            )

        return IncrementalUpdateResult(
            graph=G,
            edges_added=edges_added,
            edges_updated=edges_updated,
            edges_pruned=edges_pruned,
            edges_attenuated=edges_attenuated,
        )

    def current_graph(self) -> nx.DiGraph:
        """Return the current graph without applying any update."""
        return self._build_graph()

    def edge_state(self, source: str, target: str) -> EdgeState | None:
        """Return the live state for a specific edge, or None if not tracked."""
        return self._edge_states.get((source, target))

    def stale_edges(self, threshold_days: float = 7.0) -> list[tuple[str, str]]:
        """Return edges not observed in the last threshold_days days."""
        return [
            key for key, state in self._edge_states.items() if state.stale_days > threshold_days
        ]

    @property
    def edge_count(self) -> int:
        return len(self._edge_states)

    @property
    def last_update(self) -> datetime | None:
        return self._last_update


    def _build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for state in self._edge_states.values():
            G.add_edge(
                state.source,
                state.target,
                weight=round(state.weight, 4),
                edge_type=state.edge_type,
                channel=state.channel,
                topic=state.topic,
                last_observed=state.last_observed.isoformat(),
                update_count=state.update_count,
            )
        return G
