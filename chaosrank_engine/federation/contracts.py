"""Adapter interfaces for multi-domain risk scoring in ChaosRank.
Enables pluggable discovery of graph topology, incident history, and load metrics
across heterogeneous environments.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import networkx as nx


# Shared data types


@dataclass
class DomainIncident:
    """A single incident record from any domain."""

    component_id: str
    timestamp: datetime
    severity: str  # must map to: critical, high, medium, low
    type: str  # latency, error, timeout, or other
    request_volume: float | None  # operational load at incident time
    domain_id: str  # which domain this incident belongs to
    raw: dict = field(default_factory=dict)  # original record


@dataclass
class EdgeSpec:
    """A single directed dependency edge spec."""

    source: str
    target: str
    weight: float
    edge_type: str = "sync"  # sync | async
    channel: str | None = None  # async only: kafka, sqs, rabbitmq, etc.
    topic: str | None = None  # async only: topic or queue name
    metadata: dict = field(default_factory=dict)


@dataclass
class ComponentSpec:
    """Metadata for a graph node/component."""

    component_id: str
    domain_id: str
    component_type: str | None = None  # service, database, queue, supplier, etc.
    metadata: dict = field(default_factory=dict)


# Interface 1 — GraphSourceAdapter


class GraphSourceAdapter(ABC):
    """Produces a weighted directed dependency graph for a domain."""

    @property
    @abstractmethod
    def domain_id(self) -> str:
        """Unique identifier for this domain (e.g. 'cloud-infra', 'supply-chain')."""

    @abstractmethod
    def build_graph(self) -> tuple[nx.DiGraph, list[ComponentSpec]]:
        """
        Build and return the dependency graph for this domain.

        Returns
        -------
        graph:      nx.DiGraph with edge attributes: weight, edge_type, channel, topic
        components: list of ComponentSpec for all nodes (metadata only)

        The graph must be self-consistent: every node in the graph must have
        a corresponding ComponentSpec in the components list.
        """

    def validate(self, graph: nx.DiGraph, components: list[ComponentSpec]) -> list[str]:
        """
        Validate graph and component list consistency.
        Returns list of validation error strings (empty = valid).
        Default implementation checks node/component alignment.
        Override to add domain-specific validation.
        """
        errors = []
        graph_nodes = set(graph.nodes())
        component_ids = {c.component_id for c in components}

        for node in graph_nodes:
            if node not in component_ids:
                errors.append(f"Graph node '{node}' has no corresponding ComponentSpec")
        for comp in components:
            if comp.domain_id != self.domain_id:
                errors.append(
                    f"ComponentSpec '{comp.component_id}' has domain_id "
                    f"'{comp.domain_id}', expected '{self.domain_id}'"
                )
        for u, v, data in graph.edges(data=True):
            if data.get("weight", 0) < 0:
                errors.append(f"Edge ({u} → {v}) has negative weight")
            if data.get("edge_type", "sync") not in ("sync", "async"):
                errors.append(f"Edge ({u} → {v}) has invalid edge_type '{data.get('edge_type')}'")
        return errors


# Interface 2 — IncidentSourceAdapter


class IncidentSourceAdapter(ABC):
    """Produces an operational incident history for a domain."""

    @property
    @abstractmethod
    def domain_id(self) -> str:
        """Must match the domain_id of the corresponding GraphSourceAdapter."""

    @abstractmethod
    def fetch_incidents(
        self,
        window_days: int,
    ) -> list[DomainIncident]:
        """
        Fetch incidents for the observation window.

        Parameters
        ----------
        window_days: number of days to look back from now

        Returns
        -------
        list of DomainIncident, ordered by timestamp ascending
        """

    def normalize_severity(self, raw_severity: str) -> str:
        """
        Normalize domain-specific severity to ChaosRank severity levels.
        Default implementation handles common conventions.
        Override for domain-specific mappings.
        """
        mapping = {
            # PagerDuty
            "p1": "critical",
            "p2": "high",
            "p3": "medium",
            "p4": "low",
            "p5": "low",
            # Opsgenie
            "critical": "critical",
            "high": "high",
            "moderate": "medium",
            "medium": "medium",
            "low": "low",
            "info": "low",
            # Alertmanager
            "page": "critical",
            "ticket": "medium",
            "warning": "medium",
            # Supply chain
            "force_majeure": "critical",
            "disruption": "high",
            "delay": "medium",
            "minor": "low",
            # Power grid
            "blackout": "critical",
            "brownout": "high",
            "deviation": "medium",
            "advisory": "low",
        }
        return mapping.get(raw_severity.lower(), "medium")


# Interface 3 — LoadMetricAdapter


class LoadMetricAdapter(ABC):
    """Supports historical load queries for components in a domain."""

    @property
    @abstractmethod
    def domain_id(self) -> str:
        """Must match the domain_id of the corresponding GraphSourceAdapter."""

    @abstractmethod
    def load_at(
        self,
        component_id: str,
        timestamp: datetime,
    ) -> float | None:
        """
        Return the operational load of component at the given timestamp.

        Used for per-incident traffic normalization.
        Returns None if data is unavailable — scorer falls back to window average.
        """

    @abstractmethod
    def mean_load(
        self,
        component_id: str,
        start: datetime,
        end: datetime,
    ) -> float | None:
        """
        Return the mean operational load of component over [start, end].

        Used as fallback when point-in-time load is unavailable.
        Returns None if data is unavailable — scorer skips normalization.
        """


# Domain registration bundle


@dataclass
class DomainBundle:
    """
    A complete domain registration: graph + incidents + load metric.

    All three adapters must share the same domain_id.
    load_metric is optional — if None, per-incident normalization
    falls back to window average for all incidents in this domain.
    """

    graph: GraphSourceAdapter
    incidents: IncidentSourceAdapter
    load_metric: LoadMetricAdapter | None = None

    def __post_init__(self) -> None:
        if self.graph.domain_id != self.incidents.domain_id:
            raise ValueError(
                f"domain_id mismatch: graph={self.graph.domain_id!r}, "
                f"incidents={self.incidents.domain_id!r}"
            )
        if self.load_metric and self.load_metric.domain_id != self.graph.domain_id:
            raise ValueError(
                f"domain_id mismatch: graph={self.graph.domain_id!r}, "
                f"load_metric={self.load_metric.domain_id!r}"
            )

    @property
    def domain_id(self) -> str:
        return self.graph.domain_id
