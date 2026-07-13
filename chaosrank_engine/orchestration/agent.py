"""Data models for regional agent graph snapshots.
These models capture local dependency observations before they are merged
into the central canonical graph.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
import networkx as nx


@dataclass
class EdgeObservation:
    source: str
    target: str
    weight: float
    confidence: float
    edge_type: str = "sync"
    channel: str | None = None
    topic: str | None = None


@dataclass
class LocalGraphSnapshot:
    agent_id: str
    total_spans: int
    edges: list[EdgeObservation]
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scope_metadata: dict = field(default_factory=dict)

    def to_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for e in self.edges:
            G.add_edge(e.source, e.target, weight=e.weight, edge_type=e.edge_type)
            if e.channel:
                G[e.source][e.target]["channel"] = e.channel
            if e.topic:
                G[e.source][e.target]["topic"] = e.topic
        return G
