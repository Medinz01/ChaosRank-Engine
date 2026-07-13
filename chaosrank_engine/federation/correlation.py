"""Graph-structure-aware incident correlation for ChaosRank.
Identifies systemic failure propagation by cross-referencing incident
co-occurrence patterns with directed dependency paths.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx

from chaosrank_engine.parser.incidents import ServiceIncidents

logger = logging.getLogger(__name__)

DEFAULT_CORRELATION_WINDOW_MINUTES = 30

MIN_COOCCURRENCE_COUNT = 2

PATH_WEIGHT_NORMALIZER = 1000.0


@dataclass
class PropagationLink:
    """Represents a detected incident propagation relationship between two nodes."""

    source: str
    target: str
    cooccurrence_count: int
    path_exists: bool
    path_weight: float
    propagation_confidence: float


@dataclass
class CorrelationResult:
    """Container for detected propagation links and adjusted risk scores."""

    propagation_links: list[PropagationLink]
    adjusted_risks: dict[str, float]
    root_cause_candidates: list[str]


def correlate_incidents(
    G: nx.DiGraph,
    service_incidents: dict[str, ServiceIncidents],
    base_risks: dict[str, float],
    correlation_window: int = DEFAULT_CORRELATION_WINDOW_MINUTES,
    min_cooccurrence: int = MIN_COOCCURRENCE_COUNT,
) -> CorrelationResult:
    """
    Detect systemic failure propagation and adjust risk scores.

    Parameters
    ----------
    G:                  Dependency graph (caller → callee)
    service_incidents:  Incident history per service
    base_risks:         Risk scores from rank_services() — dict[service, risk]
    correlation_window: Minutes — incidents within this window are co-occurring
    min_cooccurrence:   Minimum co-occurrence count to consider a signal

    Returns
    -------
    CorrelationResult with adjusted risk scores
    """
    if not service_incidents or not base_risks:
        return CorrelationResult(
            propagation_links=[],
            adjusted_risks=dict(base_risks),
            root_cause_candidates=[],
        )

    buckets = _build_time_buckets(service_incidents, correlation_window)
    cooccurrence = _compute_cooccurrence(buckets)
    links = _compute_propagation_links(G, cooccurrence, min_cooccurrence)
    adjusted_risks = _adjust_risks(base_risks, links)
    root_cause_candidates = _rank_root_causes(links)

    if links:
        logger.info(
            "Incident correlation: %d propagation links detected, %d root cause candidates",
            len(links),
            len(root_cause_candidates),
        )
    else:
        logger.debug("Incident correlation: no propagation links detected")

    return CorrelationResult(
        propagation_links=links,
        adjusted_risks=adjusted_risks,
        root_cause_candidates=root_cause_candidates,
    )




def _build_time_buckets(
    service_incidents: dict[str, ServiceIncidents],
    window_minutes: int,
) -> dict[str, set[int]]:
    """
    Build a dict of service → set of time bucket indices.

    A time bucket is an integer representing a window_minutes-sized slot.
    Two incidents are co-occurring if they fall in the same bucket.
    """
    buckets: dict[str, set[int]] = defaultdict(set)
    window_seconds = window_minutes * 60

    for service, si in service_incidents.items():
        for incident in si.incidents:
            bucket = int(incident.timestamp.timestamp()) // window_seconds
            buckets[service].add(bucket)

    return dict(buckets)


def _compute_cooccurrence(
    buckets: dict[str, set[int]],
) -> dict[tuple[str, str], int]:
    """
    Compute co-occurrence counts between all service pairs.

    co-occurrence(A, B) = |buckets[A] ∩ buckets[B]|

    Only computes pairs where both services have incident history.
    """
    services = list(buckets.keys())
    cooccurrence: dict[tuple[str, str], int] = {}

    for i in range(len(services)):
        for j in range(len(services)):
            if i == j:
                continue
            s1, s2 = services[i], services[j]
            shared = buckets[s1] & buckets[s2]
            if shared:
                cooccurrence[(s1, s2)] = len(shared)

    return cooccurrence


def _compute_propagation_links(
    G: nx.DiGraph,
    cooccurrence: dict[tuple[str, str], int],
    min_cooccurrence: int,
) -> list[PropagationLink]:
    """
    Filter co-occurring pairs by graph path existence.

    For each (source, target) pair with sufficient co-occurrence:
    - Check if a directed path source → target exists in G
    - If yes: this is a propagation candidate
    - Compute propagation_confidence = cooccurrence × path_weight / normalizer
    """
    links = []

    for (source, target), count in cooccurrence.items():
        if count < min_cooccurrence:
            continue

        path_exists = nx.has_path(G, source, target) if (source in G and target in G) else False

        path_weight = 0.0
        if path_exists:
            try:
                path = nx.shortest_path(G, source, target, weight=None)
                path_weight = sum(
                    G[path[i]][path[i + 1]].get("weight", 1.0) for i in range(len(path) - 1)
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                path_exists = False
                path_weight = 0.0

        if not path_exists:
            continue

        confidence = (count * path_weight) / PATH_WEIGHT_NORMALIZER

        links.append(
            PropagationLink(
                source=source,
                target=target,
                cooccurrence_count=count,
                path_exists=True,
                path_weight=round(path_weight, 4),
                propagation_confidence=round(confidence, 6),
            )
        )

    links.sort(key=lambda link: link.propagation_confidence, reverse=True)
    return links


def _adjust_risks(
    base_risks: dict[str, float],
    links: list[PropagationLink],
) -> dict[str, float]:
    """
    Elevate risk scores for confirmed propagation sources.

    risk_adjusted(u) = risk(u) × (1 + Σ propagation_confidence(u, v))

    Scores are clamped to [0, 1] after adjustment.
    Services with no outbound propagation links are unchanged.
    """
    if not links:
        return dict(base_risks)

    outbound_confidence: dict[str, float] = defaultdict(float)
    for link in links:
        outbound_confidence[link.source] += link.propagation_confidence

    adjusted = {}
    for service, risk in base_risks.items():
        conf = outbound_confidence.get(service, 0.0)
        if conf > 0:
            adjusted[service] = min(1.0, round(risk * (1.0 + conf), 4))
            logger.debug(
                "Risk adjusted: %s  %.4f → %.4f  (propagation_confidence=%.4f)",
                service,
                risk,
                adjusted[service],
                conf,
            )
        else:
            adjusted[service] = risk

    return adjusted


def _rank_root_causes(links: list[PropagationLink]) -> list[str]:
    """
    Rank services by total outbound propagation confidence.

    Services with high total outbound confidence are most likely root causes
    — they are upstream of many incident co-occurrences that follow graph edges.
    """
    outbound: dict[str, float] = defaultdict(float)
    for link in links:
        outbound[link.source] += link.propagation_confidence

    return sorted(outbound.keys(), key=lambda s: outbound[s], reverse=True)
