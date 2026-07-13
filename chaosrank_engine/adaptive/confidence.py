"""Confidence interval computation for risk scores.
Calculates a blended uncertainty metric based on graph sparsity, incident depth,
and topology staleness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import networkx as nx

from chaosrank_engine.parser.incidents import ServiceIncidents

logger = logging.getLogger(__name__)

MIN_RELIABLE_N = 10

# Days after which a graph observation is considered stale
STALENESS_THRESHOLD_DAYS = 30.0

# Component weights in CI formula — sum to 1.0
W_SPARSITY = 0.40
W_HISTORY = 0.40
W_AGE = 0.20

# Z-critical values for common confidence levels
Z_CRITICAL = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}

# Nodes with CI width above this threshold are flagged as low confidence
LOW_CONFIDENCE_THRESHOLD = 0.35


@dataclass
class ConfidenceResult:
    service: str
    risk_score: float
    ci_width: float
    ci_lower: float
    ci_upper: float
    confidence_level: float  # e.g. 0.95
    low_confidence: bool
    sparsity_component: float  # contribution from graph sparsity
    history_component: float  # contribution from incident depth
    age_component: float  # contribution from graph staleness
    reason: str | None  # human-readable explanation if low confidence


def compute_confidence(
    service: str,
    risk_score: float,
    G: nx.DiGraph,
    service_incidents: dict[str, ServiceIncidents],
    last_observed: datetime | None = None,
    confidence_level: float = 0.95,
) -> ConfidenceResult:
    """
    Compute a confidence interval for a single service's risk score.

    Parameters
    ----------
    service:           Service name
    risk_score:        The computed risk score in [0, 1]
    G:                 The dependency graph (used for sparsity)
    service_incidents: Incident history (used for history depth)
    last_observed:     When the graph topology was last observed.
                       If None, assumes observation is current.
    confidence_level:  0.90, 0.95, or 0.99
    """
    if confidence_level not in Z_CRITICAL:
        raise ValueError(
            f"confidence_level must be one of {list(Z_CRITICAL.keys())}, got {confidence_level}"
        )

    sparsity_c = _sparsity_component(service, G)
    history_c = _history_component(service, service_incidents)
    age_c = _age_component(last_observed)

    ci_width = W_SPARSITY * sparsity_c + W_HISTORY * history_c + W_AGE * age_c

    z = Z_CRITICAL[confidence_level]
    margin = ci_width * z
    ci_lower = max(0.0, round(risk_score - margin, 4))
    ci_upper = min(1.0, round(risk_score + margin, 4))

    low_confidence = ci_width > LOW_CONFIDENCE_THRESHOLD
    reason = _low_confidence_reason(low_confidence, sparsity_c, history_c, age_c)

    return ConfidenceResult(
        service=service,
        risk_score=round(risk_score, 4),
        ci_width=round(ci_width, 4),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        low_confidence=low_confidence,
        sparsity_component=round(sparsity_c, 4),
        history_component=round(history_c, 4),
        age_component=round(age_c, 4),
        reason=reason,
    )


def compute_all_confidence(
    ranked: list[dict],
    G: nx.DiGraph,
    service_incidents: dict[str, ServiceIncidents],
    last_observed: datetime | None = None,
    confidence_level: float = 0.95,
) -> dict[str, ConfidenceResult]:
    """
    Compute confidence intervals for all ranked services.

    Parameters
    ----------
    ranked: Output of rank_services() — list of dicts with 'service' and 'risk'
    """
    results = {}
    for row in ranked:
        service = row["service"]
        risk_score = row["risk"]
        results[service] = compute_confidence(
            service=service,
            risk_score=risk_score,
            G=G,
            service_incidents=service_incidents,
            last_observed=last_observed,
            confidence_level=confidence_level,
        )
    return results


# Internal component computations


def _sparsity_component(service: str, G: nx.DiGraph) -> float:
    """
    Graph sparsity contribution for service v.

    A service with few observed edges relative to the maximum in the
    graph has high sparsity — its centrality scores are less reliable.

    sparsity(v) = 1 - (degree(v) / max_degree_in_graph)
    """
    if G.number_of_nodes() == 0:
        return 1.0

    if service not in G:
        return 1.0

    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    if max_degree == 0:
        return 1.0

    service_degree = degrees.get(service, 0)
    return 1.0 - (service_degree / max_degree)


def _history_component(
    service: str,
    service_incidents: dict[str, ServiceIncidents],
) -> float:
    """
    Incident history shallowness for service v.

    history_shallowness(v) = 1 - min(incident_count(v) / MIN_RELIABLE_N, 1.0)

    A service with 0 incidents returns 1.0 (maximum uncertainty).
    A service with >= MIN_RELIABLE_N incidents returns 0.0 (minimum uncertainty
    from this component).
    """
    si = service_incidents.get(service)
    if si is None or not si.incidents:
        return 1.0

    count = len(si.incidents)
    return 1.0 - min(count / MIN_RELIABLE_N, 1.0)


def _age_component(last_observed: datetime | None) -> float:
    """
    Graph staleness contribution.

    age(v) = days_since_last_observation / STALENESS_THRESHOLD_DAYS

    Capped at 1.0 — beyond the threshold, maximum age uncertainty applies.
    If last_observed is None (current observation), returns 0.0.
    """
    if last_observed is None:
        return 0.0

    days_old = (datetime.now(timezone.utc) - last_observed).total_seconds() / 86400
    return min(days_old / STALENESS_THRESHOLD_DAYS, 1.0)


def _low_confidence_reason(
    low_confidence: bool,
    sparsity_c: float,
    history_c: float,
    age_c: float,
) -> str | None:
    if not low_confidence:
        return None

    # Identify the dominant contributor
    components = {
        "insufficient graph observations": sparsity_c * W_SPARSITY,
        "shallow incident history": history_c * W_HISTORY,
        "stale graph topology": age_c * W_AGE,
    }
    dominant = max(components, key=components.__getitem__)
    return f"Low confidence — {dominant}"
