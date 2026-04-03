"""Multi-domain, federated graph construction for ChaosRank.
Combines independent domain graphs into a unified dependency model for
cross-cluster and cross-account risk analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx

from chaosrank_engine.federation.contracts import (
    ComponentSpec,
)
from chaosrank_engine.federation.registry import DomainRegistry
from chaosrank_engine.parser.incidents import Incident, ServiceIncidents

logger = logging.getLogger(__name__)


@dataclass
class InterDomainEdge:
    """Represents an explicit dependency between components in different domains."""

    source_domain: str
    source_component: str
    target_domain: str
    target_component: str
    weight: float = 1.0
    edge_type: str = "sync"  # sync | async
    channel: str | None = None
    topic: str | None = None


@dataclass
class FederatedGraphResult:
    """Container for the unified federated graph and associated system state."""

    graph: nx.DiGraph
    service_incidents: dict[str, ServiceIncidents]
    components: list[ComponentSpec]
    domain_node_map: dict[str, set[str]]
    inter_domain_edges: list[InterDomainEdge]
    warnings: list[str] = field(default_factory=list)


class FederatedGraphBuilder:
    """Assembles a unified federated graph from multiple registered domains."""

    def __init__(self, registry: DomainRegistry) -> None:
        self.registry = registry
        self._inter_domain_edges: list[InterDomainEdge] = []

    # Dependency management

    def add_inter_domain_edge(self, edge: InterDomainEdge) -> None:
        """Declare an explicit cross-domain dependency."""
        self._inter_domain_edges.append(edge)
        logger.debug(
            "Inter-domain edge declared: %s/%s → %s/%s (weight=%.1f, type=%s)",
            edge.source_domain,
            edge.source_component,
            edge.target_domain,
            edge.target_component,
            edge.weight,
            edge.edge_type,
        )

    def clear_inter_domain_edges(self) -> None:
        self._inter_domain_edges.clear()

    # Unified graph construction

    def build(self, window_days: int = 30) -> FederatedGraphResult:
        """
        Build the federated graph from all registered domains.

        Parameters
        ----------
        window_days: incident history lookback window

        Returns
        -------
        FederatedGraphResult ready for the scoring pipeline
        """
        if not self.registry.bundles:
            raise ValueError("No domains registered — cannot build federated graph")

        warnings: list[str] = []
        all_components: list[ComponentSpec] = []
        domain_node_map: dict[str, set[str]] = {}
        G_fed = nx.DiGraph()

        # Step 1 — build per-domain graphs and merge into G_fed
        for bundle in self.registry.bundles:
            domain_id = bundle.domain_id
            try:
                graph, components = bundle.graph.build_graph()
            except Exception as exc:
                warnings.append(f"[{domain_id}] build_graph() failed: {exc} — domain skipped")
                continue

            validation_errors = bundle.graph.validate(graph, components)
            for e in validation_errors:
                warnings.append(f"[{domain_id}] validation: {e}")

            domain_nodes: set[str] = set()
            for node in graph.nodes():
                qualified = _qualify(node, domain_id)
                G_fed.add_node(qualified, domain_id=domain_id, original_id=node)
                domain_nodes.add(qualified)

            for u, v, data in graph.edges(data=True):
                qu, qv = _qualify(u, domain_id), _qualify(v, domain_id)
                G_fed.add_edge(qu, qv, **data)

            domain_node_map[domain_id] = domain_nodes
            all_components.extend(
                ComponentSpec(
                    component_id=_qualify(c.component_id, domain_id),
                    domain_id=c.domain_id,
                    component_type=c.component_type,
                    metadata=c.metadata,
                )
                for c in components
            )

            logger.info(
                "[%s] Merged %d nodes, %d edges into federated graph",
                domain_id,
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )

        # Step 2 — add inter-domain edges
        added_inter = []
        for edge in self._inter_domain_edges:
            src = _qualify(edge.source_component, edge.source_domain)
            tgt = _qualify(edge.target_component, edge.target_domain)

            if src not in G_fed:
                warnings.append(
                    f"Inter-domain edge skipped: source node '{src}' not in federated graph"
                )
                continue
            if tgt not in G_fed:
                warnings.append(
                    f"Inter-domain edge skipped: target node '{tgt}' not in federated graph"
                )
                continue

            G_fed.add_edge(
                src,
                tgt,
                weight=edge.weight,
                edge_type=edge.edge_type,
                channel=edge.channel,
                topic=edge.topic,
                inter_domain=True,
            )
            added_inter.append(edge)
            logger.debug("Added inter-domain edge: %s → %s", src, tgt)

        logger.info(
            "Federated graph built: %d nodes, %d edges, %d inter-domain edges, %d domains",
            G_fed.number_of_nodes(),
            G_fed.number_of_edges(),
            len(added_inter),
            len(domain_node_map),
        )

        # Step 3 — collect incidents from all domains
        service_incidents = self._collect_incidents(window_days, domain_node_map, warnings)

        return FederatedGraphResult(
            graph=G_fed,
            service_incidents=service_incidents,
            components=all_components,
            domain_node_map=domain_node_map,
            inter_domain_edges=added_inter,
            warnings=warnings,
        )

    # History aggregation

    def _collect_incidents(
        self,
        window_days: int,
        domain_node_map: dict[str, set[str]],
        warnings: list[str],
    ) -> dict[str, ServiceIncidents]:
        """
        Fetch incidents from all domain IncidentSourceAdapters and convert
        to ChaosRank ServiceIncidents format with qualified node IDs.
        """
        service_incidents: dict[str, ServiceIncidents] = {}

        for bundle in self.registry.bundles:
            domain_id = bundle.domain_id
            try:
                raw_incidents = bundle.incidents.fetch_incidents(window_days)
            except Exception as exc:
                warnings.append(
                    f"[{domain_id}] fetch_incidents() failed: {exc} — "
                    f"incidents skipped for this domain"
                )
                continue

            for di in raw_incidents:
                qualified_id = _qualify(di.component_id, domain_id)

                # Resolve point-in-time load via LoadMetricAdapter if available
                request_volume = di.request_volume
                if request_volume is None and bundle.load_metric:
                    try:
                        request_volume = bundle.load_metric.load_at(di.component_id, di.timestamp)
                    except Exception:
                        pass  # fallback handled by fragility scorer

                incident = Incident(
                    timestamp=di.timestamp,
                    service=qualified_id,
                    type=di.type,
                    severity=bundle.incidents.normalize_severity(di.severity),
                    request_volume=request_volume,
                )

                if qualified_id not in service_incidents:
                    service_incidents[qualified_id] = ServiceIncidents(service=qualified_id)
                service_incidents[qualified_id].incidents.append(incident)

            logger.info(
                "[%s] Collected %d incidents",
                domain_id,
                sum(
                    len(si.incidents)
                    for sid, si in service_incidents.items()
                    if sid in (domain_node_map.get(domain_id) or set())
                ),
            )

        return service_incidents


# Internal helpers


def _qualify(component_id: str, domain_id: str) -> str:
    """
    Produce a domain-qualified node ID: '{domain_id}/{component_id}'

    This prevents node ID collisions when the same component name appears
    in multiple domains (e.g. 'api-gateway' in both cloud-infra and staging).
    """
    return f"{domain_id}/{component_id}"


def unqualify(qualified_id: str) -> tuple[str, str]:
    """
    Split a qualified ID back into (domain_id, component_id).
    Raises ValueError if the ID is not domain-qualified.
    """
    if "/" not in qualified_id:
        raise ValueError(
            f"'{qualified_id}' is not a domain-qualified ID (expected 'domain_id/component_id')"
        )
    domain_id, component_id = qualified_id.split("/", 1)
    return domain_id, component_id
