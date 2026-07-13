"""Unit and integration tests for ChaosRank's federation logic.
Covers multi-domain registry, federated graph construction, inter-domain edge
validation, and incident correlation.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import networkx as nx
import pytest

from chaosrank_engine.federation.contracts import (
    ComponentSpec,
    DomainBundle,
    DomainIncident,
    GraphSourceAdapter,
    IncidentSourceAdapter,
    LoadMetricAdapter,
)
from chaosrank_engine.federation.correlation import (
    correlate_incidents,
    _build_time_buckets,
)
from chaosrank_engine.federation.federated_graph import (
    FederatedGraphBuilder,
    InterDomainEdge,
    _qualify,
    unqualify,
)
from chaosrank_engine.federation.registry import DomainRegistry
from chaosrank_engine.parser.incidents import Incident, ServiceIncidents


# Test adapters


class StubGraphAdapter(GraphSourceAdapter):
    """Stub adapter that builds a graph from a list of EdgeSpecs."""

    def __init__(self, domain_id: str, edges: list[tuple[str, str, float, str]] = None):
        self._domain_id = domain_id
        self._edges = edges or [("svc-a", "svc-b", 100.0, "sync")]

    @property
    def domain_id(self) -> str:
        return self._domain_id

    def build_graph(self):
        G = nx.DiGraph()
        components = []
        nodes_seen = set()
        for src, tgt, weight, edge_type in self._edges:
            G.add_edge(src, tgt, weight=weight, edge_type=edge_type)
            for n in (src, tgt):
                if n not in nodes_seen:
                    components.append(
                        ComponentSpec(
                            component_id=n,
                            domain_id=self._domain_id,
                        )
                    )
                    nodes_seen.add(n)
        return G, components


class StubIncidentAdapter(IncidentSourceAdapter):
    def __init__(self, domain_id: str, incidents: list[DomainIncident] = None):
        self._domain_id = domain_id
        self._incidents = incidents or []

    @property
    def domain_id(self) -> str:
        return self._domain_id

    def fetch_incidents(self, window_days: int) -> list[DomainIncident]:
        return self._incidents


class StubLoadAdapter(LoadMetricAdapter):
    def __init__(self, domain_id: str, default_load: float = 1000.0):
        self._domain_id = domain_id
        self._default_load = default_load

    @property
    def domain_id(self) -> str:
        return self._domain_id

    def load_at(self, component_id: str, timestamp: datetime) -> float | None:
        return self._default_load

    def mean_load(self, component_id: str, start: datetime, end: datetime) -> float | None:
        return self._default_load


def _make_bundle(
    domain_id: str,
    edges: list[tuple[str, str, float, str]] = None,
    incidents: list[DomainIncident] = None,
    with_load: bool = True,
) -> DomainBundle:
    return DomainBundle(
        graph=StubGraphAdapter(domain_id, edges),
        incidents=StubIncidentAdapter(domain_id, incidents),
        load_metric=StubLoadAdapter(domain_id) if with_load else None,
    )


def _make_incident(
    component_id: str,
    domain_id: str,
    timestamp: datetime = None,
    severity: str = "high",
) -> DomainIncident:
    return DomainIncident(
        component_id=component_id,
        timestamp=timestamp or datetime.now(timezone.utc) - timedelta(days=1),
        severity=severity,
        type="error",
        request_volume=1000.0,
        domain_id=domain_id,
    )


def _make_service_incidents(service: str, timestamps: list[datetime]) -> ServiceIncidents:
    incidents = [
        Incident(
            timestamp=ts,
            service=service,
            type="error",
            severity="high",
            request_volume=1000.0,
        )
        for ts in timestamps
    ]
    return ServiceIncidents(service=service, incidents=incidents)


class TestContracts:
    """Tests for federation adapter contracts and severity normalization."""
    def test_domain_bundle_valid(self):
        bundle = _make_bundle("cloud")
        assert bundle.domain_id == "cloud"

    def test_domain_bundle_mismatched_graph_incidents(self):
        with pytest.raises(ValueError, match="domain_id mismatch"):
            DomainBundle(
                graph=StubGraphAdapter("cloud"),
                incidents=StubIncidentAdapter("supply-chain"),
            )

    def test_domain_bundle_mismatched_load_metric(self):
        with pytest.raises(ValueError, match="domain_id mismatch"):
            DomainBundle(
                graph=StubGraphAdapter("cloud"),
                incidents=StubIncidentAdapter("cloud"),
                load_metric=StubLoadAdapter("supply-chain"),
            )

    def test_domain_bundle_no_load_metric_ok(self):
        bundle = _make_bundle("cloud", with_load=False)
        assert bundle.load_metric is None

    def test_normalize_severity_pagerduty(self):
        adapter = StubIncidentAdapter("cloud")
        assert adapter.normalize_severity("p1") == "critical"
        assert adapter.normalize_severity("p2") == "high"
        assert adapter.normalize_severity("p3") == "medium"
        assert adapter.normalize_severity("p5") == "low"

    def test_normalize_severity_unknown_defaults_medium(self):
        adapter = StubIncidentAdapter("cloud")
        assert adapter.normalize_severity("unknown-level") == "medium"

    def test_normalize_severity_case_insensitive(self):
        adapter = StubIncidentAdapter("cloud")
        assert adapter.normalize_severity("CRITICAL") == "critical"
        assert adapter.normalize_severity("High") == "high"

    def test_graph_adapter_validate_catches_missing_component(self):
        adapter = StubGraphAdapter("cloud")
        G = nx.DiGraph()
        G.add_edge("a", "b", weight=10, edge_type="sync")
        # Pass empty components — both nodes should error
        errors = adapter.validate(G, [])
        assert any("a" in e or "b" in e for e in errors)

    def test_graph_adapter_validate_catches_wrong_domain(self):
        adapter = StubGraphAdapter("cloud")
        G = nx.DiGraph()
        G.add_edge("a", "b", weight=10, edge_type="sync")
        components = [
            ComponentSpec(component_id="a", domain_id="wrong-domain"),
            ComponentSpec(component_id="b", domain_id="wrong-domain"),
        ]
        errors = adapter.validate(G, components)
        assert any("wrong-domain" in e for e in errors)

    def test_graph_adapter_validate_catches_negative_weight(self):
        adapter = StubGraphAdapter("cloud")
        G = nx.DiGraph()
        G.add_edge("a", "b", weight=-5, edge_type="sync")
        components = [
            ComponentSpec(component_id="a", domain_id="cloud"),
            ComponentSpec(component_id="b", domain_id="cloud"),
        ]
        errors = adapter.validate(G, components)
        assert any("negative weight" in e for e in errors)

    def test_graph_adapter_validate_catches_invalid_edge_type(self):
        adapter = StubGraphAdapter("cloud")
        G = nx.DiGraph()
        G.add_edge("a", "b", weight=10, edge_type="ftp")
        components = [
            ComponentSpec(component_id="a", domain_id="cloud"),
            ComponentSpec(component_id="b", domain_id="cloud"),
        ]
        errors = adapter.validate(G, components)
        assert any("invalid edge_type" in e for e in errors)


class TestDomainRegistry:
    """Tests for the domain discovery and lifecycle registry."""
    def test_register_and_retrieve(self):
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud"))
        assert "cloud" in reg
        assert reg.get("cloud").domain_id == "cloud"

    def test_register_duplicate_raises(self):
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_make_bundle("cloud"))

    def test_replace_updates_registration(self):
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud"))
        new_bundle = _make_bundle("cloud", edges=[("x", "y", 50.0, "sync")])
        reg.replace(new_bundle)
        assert reg.get("cloud").graph._edges == [("x", "y", 50.0, "sync")]

    def test_unregister_removes_domain(self):
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud"))
        reg.unregister("cloud")
        assert "cloud" not in reg

    def test_unregister_missing_raises(self):
        reg = DomainRegistry()
        with pytest.raises(KeyError):
            reg.unregister("nonexistent")

    def test_get_missing_raises(self):
        reg = DomainRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_validate_empty_registry(self):
        reg = DomainRegistry()
        errors = reg.validate()
        assert errors == ["No domains registered"]

    def test_validate_single_domain_valid(self):
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud"))
        errors = reg.validate()
        assert errors == []

    def test_validate_detects_node_collision(self):
        reg = DomainRegistry()
        # Both domains have a node called "api-gateway"
        reg.register(_make_bundle("cloud", edges=[("api-gateway", "payment", 100.0, "sync")]))
        reg.register(_make_bundle("staging", edges=[("api-gateway", "cart", 100.0, "sync")]))
        errors = reg.validate()
        assert any("api-gateway" in e and "collision" in e for e in errors)

    def test_domain_ids_returns_all(self):
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud"))
        reg.register(_make_bundle("supply-chain"))
        assert set(reg.domain_ids) == {"cloud", "supply-chain"}

    def test_len(self):
        reg = DomainRegistry()
        assert len(reg) == 0
        reg.register(_make_bundle("cloud"))
        assert len(reg) == 1

    def test_summary_contains_domains(self):
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud"))
        summary = reg.summary()
        assert "cloud" in summary
        assert "nodes" in summary["cloud"]


class TestFederatedGraphBuilder:
    """Tests for building a unified graph from multiple independent domains."""
    def test_build_single_domain(self):
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud"))
        builder = FederatedGraphBuilder(reg)
        result = builder.build(window_days=7)

        assert result.graph.number_of_nodes() > 0
        assert result.graph.number_of_edges() > 0
        assert "cloud" in result.domain_node_map

    def test_build_qualifies_node_ids(self):
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud", edges=[("svc-a", "svc-b", 100.0, "sync")]))
        builder = FederatedGraphBuilder(reg)
        result = builder.build()

        nodes = set(result.graph.nodes())
        assert "cloud/svc-a" in nodes
        assert "cloud/svc-b" in nodes
        assert "svc-a" not in nodes  # unqualified IDs should not appear

    def test_build_two_domains_no_collision(self):
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud", edges=[("api", "db", 100.0, "sync")]))
        reg.register(_make_bundle("supply-chain", edges=[("supplier", "warehouse", 50.0, "sync")]))
        builder = FederatedGraphBuilder(reg)
        result = builder.build()

        nodes = set(result.graph.nodes())
        assert "cloud/api" in nodes
        assert "supply-chain/supplier" in nodes
        assert result.graph.number_of_nodes() == 4

    def test_inter_domain_edge_added(self):
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud", edges=[("order-processor", "payment", 100.0, "sync")]))
        reg.register(
            _make_bundle("supply-chain", edges=[("supplier-api", "warehouse", 50.0, "sync")])
        )
        builder = FederatedGraphBuilder(reg)
        builder.add_inter_domain_edge(
            InterDomainEdge(
                source_domain="cloud",
                source_component="order-processor",
                target_domain="supply-chain",
                target_component="supplier-api",
                weight=80.0,
                edge_type="sync",
            )
        )
        result = builder.build()

        assert result.graph.has_edge("cloud/order-processor", "supply-chain/supplier-api")
        assert len(result.inter_domain_edges) == 1

    def test_inter_domain_edge_missing_node_warned(self):
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud"))
        builder = FederatedGraphBuilder(reg)
        builder.add_inter_domain_edge(
            InterDomainEdge(
                source_domain="cloud",
                source_component="nonexistent",
                target_domain="cloud",
                target_component="svc-b",
                weight=10.0,
            )
        )
        result = builder.build()
        assert any("nonexistent" in w for w in result.warnings)

    def test_empty_registry_raises(self):
        reg = DomainRegistry()
        builder = FederatedGraphBuilder(reg)
        with pytest.raises(ValueError, match="No domains registered"):
            builder.build()

    def test_incidents_collected_with_qualified_ids(self):
        now = datetime.now(timezone.utc)
        inc = _make_incident("svc-a", "cloud", timestamp=now - timedelta(days=1))
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud", incidents=[inc]))
        builder = FederatedGraphBuilder(reg)
        result = builder.build(window_days=30)

        assert "cloud/svc-a" in result.service_incidents
        assert len(result.service_incidents["cloud/svc-a"].incidents) == 1

    def test_load_metric_populates_request_volume(self):
        now = datetime.now(timezone.utc)
        # Incident with no request_volume — should be populated by LoadMetricAdapter
        inc = DomainIncident(
            component_id="svc-a",
            timestamp=now - timedelta(days=1),
            severity="high",
            type="error",
            request_volume=None,
            domain_id="cloud",
        )
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud", incidents=[inc], with_load=True))
        builder = FederatedGraphBuilder(reg)
        result = builder.build(window_days=30)

        si = result.service_incidents.get("cloud/svc-a")
        assert si is not None
        assert si.incidents[0].request_volume == 1000.0  # from StubLoadAdapter

    def test_no_load_metric_leaves_none_volume(self):
        now = datetime.now(timezone.utc)
        inc = DomainIncident(
            component_id="svc-a",
            timestamp=now - timedelta(days=1),
            severity="high",
            type="error",
            request_volume=None,
            domain_id="cloud",
        )
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud", incidents=[inc], with_load=False))
        builder = FederatedGraphBuilder(reg)
        result = builder.build(window_days=30)

        si = result.service_incidents.get("cloud/svc-a")
        assert si is not None
        assert si.incidents[0].request_volume is None

    def test_clear_inter_domain_edges(self):
        reg = DomainRegistry()
        reg.register(_make_bundle("cloud"))
        builder = FederatedGraphBuilder(reg)
        builder.add_inter_domain_edge(
            InterDomainEdge(
                source_domain="cloud",
                source_component="svc-a",
                target_domain="cloud",
                target_component="svc-b",
                weight=10.0,
            )
        )
        builder.clear_inter_domain_edges()
        result = builder.build()
        assert result.inter_domain_edges == []


# ID qualification


class TestQualify:
    def test_qualify(self):
        assert _qualify("svc-a", "cloud") == "cloud/svc-a"

    def test_unqualify(self):
        domain, component = unqualify("cloud/svc-a")
        assert domain == "cloud"
        assert component == "svc-a"

    def test_unqualify_invalid_raises(self):
        with pytest.raises(ValueError, match="not a domain-qualified ID"):
            unqualify("svc-a")

    def test_qualify_unqualify_roundtrip(self):
        original = ("supply-chain", "supplier-api-v2")
        qualified = _qualify(original[1], original[0])
        domain, component = unqualify(qualified)
        assert domain == original[0]
        assert component == original[1]


class TestCorrelation:
    """Tests for incident co-occurrence and risk propagation correlation."""
    def _base_graph(self):
        G = nx.DiGraph()
        G.add_edge("svc-a", "svc-b", weight=100.0)
        G.add_edge("svc-b", "svc-c", weight=50.0)
        return G

    def test_no_incidents_returns_unchanged_risks(self):
        G = self._base_graph()
        base = {"svc-a": 0.8, "svc-b": 0.6, "svc-c": 0.4}
        result = correlate_incidents(G, {}, base)
        assert result.adjusted_risks == base
        assert result.propagation_links == []

    def test_no_cooccurrence_no_links(self):
        G = self._base_graph()
        now = datetime.now(timezone.utc)
        # svc-a incidents in hour 0, svc-b incidents in hour 5 — no overlap
        si = {
            "svc-a": _make_service_incidents("svc-a", [now - timedelta(hours=10)]),
            "svc-b": _make_service_incidents("svc-b", [now - timedelta(hours=1)]),
        }
        base = {"svc-a": 0.8, "svc-b": 0.6}
        result = correlate_incidents(G, si, base, correlation_window=30)
        assert result.propagation_links == []

    def test_cooccurrence_with_path_creates_link(self):
        G = self._base_graph()
        now = datetime.now(timezone.utc)
        # Both services have incidents at the same time → same bucket
        ts = now - timedelta(hours=1)
        si = {
            "svc-a": _make_service_incidents("svc-a", [ts, ts + timedelta(minutes=5)]),
            "svc-b": _make_service_incidents(
                "svc-b", [ts + timedelta(minutes=2), ts + timedelta(minutes=7)]
            ),
        }
        base = {"svc-a": 0.8, "svc-b": 0.6}
        result = correlate_incidents(G, si, base, correlation_window=30, min_cooccurrence=1)
        # svc-a → svc-b path exists
        links_src = [link for link in result.propagation_links if link.source == "svc-a"]
        assert len(links_src) > 0
        assert links_src[0].path_exists

    def test_cooccurrence_without_path_no_link(self):
        G = self._base_graph()  # svc-a → svc-b → svc-c, no reverse edges
        now = datetime.now(timezone.utc)
        ts = now - timedelta(hours=1)
        # svc-c and svc-a co-occur but no path svc-c → svc-a
        si = {
            "svc-c": _make_service_incidents("svc-c", [ts, ts + timedelta(minutes=5)]),
            "svc-a": _make_service_incidents("svc-a", [ts + timedelta(minutes=2)]),
        }
        base = {"svc-a": 0.8, "svc-c": 0.4}
        result = correlate_incidents(G, si, base, correlation_window=30, min_cooccurrence=1)
        # No path svc-c → svc-a so no propagation link for this direction
        links = [
            link
            for link in result.propagation_links
            if link.source == "svc-c" and link.target == "svc-a"
        ]
        assert links == []

    def test_risk_elevated_for_propagation_source(self):
        G = self._base_graph()
        now = datetime.now(timezone.utc)
        ts = now - timedelta(hours=1)
        si = {
            "svc-a": _make_service_incidents(
                "svc-a", [ts, ts + timedelta(minutes=3), ts + timedelta(minutes=8)]
            ),
            "svc-b": _make_service_incidents(
                "svc-b",
                [ts + timedelta(minutes=2), ts + timedelta(minutes=7), ts + timedelta(minutes=12)],
            ),
        }
        base = {"svc-a": 0.5, "svc-b": 0.6}
        result = correlate_incidents(G, si, base, correlation_window=30, min_cooccurrence=2)
        assert result.adjusted_risks["svc-a"] >= base["svc-a"]

    def test_root_cause_candidates_ordered_by_confidence(self):
        G = nx.DiGraph()
        G.add_edge("root", "child-a", weight=200.0)
        G.add_edge("root", "child-b", weight=150.0)
        now = datetime.now(timezone.utc)
        ts = now - timedelta(hours=1)
        si = {
            "root": _make_service_incidents(
                "root", [ts, ts + timedelta(minutes=5), ts + timedelta(minutes=10)]
            ),
            "child-a": _make_service_incidents(
                "child-a",
                [ts + timedelta(minutes=2), ts + timedelta(minutes=7), ts + timedelta(minutes=12)],
            ),
            "child-b": _make_service_incidents(
                "child-b",
                [ts + timedelta(minutes=3), ts + timedelta(minutes=8), ts + timedelta(minutes=13)],
            ),
        }
        base = {"root": 0.7, "child-a": 0.5, "child-b": 0.4}
        result = correlate_incidents(G, si, base, correlation_window=30, min_cooccurrence=2)
        if result.root_cause_candidates:
            assert result.root_cause_candidates[0] == "root"

    def test_adjusted_risks_clamped_to_one(self):
        G = nx.DiGraph()
        G.add_edge("a", "b", weight=10000.0)  # very high weight
        now = datetime.now(timezone.utc)
        ts = now - timedelta(hours=1)
        si = {
            "a": _make_service_incidents("a", [ts] * 10),
            "b": _make_service_incidents("b", [ts] * 10),
        }
        base = {"a": 0.99, "b": 0.5}
        result = correlate_incidents(G, si, base, correlation_window=60, min_cooccurrence=1)
        for risk in result.adjusted_risks.values():
            assert risk <= 1.0

    def test_time_bucket_grouping(self):
        # Using a fixed datetime to avoid boundary flakiness
        now = datetime(2026, 4, 3, 12, 0, 0)
        # Two incidents 10 minutes apart in a 30-minute window → same bucket
        si = {
            "svc": _make_service_incidents(
                "svc",
                [
                    now - timedelta(minutes=20),
                    now - timedelta(minutes=15),
                ],
            )
        }
        buckets = _build_time_buckets(si, window_minutes=30)
        # Both in the same 30-min bucket
        assert len(buckets["svc"]) == 1

    def test_time_bucket_different_windows(self):
        # Using a fixed datetime to avoid boundary flakiness
        now = datetime(2026, 4, 3, 12, 0, 0)
        # Two incidents 60 minutes apart in a 30-minute window → different buckets
        si = {
            "svc": _make_service_incidents(
                "svc",
                [
                    now - timedelta(minutes=90),
                    now - timedelta(minutes=20),
                ],
            )
        }
        buckets = _build_time_buckets(si, window_minutes=30)
        assert len(buckets["svc"]) == 2

    def test_empty_service_incidents(self):
        G = self._base_graph()
        base = {"svc-a": 0.8}
        result = correlate_incidents(G, {}, base)
        assert result.adjusted_risks == base
        assert result.propagation_links == []
        assert result.root_cause_candidates == []
