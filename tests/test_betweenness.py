"""Unit tests for the opt-in betweenness centrality feature in ChaosRank.
Verifies that bridge nodes are correctly identified and that weight inversion
logic treats high-traffic edges as shorter paths.
"""
from __future__ import annotations

import logging
import pytest
import networkx as nx

from chaosrank_engine.graph.blast_radius import (
    compute_blast_radius,
    _resolve_betweenness_weights,
    _compute_betweenness,
    _normalize,
    DEFAULT_W_BC,
)


# Graph fixtures


def _linear_graph() -> nx.DiGraph:
    """A -> B -> C -> D  (linear chain, C is bridge between B and D)"""
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=10)
    G.add_edge("B", "C", weight=10)
    G.add_edge("C", "D", weight=10)
    return G


def _hub_graph() -> nx.DiGraph:
    """A, B, C, D all call payment-service (shallow-wide hub)"""
    G = nx.DiGraph()
    for caller in ("A", "B", "C", "D"):
        G.add_edge(caller, "payment-service", weight=100)
    return G


def _bridge_graph() -> nx.DiGraph:
    """Two clusters connected by a bridge node.

    cluster-1 (a1, a2) -> bridge -> cluster-2 (b1, b2)
    All traffic between clusters must flow through bridge.
    bridge should have high betweenness, low in-degree.
    """
    G = nx.DiGraph()
    G.add_edge("a1", "bridge", weight=50)
    G.add_edge("a2", "bridge", weight=50)
    G.add_edge("bridge", "b1", weight=50)
    G.add_edge("bridge", "b2", weight=50)
    # Direct edges within clusters (no betweenness on cluster nodes)
    G.add_edge("a1", "a2", weight=20)
    G.add_edge("b1", "b2", weight=20)
    return G


def _weighted_graph() -> nx.DiGraph:
    """A --(weight=1000)--> B --(weight=1)--> C
    High-frequency A->B should be treated as shorter path for betweenness.
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1000)
    G.add_edge("B", "C", weight=1)
    return G


class TestBackwardsCompatibility:
    """Verifies that existing behavior is preserved when betweenness is disabled."""
    def test_default_call_unchanged(self):
        G = _hub_graph()
        scores = compute_blast_radius(G)
        assert set(scores.keys()) == set(G.nodes())
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_w_pr_w_od_validation_still_fires(self):
        G = _hub_graph()
        with pytest.raises(ValueError, match="w_pr \\+ w_od must equal 1.0"):
            compute_blast_radius(G, w_pr=0.6, w_od=0.6)

    def test_w_bc_ignored_when_betweenness_off(self):
        """Passing w_bc without use_betweenness should not affect scores."""
        G = _hub_graph()
        scores_no_bc = compute_blast_radius(G)
        scores_with_bc = compute_blast_radius(G, w_bc=0.2)  # ignored
        assert scores_no_bc == scores_with_bc

    def test_empty_graph_returns_empty(self):
        assert compute_blast_radius(nx.DiGraph(), use_betweenness=True) == {}


class TestResolveBetweennessWeights:
    """Tests the weight validation and auto-adjustment logic for the tri-component score."""
    def test_auto_adjust_default_weights(self):
        """Default w_pr=0.5, w_od=0.5, w_bc=None → 0.4, 0.4, 0.2"""
        w_pr, w_od, w_bc = _resolve_betweenness_weights(0.5, 0.5, None)
        assert abs(w_pr + w_od + w_bc - 1.0) < 1e-6
        assert abs(w_bc - DEFAULT_W_BC) < 1e-6
        assert abs(w_pr - w_od) < 1e-6  # ratio preserved: equal in, equal out

    def test_auto_adjust_preserves_ratio(self):
        """w_pr=0.6, w_od=0.4 → ratio 3:2 should be preserved after scaling."""
        w_pr, w_od, w_bc = _resolve_betweenness_weights(0.6, 0.4, None)
        assert abs(w_pr + w_od + w_bc - 1.0) < 1e-6
        assert abs(w_bc - DEFAULT_W_BC) < 1e-6
        # Original ratio: 0.6/0.4 = 1.5; scaled: w_pr/w_od should still ≈ 1.5
        assert abs(w_pr / w_od - 1.5) < 1e-4

    def test_auto_adjust_emits_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="chaosrank.graph.blast_radius"):
            _resolve_betweenness_weights(0.5, 0.5, None)
        assert "Auto-adjusted weights" in caplog.text
        assert "w_bc not provided" in caplog.text

    def test_explicit_valid_weights_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="chaosrank.graph.blast_radius"):
            w_pr, w_od, w_bc = _resolve_betweenness_weights(0.4, 0.4, 0.2)
        assert "Auto-adjusted" not in caplog.text
        assert abs(w_pr + w_od + w_bc - 1.0) < 1e-6

    def test_explicit_invalid_sum_raises(self):
        with pytest.raises(ValueError, match="must equal 1.0"):
            _resolve_betweenness_weights(0.4, 0.4, 0.3)  # sum = 1.1

    def test_explicit_zero_w_bc_valid(self):
        """w_bc=0.0 explicitly is allowed — betweenness component just zeroes out."""
        w_pr, w_od, w_bc = _resolve_betweenness_weights(0.5, 0.5, 0.0)
        assert abs(w_pr + w_od + w_bc - 1.0) < 1e-6
        assert w_bc == 0.0

    def test_zero_pr_od_raises(self):
        with pytest.raises(ValueError, match="w_pr \\+ w_od must be positive"):
            _resolve_betweenness_weights(0.0, 0.0, None)


class TestComputeBetweenness:
    """Tests the core betweenness computation on various graph topologies."""
    def test_returns_score_per_node(self):
        G = _bridge_graph()
        bc = _compute_betweenness(G)
        assert set(bc.keys()) == set(G.nodes())

    def test_bridge_has_highest_betweenness(self):
        G = _bridge_graph()
        bc = _compute_betweenness(G)
        assert bc["bridge"] == max(bc.values())

    def test_zero_weight_edge_no_crash(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=0)
        G.add_edge("B", "C", weight=5)
        # Should not raise ZeroDivisionError
        bc = _compute_betweenness(G)
        assert set(bc.keys()) == {"A", "B", "C"}

    def test_caller_graph_not_mutated(self):
        G = _linear_graph()
        original_weights = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
        _compute_betweenness(G)
        for (u, v), w in original_weights.items():
            assert G[u][v]["weight"] == w

    def test_weight_inversion_high_freq_edge_shorter(self):
        """High-weight A->B edge should be inverted to low distance.

        In G: A --(1000)--> B --(1)--> C
        After inversion: A --(0.001)--> B --(1.0)--> C
        Shortest path A->C goes through B, making B a bridge.
        B should have higher betweenness than A or C.
        """
        G = _weighted_graph()
        bc = _compute_betweenness(G)
        # B sits on the only path A->C, so it should have betweenness > 0
        assert bc["B"] > bc["A"]


class TestNormalize:
    """Tests the result normalization logic for betweenness scores."""
    def test_spread_values_produce_0_and_1(self):
        scores = {"a": 0.0, "b": 0.5, "c": 1.0}
        result = _normalize(scores)
        assert result["a"] == pytest.approx(0.0)
        assert result["c"] == pytest.approx(1.0)

    def test_uniform_values_produce_0_5(self):
        scores = {"a": 3.0, "b": 3.0, "c": 3.0}
        result = _normalize(scores)
        assert all(v == pytest.approx(0.5) for v in result.values())

    def test_single_node_returns_0_5(self):
        result = _normalize({"only": 42.0})
        assert result["only"] == pytest.approx(0.5)

    def test_empty_dict_returns_empty(self):
        assert _normalize({}) == {}

    def test_all_values_in_0_1(self):
        import random

        random.seed(0)
        scores = {f"svc-{i}": random.uniform(0, 100) for i in range(20)}
        result = _normalize(scores)
        assert all(0.0 <= v <= 1.0 for v in result.values())


class TestComputeBlastRadiusWithBetweenness:
    """End-to-end tests for the integrated blast radius + betweenness scoring."""
    def test_returns_score_per_node(self):
        G = _bridge_graph()
        scores = compute_blast_radius(G, use_betweenness=True, w_pr=0.4, w_od=0.4, w_bc=0.2)
        assert set(scores.keys()) == set(G.nodes())

    def test_all_scores_in_0_1(self):
        G = _bridge_graph()
        scores = compute_blast_radius(G, use_betweenness=True, w_pr=0.4, w_od=0.4, w_bc=0.2)
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_bridge_ranks_higher_with_betweenness(self):
        """Bridge node has low in-degree (only a1,a2 call it) but sits on all
        cross-cluster paths. With betweenness enabled it should rank higher
        relative to cluster nodes than without it.
        """
        G = _bridge_graph()
        scores_off = compute_blast_radius(G, use_betweenness=False)
        scores_on = compute_blast_radius(G, use_betweenness=True, w_pr=0.4, w_od=0.4, w_bc=0.2)

        # bridge rank (position) should be better (lower index) with betweenness on
        ranked_off = sorted(scores_off, key=scores_off.get, reverse=True)
        ranked_on = sorted(scores_on, key=scores_on.get, reverse=True)
        assert ranked_on.index("bridge") <= ranked_off.index("bridge")

    def test_auto_adjust_weights_sum_to_1(self, caplog):
        G = _hub_graph()
        with caplog.at_level(logging.WARNING, logger="chaosrank.graph.blast_radius"):
            scores = compute_blast_radius(G, use_betweenness=True)
        assert "Auto-adjusted" in caplog.text
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_explicit_weights_no_warning(self, caplog):
        G = _hub_graph()
        with caplog.at_level(logging.WARNING, logger="chaosrank.graph.blast_radius"):
            compute_blast_radius(G, use_betweenness=True, w_pr=0.4, w_od=0.4, w_bc=0.2)
        assert "Auto-adjusted" not in caplog.text

    def test_invalid_explicit_weights_raise(self):
        G = _hub_graph()
        with pytest.raises(ValueError, match="must equal 1.0"):
            compute_blast_radius(G, use_betweenness=True, w_pr=0.5, w_od=0.4, w_bc=0.2)

    def test_w_pr_w_od_sum_check_not_fired_with_betweenness(self):
        """w_pr + w_od = 0.8 (not 1.0) is valid when w_bc=0.2 makes total 1.0."""
        G = _hub_graph()
        # Should NOT raise even though w_pr + w_od != 1.0
        scores = compute_blast_radius(G, use_betweenness=True, w_pr=0.4, w_od=0.4, w_bc=0.2)
        assert scores

    def test_single_node_no_crash(self):
        G = nx.DiGraph()
        G.add_node("lonely")
        scores = compute_blast_radius(G, use_betweenness=True, w_pr=0.4, w_od=0.4, w_bc=0.2)
        assert "lonely" in scores

    def test_scores_differ_from_no_betweenness(self):
        """Betweenness enabled should produce at least some different scores
        on a graph where betweenness actually matters (bridge graph).
        """
        G = _bridge_graph()
        off = compute_blast_radius(G, use_betweenness=False)
        on = compute_blast_radius(G, use_betweenness=True, w_pr=0.4, w_od=0.4, w_bc=0.2)
        assert off != on
