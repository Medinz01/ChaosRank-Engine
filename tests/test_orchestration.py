"""Unit and integration tests for ChaosRank's orchestration logic.
Covers agent snapshots, central merging, incremental graph state, and
delta-triggered streaming scoring.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import networkx as nx
import pytest

from chaosrank_engine.orchestration.agent import (
    EdgeObservation,
    LocalGraphSnapshot,
)
from chaosrank_engine.orchestration.incremental import (
    IncrementalGraphState,
    IncrementalUpdateResult,
)
from chaosrank_engine.orchestration.merger import (
    CanonicalEdge,
    CentralMerger,
    MergeResult,
)
from chaosrank_engine.orchestration.streaming import (
    RecomputeStrategy,
    StreamingScorer,
)


# Test helpers


def _snapshot(
    agent_id: str,
    edges: list[tuple[str, str, float, float, str]] = None,
    total_spans: int = 1000,
    observed_at: datetime = None,
) -> LocalGraphSnapshot:
    """
    Build a LocalGraphSnapshot.
    edges: list of (source, target, weight, confidence, edge_type)
    """
    obs = [
        EdgeObservation(
            source=src,
            target=tgt,
            weight=w,
            confidence=c,
            edge_type=et,
        )
        for src, tgt, w, c, et in (edges or [("a", "b", 100.0, 0.1, "sync")])
    ]
    return LocalGraphSnapshot(
        agent_id=agent_id,
        observed_at=observed_at or datetime.utcnow(),
        total_spans=total_spans,
        edges=obs,
    )


def _merge_result(
    edges: list[tuple[str, str, float]] = None,
) -> MergeResult:
    """
    Build a MergeResult directly without going through the merger.
    edges: list of (source, target, canonical_weight)
    """
    canonical = [
        CanonicalEdge(
            source=src,
            target=tgt,
            canonical_weight=w,
            contributing_agents=["test-agent"],
            total_confidence=0.1,
            single_agent=True,
            edge_type="sync",
        )
        for src, tgt, w in (edges or [("a", "b", 100.0)])
    ]
    G = nx.DiGraph()
    for e in canonical:
        G.add_edge(e.source, e.target, weight=e.canonical_weight, edge_type=e.edge_type)
    return MergeResult(
        graph=G,
        canonical_edges=canonical,
        agent_count=1,
        snapshot_times={"test-agent": datetime.utcnow()},
    )


class TestLocalGraphSnapshot:
    """Tests for agent graph snapshot serialization and model conversion."""
    def test_to_graph_builds_digraph(self):
        snap = _snapshot("agent-a", [("svc-a", "svc-b", 100.0, 0.1, "sync")])
        G = snap.to_graph()
        assert G.has_edge("svc-a", "svc-b")
        assert G["svc-a"]["svc-b"]["weight"] == 100.0

    def test_to_graph_preserves_edge_type(self):
        snap = _snapshot("agent-a", [("prod", "cons", 50.0, 0.05, "async")])
        G = snap.to_graph()
        assert G["prod"]["cons"]["edge_type"] == "async"

    def test_snapshot_attributes(self):
        now = datetime.utcnow()
        snap = _snapshot("region-1", total_spans=5000, observed_at=now)
        assert snap.agent_id == "region-1"
        assert snap.total_spans == 5000
        assert snap.observed_at == now


class TestCentralMerger:
    """Tests for the weighted confidence merging of multiple agent snapshots."""
    def test_ingest_and_merge_single_agent(self):
        merger = CentralMerger(min_call_frequency=10)
        snapshot = _snapshot("agent-a", [("svc-a", "svc-b", 100.0, 0.1, "sync")])
        merger.ingest(snapshot)
        result = merger.merge()

        assert result.graph.has_edge("svc-a", "svc-b")
        assert result.agent_count == 1

    def test_merge_empty_raises(self):
        merger = CentralMerger()
        with pytest.raises(ValueError, match="No snapshots"):
            merger.merge()

    def test_ready_false_below_min_agents(self):
        merger = CentralMerger(min_agents=2)
        merger.ingest(_snapshot("a"))
        assert not merger.ready()

    def test_ready_true_at_min_agents(self):
        merger = CentralMerger(min_agents=2)
        merger.ingest(_snapshot("a"))
        merger.ingest(_snapshot("b"))
        assert merger.ready()

    def test_duplicate_agent_replaced(self):
        merger = CentralMerger(min_call_frequency=1)
        merger.ingest(_snapshot("agent-a", [("x", "y", 100.0, 0.1, "sync")]))
        merger.ingest(_snapshot("agent-a", [("x", "y", 200.0, 0.2, "sync")]))
        result = merger.merge()
        # Only one agent — latest snapshot wins
        assert result.agent_count == 1
        weight = result.graph["x"]["y"]["weight"]
        assert weight == pytest.approx(200.0, abs=1e-3)

    def test_weighted_confidence_merge_two_agents(self):
        """
        Agent A: weight=100, confidence=0.1
        Agent B: weight=200, confidence=0.4
        canonical = (100*0.1 + 200*0.4) / (0.1 + 0.4) = 90/0.5 = 180.0
        """
        merger = CentralMerger(min_call_frequency=10)
        snap_a = _snapshot("agent-a", [("svc", "db", 100.0, 0.1, "sync")])
        snap_b = _snapshot("agent-b", [("svc", "db", 200.0, 0.4, "sync")])
        merger.ingest(snap_a)
        merger.ingest(snap_b)
        result = merger.merge()

        canonical_w = result.graph["svc"]["db"]["weight"]
        assert canonical_w == pytest.approx(180.0, abs=0.01)

    def test_min_call_frequency_filters_low_weight(self):
        merger = CentralMerger(min_call_frequency=50)
        snap = _snapshot("agent-a", [("a", "b", 10.0, 0.01, "sync")])
        merger.ingest(snap)
        result = merger.merge()
        # canonical_weight = 10.0 < 50 → edge should be filtered out
        assert not result.graph.has_edge("a", "b")

    def test_single_agent_edge_flagged(self):
        merger = CentralMerger(min_call_frequency=1, single_agent_warn=True)
        merger.ingest(_snapshot("agent-a"))
        result = merger.merge()
        assert result.canonical_edges[0].single_agent is True
        assert any("one agent" in w for w in result.warnings)

    def test_corroborated_edge_not_single_agent(self):
        merger = CentralMerger(min_call_frequency=1)
        merger.ingest(_snapshot("agent-a", [("x", "y", 100.0, 0.1, "sync")]))
        merger.ingest(_snapshot("agent-b", [("x", "y", 120.0, 0.15, "sync")]))
        result = merger.merge()
        edge = next(e for e in result.canonical_edges if e.source == "x")
        assert edge.single_agent is False
        assert len(edge.contributing_agents) == 2

    def test_edge_type_majority_vote_async(self):
        merger = CentralMerger(min_call_frequency=1)
        # 2 async, 1 sync → async wins
        merger.ingest(_snapshot("a", [("p", "c", 100.0, 0.1, "async")]))
        merger.ingest(_snapshot("b", [("p", "c", 100.0, 0.1, "async")]))
        merger.ingest(_snapshot("c", [("p", "c", 100.0, 0.1, "sync")]))
        result = merger.merge()
        edge = result.graph["p"]["c"]
        assert edge["edge_type"] == "async"

    def test_clear_resets_snapshots(self):
        merger = CentralMerger(min_call_frequency=1)
        merger.ingest(_snapshot("agent-a"))
        merger.clear()
        assert merger.agent_count == 0
        with pytest.raises(ValueError):
            merger.merge()

    def test_merge_result_counts(self):
        merger = CentralMerger(min_call_frequency=1)
        merger.ingest(
            _snapshot("a", [("x", "y", 50.0, 0.05, "sync"), ("y", "z", 80.0, 0.08, "sync")])
        )
        result = merger.merge()
        assert result.corroborated_edge_count == 0
        assert result.single_agent_edge_count == 2


class TestIncrementalGraphState:
    """Tests for stateful graph evolution, EMA updates, and staleness pruning."""
    def test_apply_adds_new_edges(self):
        state = IncrementalGraphState()
        result = state.apply(_merge_result([("a", "b", 100.0)]))
        assert result.graph.has_edge("a", "b")
        assert len(result.edges_added) == 1
        assert len(result.edges_updated) == 0

    def test_apply_ema_updates_existing_edge(self):
        state = IncrementalGraphState(ema_alpha=0.5)
        state.apply(_merge_result([("a", "b", 100.0)]))
        result = state.apply(_merge_result([("a", "b", 200.0)]))
        # EMA: 0.5 * 200 + 0.5 * 100 = 150
        weight = result.graph["a"]["b"]["weight"]
        assert weight == pytest.approx(150.0, abs=0.1)
        assert len(result.edges_updated) == 1
        assert len(result.edges_added) == 0

    def test_apply_attenuates_absent_edges(self):
        state = IncrementalGraphState(staleness_factor=0.5, min_weight=1.0)
        state.apply(_merge_result([("a", "b", 100.0), ("x", "y", 100.0)]))
        # Second merge only has (a,b) — (x,y) should be attenuated
        result = state.apply(_merge_result([("a", "b", 100.0)]))
        assert ("x", "y") in result.edges_attenuated
        # Weight should have decreased
        xy_state = state.edge_state("x", "y")
        assert xy_state is not None
        assert xy_state.weight < 100.0

    def test_apply_prunes_below_min_weight(self):
        state = IncrementalGraphState(staleness_factor=0.5, min_weight=50.0)
        state.apply(_merge_result([("a", "b", 100.0), ("x", "y", 51.0)]))
        # Backdate (x,y) so stale_days=2 → weight = 51 * 0.5^2 = 12.75 < 50
        from chaosrank_engine.orchestration.incremental import EdgeState

        key = ("x", "y")
        es = state._edge_states[key]
        state._edge_states[key] = EdgeState(
            source=es.source,
            target=es.target,
            weight=es.weight,
            edge_type=es.edge_type,
            last_observed=datetime.utcnow() - timedelta(days=2),
            first_observed=es.first_observed,
            update_count=es.update_count,
        )
        result = state.apply(_merge_result([("a", "b", 100.0)]))
        assert ("x", "y") in result.edges_pruned
        assert not result.graph.has_edge("x", "y")

    def test_stale_edges_detection(self):
        state = IncrementalGraphState()
        state.apply(_merge_result([("a", "b", 100.0)]))
        # Manually backdate last_observed
        key = ("a", "b")
        edge_state = state._edge_states[key]
        state._edge_states[key] = type(edge_state)(
            source=edge_state.source,
            target=edge_state.target,
            weight=edge_state.weight,
            edge_type=edge_state.edge_type,
            last_observed=datetime.utcnow() - timedelta(days=10),
            first_observed=edge_state.first_observed,
            update_count=edge_state.update_count,
        )
        stale = state.stale_edges(threshold_days=7.0)
        assert ("a", "b") in stale

    def test_current_graph_without_update(self):
        state = IncrementalGraphState()
        state.apply(_merge_result([("a", "b", 100.0)]))
        G = state.current_graph()
        assert G.has_edge("a", "b")

    def test_ema_alpha_validation(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            IncrementalGraphState(ema_alpha=0.0)
        with pytest.raises(ValueError, match="ema_alpha"):
            IncrementalGraphState(ema_alpha=1.5)

    def test_staleness_factor_validation(self):
        with pytest.raises(ValueError, match="staleness_factor"):
            IncrementalGraphState(staleness_factor=0.0)

    def test_multiple_ema_updates_converge(self):
        """EMA should converge toward new value over repeated observations."""
        state = IncrementalGraphState(ema_alpha=0.5)
        state.apply(_merge_result([("a", "b", 100.0)]))
        for _ in range(10):
            state.apply(_merge_result([("a", "b", 200.0)]))
        weight = state.edge_state("a", "b").weight
        # After 10 updates with alpha=0.5, should be close to 200
        assert weight > 180.0


class TestStreamingScorer:
    """Tests for delta-triggered partial recomputation of blast radius scores."""
    def _simple_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_edge("frontend", "payment", weight=200.0, edge_type="sync")
        G.add_edge("frontend", "cart", weight=100.0, edge_type="sync")
        G.add_edge("payment", "db", weight=300.0, edge_type="sync")
        return G

    def _update_with_changes(
        self,
        added: list[tuple[str, str]] = None,
        pruned: list[tuple[str, str]] = None,
        updated: list[tuple[str, str]] = None,
        attenuated: list[tuple[str, str]] = None,
    ) -> IncrementalUpdateResult:
        G = self._simple_graph()
        return IncrementalUpdateResult(
            graph=G,
            edges_added=added or [],
            edges_updated=updated or [],
            edges_pruned=pruned or [],
            edges_attenuated=attenuated or [],
        )

    def test_first_update_always_full(self):
        scorer = StreamingScorer()
        G = self._simple_graph()
        update = self._update_with_changes()
        result = scorer.update(G, update)
        assert result.strategy_used == RecomputeStrategy.FULL
        assert len(result.scores) > 0

    def test_empty_graph_returns_empty(self):
        scorer = StreamingScorer()
        G = nx.DiGraph()
        update = IncrementalUpdateResult(
            graph=G,
            edges_added=[],
            edges_updated=[],
            edges_pruned=[],
            edges_attenuated=[],
        )
        result = scorer.update(G, update)
        assert result.scores == {}

    def test_topology_change_triggers_neighborhood(self):
        scorer = StreamingScorer()
        G = self._simple_graph()
        update = self._update_with_changes()
        scorer.update(G, update)  # prime cache

        # Now add a new edge
        G2 = G.copy()
        G2.add_edge("cart", "db", weight=50.0, edge_type="sync")
        update2 = self._update_with_changes(
            added=[("cart", "db")],
        )
        result2 = scorer.update(G2, update2)
        assert result2.strategy_used in (RecomputeStrategy.NEIGHBORHOOD, RecomputeStrategy.FULL)

    def test_large_topology_change_triggers_full(self):
        scorer = StreamingScorer(full_recompute_threshold=0.1)
        G = self._simple_graph()
        update = self._update_with_changes()
        scorer.update(G, update)  # prime cache

        # Add many edges (>10% of total → full)
        G2 = G.copy()
        for i in range(5):
            G2.add_edge(f"new-{i}", "db", weight=10.0, edge_type="sync")
        update2 = self._update_with_changes(
            added=[(f"new-{i}", "db") for i in range(5)],
        )
        result2 = scorer.update(G2, update2)
        assert result2.strategy_used == RecomputeStrategy.FULL

    def test_weight_only_change_uses_delta(self):
        scorer = StreamingScorer(min_delta=1000.0)  # high threshold
        G = self._simple_graph()
        update = self._update_with_changes()
        scorer.update(G, update)  # prime cache

        # Only weight updates, small delta
        update2 = self._update_with_changes(
            updated=[("frontend", "payment")],
        )
        result2 = scorer.update(G, update2)
        assert result2.strategy_used == RecomputeStrategy.DELTA

    def test_delta_skip_returns_cached_scores(self):
        scorer = StreamingScorer(min_delta=10000.0)  # very high threshold
        G = self._simple_graph()
        update = self._update_with_changes()
        first = scorer.update(G, update)

        update2 = self._update_with_changes(
            updated=[("frontend", "payment")],
        )
        second = scorer.update(G, update2)
        assert second.updated_nodes == []
        assert second.scores == first.scores

    def test_force_full_recompute(self):
        scorer = StreamingScorer()
        G = self._simple_graph()
        result = scorer.force_full_recompute(G)
        assert result.strategy_used == RecomputeStrategy.FULL
        assert len(result.scores) == G.number_of_nodes()

    def test_scores_all_nodes_covered(self):
        scorer = StreamingScorer()
        G = self._simple_graph()
        update = self._update_with_changes()
        result = scorer.update(G, update)
        assert set(result.scores.keys()) == set(G.nodes())

    def test_scores_normalized_zero_to_one(self):
        scorer = StreamingScorer()
        G = self._simple_graph()
        update = self._update_with_changes()
        result = scorer.update(G, update)
        for score in result.scores.values():
            assert 0.0 <= score <= 1.0

    def test_cached_scores_property(self):
        scorer = StreamingScorer()
        G = self._simple_graph()
        update = self._update_with_changes()
        result = scorer.update(G, update)
        assert scorer.cached_scores == result.scores

    def test_change_magnitude_reported(self):
        scorer = StreamingScorer()
        G = self._simple_graph()
        update = self._update_with_changes(added=[("a", "b")])
        result = scorer.update(G, update)
        assert 0.0 <= result.change_magnitude <= 1.0
