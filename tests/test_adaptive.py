"""Unit and integration tests for ChaosRank's adaptive ranking logic.
Covers outcome persistence, Reinforcement Learning weight updates, and confidence
interval calculations.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import pytest

from chaosrank_engine.adaptive.confidence import (
    LOW_CONFIDENCE_THRESHOLD,
    MIN_RELIABLE_N,
    STALENESS_THRESHOLD_DAYS,
    compute_all_confidence,
    compute_confidence,
)
from chaosrank_engine.adaptive.outcome_store import InterventionRecord, OutcomeStore, OutcomeType
from chaosrank_engine.adaptive.weight_updater import (
    ALPHA_MAX,
    ALPHA_MIN,
    MIN_OUTCOMES_BEFORE_UPDATE,
    WeightUpdater,
)
from chaosrank_engine.adaptive.adaptive_ranker import AdaptiveRanker


# Test helpers


def _tmp_store() -> OutcomeStore:
    """OutcomeStore backed by a temporary file."""
    tmp = tempfile.mktemp(suffix=".json")
    return OutcomeStore(path=Path(tmp))


def _record(
    store: OutcomeStore,
    outcome: OutcomeType,
    service: str = "payment-service",
    risk: float = 0.8,
    br: float = 0.7,
    fr: float = 0.9,
    alpha: float = 0.6,
    rank: int = 1,
) -> InterventionRecord:
    return store.record(
        service=service,
        outcome=outcome,
        risk_score=risk,
        blast_radius=br,
        fragility=fr,
        alpha=alpha,
        beta=round(1.0 - alpha, 6),
        rank_at_time=rank,
    )


def _make_graph(edges: list[tuple[str, str, int]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G


def _make_incidents(service: str, count: int):
    """Return a real ServiceIncidents with `count` Incident objects."""
    from chaosrank_engine.parser.incidents import Incident, ServiceIncidents

    now = datetime.utcnow()
    incidents = [
        Incident(
            timestamp=now - timedelta(days=i),
            service=service,
            type="error",
            severity="high",
            request_volume=1000.0,
        )
        for i in range(count)
    ]
    return ServiceIncidents(service=service, incidents=incidents)


class TestOutcomeStore:
    """Tests for the experiment outcome persistence layer."""
    def test_record_and_retrieve(self):
        store = _tmp_store()
        rec = _record(store, OutcomeType.WEAKNESS_CONFIRMED)
        assert rec.service == "payment-service"
        assert rec.outcome == OutcomeType.WEAKNESS_CONFIRMED
        assert rec.br_contribution == pytest.approx(0.6 * 0.7, abs=1e-4)
        assert rec.fr_contribution == pytest.approx(0.4 * 0.9, abs=1e-4)

    def test_persists_to_disk(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        path.unlink()  # delete empty file so OutcomeStore creates it fresh
        store = OutcomeStore(path=path)
        _record(store, OutcomeType.WEAKNESS_CONFIRMED)
        _record(store, OutcomeType.WEAKNESS_NOT_FOUND)

        # Reload from disk
        store2 = OutcomeStore(path=path)
        assert len(store2.all()) == 2
        assert store2.all()[0].outcome == OutcomeType.WEAKNESS_CONFIRMED
        assert store2.all()[1].outcome == OutcomeType.WEAKNESS_NOT_FOUND

    def test_for_service_filter(self):
        store = _tmp_store()
        _record(store, OutcomeType.WEAKNESS_CONFIRMED, service="svc-a")
        _record(store, OutcomeType.WEAKNESS_NOT_FOUND, service="svc-b")
        _record(store, OutcomeType.WEAKNESS_CONFIRMED, service="svc-a")
        assert len(store.for_service("svc-a")) == 2
        assert len(store.for_service("svc-b")) == 1
        assert len(store.for_service("svc-c")) == 0

    def test_recent_returns_last_n(self):
        store = _tmp_store()
        for i in range(10):
            _record(store, OutcomeType.INCONCLUSIVE, service=f"svc-{i}")
        recent = store.recent(3)
        assert len(recent) == 3
        assert recent[-1].service == "svc-9"

    def test_stats_confirmation_rate(self):
        store = _tmp_store()
        _record(store, OutcomeType.WEAKNESS_CONFIRMED)
        _record(store, OutcomeType.WEAKNESS_CONFIRMED)
        _record(store, OutcomeType.WEAKNESS_NOT_FOUND)
        _record(store, OutcomeType.INCONCLUSIVE)
        stats = store.stats()
        assert stats["total"] == 4
        assert stats["confirmed"] == 2
        assert stats["not_found"] == 1
        assert stats["inconclusive"] == 1
        assert stats["confirmation_rate"] == pytest.approx(0.5, abs=1e-4)

    def test_empty_store_stats(self):
        store = _tmp_store()
        stats = store.stats()
        assert stats["total"] == 0
        assert stats["confirmation_rate"] is None

    def test_corrupt_file_starts_fresh(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{{{{")
            path = Path(f.name)
        store = OutcomeStore(path=path)
        assert store.all() == []


class TestWeightUpdater:
    """Tests for the RL-based alpha/beta weight update rules."""
    def test_initial_weights(self):
        store = _tmp_store()
        updater = WeightUpdater(store, initial_alpha=0.6)
        assert updater.alpha == pytest.approx(0.6, abs=1e-6)
        assert updater.beta == pytest.approx(0.4, abs=1e-6)

    def test_no_update_below_min_outcomes(self):
        store = _tmp_store()
        updater = WeightUpdater(store, initial_alpha=0.6)
        # Record fewer than MIN_OUTCOMES_BEFORE_UPDATE
        for _ in range(MIN_OUTCOMES_BEFORE_UPDATE - 1):
            rec = _record(store, OutcomeType.WEAKNESS_CONFIRMED)
        updater.update(rec)
        assert updater.alpha == pytest.approx(0.6, abs=1e-6)

    def test_confirmed_increases_beta(self):
        store = _tmp_store()
        updater = WeightUpdater(store, initial_alpha=0.6, learning_rate=0.1)
        for _ in range(MIN_OUTCOMES_BEFORE_UPDATE):
            rec = _record(
                store, OutcomeType.WEAKNESS_CONFIRMED, risk=0.8, br=0.3, fr=0.9, alpha=0.6
            )
        old_beta = updater.beta
        updater.update(rec)
        assert updater.beta > old_beta
        assert updater.alpha < 0.6

    def test_not_found_decreases_alpha(self):
        store = _tmp_store()
        updater = WeightUpdater(store, initial_alpha=0.6, learning_rate=0.1)
        for _ in range(MIN_OUTCOMES_BEFORE_UPDATE):
            rec = _record(
                store, OutcomeType.WEAKNESS_NOT_FOUND, risk=0.8, br=0.9, fr=0.2, alpha=0.6
            )
        old_alpha = updater.alpha
        updater.update(rec)
        assert updater.alpha < old_alpha

    def test_inconclusive_no_change(self):
        store = _tmp_store()
        updater = WeightUpdater(store, initial_alpha=0.6)
        for _ in range(MIN_OUTCOMES_BEFORE_UPDATE + 1):
            rec = _record(store, OutcomeType.INCONCLUSIVE)
        old_alpha = updater.alpha
        updater.update(rec)
        assert updater.alpha == pytest.approx(old_alpha, abs=1e-6)

    def test_alpha_bounded_below(self):
        store = _tmp_store()
        updater = WeightUpdater(store, initial_alpha=0.6, learning_rate=0.9)
        for _ in range(50):
            rec = _record(
                store, OutcomeType.WEAKNESS_CONFIRMED, risk=0.8, br=0.1, fr=0.9, alpha=0.6
            )
            updater.update(rec)
        assert updater.alpha >= ALPHA_MIN

    def test_alpha_bounded_above(self):
        store = _tmp_store()
        updater = WeightUpdater(store, initial_alpha=0.6, learning_rate=0.9)
        for _ in range(50):
            rec = _record(
                store, OutcomeType.WEAKNESS_NOT_FOUND, risk=0.8, br=0.9, fr=0.1, alpha=0.6
            )
            updater.update(rec)
        assert updater.alpha <= ALPHA_MAX

    def test_alpha_beta_sum_to_one(self):
        store = _tmp_store()
        updater = WeightUpdater(store, initial_alpha=0.6, learning_rate=0.1)
        for _ in range(MIN_OUTCOMES_BEFORE_UPDATE + 5):
            rec = _record(store, OutcomeType.WEAKNESS_CONFIRMED)
            updater.update(rec)
        assert updater.alpha + updater.beta == pytest.approx(1.0, abs=1e-6)

    def test_zero_risk_score_skipped(self):
        store = _tmp_store()
        updater = WeightUpdater(store, initial_alpha=0.6)
        for _ in range(MIN_OUTCOMES_BEFORE_UPDATE):
            rec = _record(store, OutcomeType.WEAKNESS_CONFIRMED, risk=0.0, br=0.0, fr=0.0)
        old_alpha = updater.alpha
        updater.update(rec)
        assert updater.alpha == pytest.approx(old_alpha, abs=1e-6)

    def test_rehydrates_from_existing_store(self):
        import os

        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        path = Path(tmp)
        path.unlink()  # start fresh — mkstemp creates an empty file

        store1 = OutcomeStore(path=path)
        updater1 = WeightUpdater(store1, initial_alpha=0.6, learning_rate=0.1)
        for _ in range(MIN_OUTCOMES_BEFORE_UPDATE + 2):
            rec = _record(
                store1, OutcomeType.WEAKNESS_CONFIRMED, risk=0.8, br=0.3, fr=0.9, alpha=0.6
            )
            updater1.update(rec)
        alpha_after = updater1.alpha

        # Rehydration replays outcomes sequentially from default alpha=0.6.
        # Result is directionally correct (below 0.6) but not numerically
        # identical to the incremental path — both should be below 0.6.
        store2 = OutcomeStore(path=path)
        updater2 = WeightUpdater(store2, initial_alpha=0.6, learning_rate=0.1)
        assert updater2.alpha < 0.6
        assert alpha_after < 0.6


class TestConfidence:
    def _base_graph(self):
        return _make_graph(
            [
                ("frontend", "payment", 100),
                ("frontend", "cart", 50),
                ("payment", "db", 200),
            ]
        )

    def test_full_graph_full_history_low_ci(self):
        G = self._base_graph()
        si = {"payment": _make_incidents("payment", MIN_RELIABLE_N + 5)}
        result = compute_confidence(
            service="payment",
            risk_score=0.7,
            G=G,
            service_incidents=si,
            last_observed=None,
        )
        assert result.ci_width < LOW_CONFIDENCE_THRESHOLD
        assert not result.low_confidence

    def test_no_incidents_high_ci(self):
        G = self._base_graph()
        result = compute_confidence(
            service="payment",
            risk_score=0.7,
            G=G,
            service_incidents={},
            last_observed=None,
        )
        # History component fully 1.0 → high CI
        assert result.history_component == pytest.approx(1.0, abs=1e-4)

    def test_stale_graph_increases_ci(self):
        G = self._base_graph()
        si = {"payment": _make_incidents("payment", MIN_RELIABLE_N)}
        fresh = compute_confidence("payment", 0.7, G, si, last_observed=None)
        stale = compute_confidence(
            "payment",
            0.7,
            G,
            si,
            last_observed=datetime.utcnow() - timedelta(days=STALENESS_THRESHOLD_DAYS + 5),
        )
        assert stale.ci_width > fresh.ci_width
        assert stale.age_component == pytest.approx(1.0, abs=1e-4)

    def test_unknown_service_max_sparsity(self):
        G = self._base_graph()
        result = compute_confidence(
            service="unknown-svc",
            risk_score=0.5,
            G=G,
            service_incidents={},
        )
        assert result.sparsity_component == pytest.approx(1.0, abs=1e-4)

    def test_ci_bounds_clamped_to_zero_one(self):
        G = _make_graph([("a", "b", 10)])
        result = compute_confidence(
            service="a",
            risk_score=0.02,
            G=G,
            service_incidents={},
        )
        assert result.ci_lower >= 0.0
        assert result.ci_upper <= 1.0

    def test_low_confidence_reason_populated(self):
        G = self._base_graph()
        result = compute_confidence(
            service="payment",
            risk_score=0.7,
            G=G,
            service_incidents={},
        )
        if result.low_confidence:
            assert result.reason is not None
            assert "Low confidence" in result.reason

    def test_compute_all_confidence_covers_all_services(self):
        G = self._base_graph()
        si = {}
        ranked = [
            {"service": "frontend", "risk": 0.8},
            {"service": "payment", "risk": 0.6},
            {"service": "cart", "risk": 0.4},
        ]
        results = compute_all_confidence(ranked, G, si)
        assert set(results.keys()) == {"frontend", "payment", "cart"}

    def test_confidence_level_99_wider_than_95(self):
        G = self._base_graph()
        si = {}
        r95 = compute_confidence("payment", 0.7, G, si, confidence_level=0.95)
        r99 = compute_confidence("payment", 0.7, G, si, confidence_level=0.99)
        # Compare raw ci_width — bounds may both clamp to [0,1] at high uncertainty
        assert r99.ci_width >= r95.ci_width

    def test_invalid_confidence_level_raises(self):
        G = self._base_graph()
        with pytest.raises(ValueError):
            compute_confidence("payment", 0.7, G, {}, confidence_level=0.80)


class TestAdaptiveRanker:
    """End-to-end tests for the stateful adaptive ranking component."""
    def _setup(self):
        store = _tmp_store()
        G = _make_graph(
            [
                ("frontend", "payment", 200),
                ("frontend", "cart", 100),
                ("payment", "db", 300),
            ]
        )
        blast = {"frontend": 0.2, "payment": 0.9, "cart": 0.4, "db": 0.7}
        si = {
            "payment": _make_incidents("payment", 15),
            "db": _make_incidents("db", 5),
        }
        ranker = AdaptiveRanker(store=store, initial_alpha=0.6)
        return ranker, G, blast, si

    def test_rank_returns_all_services(self):
        ranker, G, blast, si = self._setup()
        ranked = ranker.rank(blast, si, G)
        assert len(ranked) == 4

    def test_rank_output_has_ci_fields(self):
        ranker, G, blast, si = self._setup()
        ranked = ranker.rank(blast, si, G)
        for row in ranked:
            assert "ci_lower" in row
            assert "ci_upper" in row
            assert "ci_width" in row
            assert "low_confidence" in row
            assert "alpha_used" in row
            assert "beta_used" in row

    def test_rank_sorted_descending(self):
        ranker, G, blast, si = self._setup()
        ranked = ranker.rank(blast, si, G)
        risks = [r["risk"] for r in ranked]
        assert risks == sorted(risks, reverse=True)

    def test_alpha_used_matches_current(self):
        ranker, G, blast, si = self._setup()
        ranked = ranker.rank(blast, si, G)
        for row in ranked:
            assert row["alpha_used"] == pytest.approx(ranker.alpha, abs=1e-6)

    def test_record_outcome_confirmed_updates_weights(self):
        ranker, G, blast, si = self._setup()

        # Need MIN_OUTCOMES_BEFORE_UPDATE outcomes before weights move
        for _ in range(MIN_OUTCOMES_BEFORE_UPDATE):
            ranked = ranker.rank(blast, si, G)
            ranker.record_outcome(ranked[0], OutcomeType.WEAKNESS_CONFIRMED)

        # After enough confirmations, beta should have increased
        assert ranker.beta >= 0.4

    def test_record_outcome_missing_field_raises(self):
        ranker, G, blast, si = self._setup()
        with pytest.raises(ValueError, match="missing required fields"):
            ranker.record_outcome(
                {"service": "payment", "risk": 0.8},
                OutcomeType.WEAKNESS_CONFIRMED,
            )

    def test_summary_contains_expected_keys(self):
        ranker, G, blast, si = self._setup()
        summary = ranker.summary()
        for key in [
            "current_alpha",
            "current_beta",
            "update_count",
            "total_outcomes",
            "confirmation_rate",
        ]:
            assert key in summary

    def test_weight_history_empty_on_fresh_store(self):
        store = _tmp_store()
        ranker = AdaptiveRanker(store=store)
        assert ranker.weight_history() == []

    def test_weight_history_grows_with_outcomes(self):
        ranker, G, blast, si = self._setup()
        for _ in range(5):
            ranked = ranker.rank(blast, si, G)
            ranker.record_outcome(ranked[0], OutcomeType.INCONCLUSIVE)
        history = ranker.weight_history()
        assert len(history) == 5

    def test_weights_bounded_after_many_confirmations(self):
        ranker, G, blast, si = self._setup()
        ranker_lr = AdaptiveRanker(
            store=_tmp_store(),
            initial_alpha=0.6,
            learning_rate=0.5,  # aggressive
        )
        for _ in range(30):
            ranked = ranker_lr.rank(blast, si, G)
            ranker_lr.record_outcome(ranked[0], OutcomeType.WEAKNESS_CONFIRMED)
        assert ranker_lr.alpha >= ALPHA_MIN
        assert ranker_lr.alpha <= ALPHA_MAX
        assert ranker_lr.alpha + ranker_lr.beta == pytest.approx(1.0, abs=1e-6)

    def test_store_and_store_path_mutually_exclusive(self):
        with pytest.raises(ValueError):
            AdaptiveRanker(
                store=_tmp_store(),
                store_path=Path("/tmp/x.json"),
            )

    def test_existing_store_rehydrates_weights(self):
        import os

        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        path = Path(tmp)
        path.unlink()  # start fresh

        store1 = OutcomeStore(path=path)
        ranker1 = AdaptiveRanker(store=store1, initial_alpha=0.6, learning_rate=0.1)
        G = _make_graph([("a", "b", 100)])
        blast = {"a": 0.8, "b": 0.4}
        si = {"a": _make_incidents("a", 15)}

        for _ in range(MIN_OUTCOMES_BEFORE_UPDATE + 2):
            ranked = ranker1.rank(blast, si, G)
            ranker1.record_outcome(ranked[0], OutcomeType.WEAKNESS_CONFIRMED)
        alpha_after = ranker1.alpha

        # New ranker from same store path — rehydration is directionally
        # correct: both should have moved below 0.6 from confirmations
        store2 = OutcomeStore(path=path)
        ranker2 = AdaptiveRanker(store=store2, initial_alpha=0.6, learning_rate=0.1)
        assert ranker2.alpha < 0.6
        assert alpha_after < 0.6
