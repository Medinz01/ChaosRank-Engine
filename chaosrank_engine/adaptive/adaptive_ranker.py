"""Adaptive ranker with Reinforcement Learning.
Adjusts alpha/beta weights dynamically based on experiment outcomes and
appends confidence intervals to risk scores.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import networkx as nx

from chaosrank_engine.adaptive.confidence import compute_all_confidence
from chaosrank_engine.adaptive.outcome_store import OutcomeStore, OutcomeType, DEFAULT_STORE_PATH
from chaosrank_engine.adaptive.weight_updater import WeightUpdater
from chaosrank_engine.parser.incidents import ServiceIncidents
from chaosrank_engine.scorer.ranker import rank_services

logger = logging.getLogger(__name__)


class AdaptiveRanker:
    """Stateful ranker that learns optimal weights from experiment outcomes."""

    def __init__(
        self,
        store: OutcomeStore | None = None,
        store_path: Path | None = None,
        initial_alpha: float = 0.6,
        learning_rate: float = 0.05,
        outcome_window: int = 20,
        confidence_level: float = 0.95,
    ) -> None:
        if store is not None and store_path is not None:
            raise ValueError("Provide either store or store_path, not both")

        self.store = store or OutcomeStore(path=store_path or DEFAULT_STORE_PATH)
        self.updater = WeightUpdater(
            store=self.store,
            initial_alpha=initial_alpha,
            learning_rate=learning_rate,
            outcome_window=outcome_window,
        )
        self.confidence_level = confidence_level

        logger.info(
            "AdaptiveRanker initialised — alpha=%.4f beta=%.4f (from %d recorded outcomes)",
            self.updater.alpha,
            self.updater.beta,
            len(self.store.all()),
        )

    # Primary interface

    def rank(
        self,
        blast_radius: dict[str, float],
        service_incidents: dict[str, ServiceIncidents],
        G: nx.DiGraph,
        decay_lambda: float = 0.10,
        base_window: float = 5.0,
        last_observed: datetime | None = None,
    ) -> list[dict]:
        """Rank services using live weights with confidence intervals."""
        alpha = self.updater.alpha
        beta = self.updater.beta

        # Base ranking using live weights
        ranked = rank_services(
            blast_radius=blast_radius,
            service_incidents=service_incidents,
            alpha=alpha,
            beta=beta,
            decay_lambda=decay_lambda,
            base_window=base_window,
        )

        # Confidence intervals for all ranked services
        ci_results = compute_all_confidence(
            ranked=ranked,
            G=G,
            service_incidents=service_incidents,
            last_observed=last_observed,
            confidence_level=self.confidence_level,
        )

        # Merge CI results into ranked output
        for row in ranked:
            ci = ci_results.get(row["service"])
            if ci:
                row["alpha_used"] = alpha
                row["beta_used"] = beta
                row["ci_lower"] = ci.ci_lower
                row["ci_upper"] = ci.ci_upper
                row["ci_width"] = ci.ci_width
                row["low_confidence"] = ci.low_confidence
                row["confidence_note"] = ci.reason or ""
            else:
                row["alpha_used"] = alpha
                row["beta_used"] = beta
                row["ci_lower"] = None
                row["ci_upper"] = None
                row["ci_width"] = None
                row["low_confidence"] = None
                row["confidence_note"] = ""

        weight_state = self.updater.state()
        logger.info(
            "Adaptive ranking complete — alpha=%.4f beta=%.4f update_count=%d confirmation_rate=%s",
            alpha,
            beta,
            weight_state.update_count,
            f"{weight_state.confirmation_rate:.2%}"
            if weight_state.confirmation_rate is not None
            else "n/a",
        )

        return ranked

    def record_outcome(
        self,
        ranked_row: dict,
        outcome: OutcomeType,
        graph_state_hash: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Record an experiment outcome and trigger weight updates."""
        required = {
            "service",
            "risk",
            "blast_radius",
            "fragility",
            "rank",
            "alpha_used",
            "beta_used",
        }
        missing = required - set(ranked_row.keys())
        if missing:
            raise ValueError(
                f"ranked_row is missing required fields: {missing}. "
                f"Make sure you pass a row from AdaptiveRanker.rank(), "
                f"not from rank_services() directly."
            )

        record = self.store.record(
            service=ranked_row["service"],
            outcome=outcome,
            risk_score=ranked_row["risk"],
            blast_radius=ranked_row["blast_radius"],
            fragility=ranked_row["fragility"],
            alpha=ranked_row["alpha_used"],
            beta=ranked_row["beta_used"],
            rank_at_time=ranked_row["rank"],
            graph_state_hash=graph_state_hash,
            notes=notes,
        )

        old_alpha = self.updater.alpha
        new_alpha, new_beta = self.updater.update(record)

        if abs(new_alpha - old_alpha) > 1e-6:
            logger.info(
                "Weights updated after %s on %s: alpha %.4f→%.4f beta %.4f→%.4f",
                outcome.value,
                ranked_row["service"],
                old_alpha,
                new_alpha,
                1.0 - old_alpha,
                new_beta,
            )

    # Inspection helpers

    @property
    def alpha(self) -> float:
        return self.updater.alpha

    @property
    def beta(self) -> float:
        return self.updater.beta

    def weight_history(self) -> list[dict]:
        """
        Return a timeline of weight states inferred from outcome history.
        Useful for visualizing how weights evolved over time.
        """
        outcomes = self.store.all()
        if not outcomes:
            return []

        history = []
        alpha = 0.6  # start from default
        from chaosrank_engine.adaptive.weight_updater import (
            ALPHA_MAX,
            ALPHA_MIN,
            MIN_OUTCOMES_BEFORE_UPDATE,
        )

        for i, rec in enumerate(outcomes):
            if (i + 1) >= MIN_OUTCOMES_BEFORE_UPDATE and rec.risk_score > 0:
                from chaosrank_engine.adaptive.outcome_store import OutcomeType as OT

                current_beta = 1.0 - alpha
                lr = self.updater.learning_rate

                if rec.outcome == OT.WEAKNESS_CONFIRMED:
                    fr_frac = rec.fr_contribution / rec.risk_score
                    delta = lr * fr_frac * (1.0 - current_beta)
                    new_beta = current_beta + delta
                    alpha = max(ALPHA_MIN, min(ALPHA_MAX, 1.0 - new_beta))
                elif rec.outcome == OT.WEAKNESS_NOT_FOUND:
                    br_frac = rec.br_contribution / rec.risk_score
                    delta = lr * br_frac * alpha
                    alpha = max(ALPHA_MIN, min(ALPHA_MAX, alpha - delta))

            history.append(
                {
                    "outcome_index": i + 1,
                    "timestamp": rec.timestamp,
                    "service": rec.service,
                    "outcome": rec.outcome.value,
                    "alpha": round(alpha, 4),
                    "beta": round(1.0 - alpha, 4),
                }
            )

        return history

    def summary(self) -> dict:
        """Human-readable summary of adaptive state."""
        state = self.updater.state()
        stats = self.store.stats()
        return {
            "current_alpha": state.alpha,
            "current_beta": state.beta,
            "update_count": state.update_count,
            "last_updated": state.last_updated,
            "total_outcomes": stats["total"],
            "confirmed": stats["confirmed"],
            "not_found": stats["not_found"],
            "inconclusive": stats["inconclusive"],
            "confirmation_rate": stats["confirmation_rate"],
            "confidence_level": self.confidence_level,
        }
