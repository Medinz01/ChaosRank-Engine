"""Reinforcement Learning-based weight updater for ChaosRank.
Dynamically adjusts alpha/beta weights based on experiment outcomes to optimize
ranking accuracy for specific environments.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from chaosrank_engine.adaptive.outcome_store import InterventionRecord, OutcomeStore, OutcomeType

logger = logging.getLogger(__name__)

# Weight bounds — prevent degenerate single-signal scoring
ALPHA_MIN = 0.3
ALPHA_MAX = 0.9

# Default learning rate — conservative to prevent single-outcome overfitting
DEFAULT_LEARNING_RATE = 0.05

# Minimum outcomes before updates are applied (set to 1 to avoid cold-start stagnation)
MIN_OUTCOMES_BEFORE_UPDATE = 1


@dataclass
class WeightState:
    """Current alpha/beta weights with update history summary."""

    alpha: float
    beta: float
    update_count: int
    last_updated: str | None  # ISO 8601 timestamp of last update
    confirmation_rate: float | None


class WeightUpdater:
    """Maintains and updates alpha/beta weights based on confirmation outcomes."""

    def __init__(
        self,
        store: OutcomeStore,
        initial_alpha: float = 0.6,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        outcome_window: int = 20,
    ) -> None:
        if abs(initial_alpha + (1.0 - initial_alpha) - 1.0) > 1e-6:
            raise ValueError("initial_alpha must be in (0, 1)")
        if not ALPHA_MIN <= initial_alpha <= ALPHA_MAX:
            raise ValueError(
                f"initial_alpha {initial_alpha} outside bounds [{ALPHA_MIN}, {ALPHA_MAX}]"
            )

        self.store = store
        self.learning_rate = learning_rate
        self.outcome_window = outcome_window
        self._alpha = initial_alpha
        self._update_count = 0
        self._last_updated: str | None = None

        # Rehydrate from existing outcomes if store has data
        self._rehydrate()

    # State access

    @property
    def alpha(self) -> float:
        return round(self._alpha, 6)

    @property
    def beta(self) -> float:
        return round(1.0 - self._alpha, 6)

    def state(self) -> WeightState:
        stats = self.store.stats()
        return WeightState(
            alpha=self.alpha,
            beta=self.beta,
            update_count=self._update_count,
            last_updated=self._last_updated,
            confirmation_rate=stats.get("confirmation_rate"),
        )

    def update(self, record: InterventionRecord) -> tuple[float, float]:
        """
        Apply weight update from a single intervention record.

        Returns (new_alpha, new_beta).
        Does nothing and returns current weights if outcome is INCONCLUSIVE
        or if minimum outcome threshold has not been reached.
        """
        total_outcomes = len(self.store.all())
        if total_outcomes < MIN_OUTCOMES_BEFORE_UPDATE:
            logger.debug(
                "Weight update deferred — only %d outcomes recorded (minimum %d required)",
                total_outcomes,
                MIN_OUTCOMES_BEFORE_UPDATE,
            )
            return self.alpha, self.beta

        if record.outcome == OutcomeType.INCONCLUSIVE:
            logger.debug("INCONCLUSIVE outcome — no weight update")
            return self.alpha, self.beta

        if record.risk_score <= 0:
            logger.warning(
                "Risk score is zero for %s — cannot compute decomposition, skipping update",
                record.service,
            )
            return self.alpha, self.beta

        new_alpha = self._compute_new_alpha(record)
        new_alpha = max(ALPHA_MIN, min(ALPHA_MAX, new_alpha))

        old_alpha = self._alpha
        self._alpha = new_alpha
        self._update_count += 1
        self._last_updated = record.timestamp

        logger.info(
            "Weight update [%s] service=%s  alpha: %.4f → %.4f  beta: %.4f → %.4f",
            record.outcome.value,
            record.service,
            old_alpha,
            new_alpha,
            1.0 - old_alpha,
            1.0 - new_alpha,
        )

        return self.alpha, self.beta

    def update_from_recent(self) -> tuple[float, float]:
        """
        Recompute weights from the outcome_window most recent outcomes.
        Use this to rehydrate weights from a loaded store.
        """
        recent = self.store.recent(self.outcome_window)
        actionable = [r for r in recent if r.outcome != OutcomeType.INCONCLUSIVE]

        if len(actionable) < MIN_OUTCOMES_BEFORE_UPDATE:
            logger.debug(
                "Insufficient actionable outcomes (%d) for weight update",
                len(actionable),
            )
            return self.alpha, self.beta

        # Apply each update sequentially — order matters
        for record in actionable:
            if record.risk_score > 0:
                self._alpha = max(ALPHA_MIN, min(ALPHA_MAX, self._compute_new_alpha(record)))
                self._update_count += 1

        self._last_updated = actionable[-1].timestamp if actionable else None
        logger.info(
            "Weights rehydrated from %d outcomes — alpha=%.4f beta=%.4f",
            len(actionable),
            self.alpha,
            self.beta,
        )
        return self.alpha, self.beta

    # Internal update logic

    def _compute_new_alpha(self, record: InterventionRecord) -> float:
        """
        Core update rule from Claim 3 of invention_disclosure_v2.md.

        CONFIRMED: fragility was right — increase beta (decrease alpha)
          delta = lr × (fr_contribution / risk_score) × (1 - beta)
          beta_new = beta + delta

        NOT_FOUND: blast_radius was wrong — decrease alpha
          delta = lr × (br_contribution / risk_score) × alpha
          alpha_new = alpha - delta
        """
        lr = self.learning_rate
        risk = record.risk_score
        br_contrib = record.br_contribution
        fr_contrib = record.fr_contribution
        current_beta = 1.0 - self._alpha

        if record.outcome == OutcomeType.WEAKNESS_CONFIRMED:
            # Fragility signal was predictive — reward beta
            fr_fraction = fr_contrib / risk
            delta = lr * fr_fraction * (1.0 - current_beta)
            new_beta = current_beta + delta
            return 1.0 - new_beta

        elif record.outcome == OutcomeType.WEAKNESS_NOT_FOUND:
            # Blast radius signal over-predicted — penalize alpha
            br_fraction = br_contrib / risk
            delta = lr * br_fraction * self._alpha
            return self._alpha - delta

        # Should not reach here — INCONCLUSIVE handled upstream
        return self._alpha

    # State rehydration

    def _rehydrate(self) -> None:
        """On init, recompute weights from existing outcome history."""
        all_outcomes = self.store.all()
        if not all_outcomes:
            return
        logger.debug("Rehydrating weights from %d existing outcomes", len(all_outcomes))
        self.update_from_recent()
