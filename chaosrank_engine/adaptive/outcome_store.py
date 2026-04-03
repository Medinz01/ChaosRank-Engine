"""Persistent storage for chaos experiment outcomes.
Records results alongside score decompositions to drive adaptive weight updates.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

import os

if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    DEFAULT_STORE_PATH = Path("/tmp/.chaosrank/outcomes.json")
else:
    DEFAULT_STORE_PATH = Path(".chaosrank/outcomes.json")


class OutcomeType(str, Enum):
    WEAKNESS_CONFIRMED = "WEAKNESS_CONFIRMED"
    WEAKNESS_NOT_FOUND = "WEAKNESS_NOT_FOUND"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class InterventionRecord:
    """
    A single recorded experiment outcome with full score decomposition.

    score_decomposition captures exactly how much each signal contributed
    to ranking this service — this is the input to the weight update rule.
    """

    service: str
    outcome: OutcomeType
    timestamp: str  # ISO 8601

    # Score decomposition at time of intervention
    risk_score: float
    blast_radius: float
    fragility: float
    alpha: float  # weights active at time of intervention
    beta: float
    br_contribution: float  # alpha * blast_radius
    fr_contribution: float  # beta * fragility

    # Context
    rank_at_time: int  # rank position when selected
    graph_state_hash: str | None  # hash of graph topology, for drift detection
    notes: str | None  # optional free-text


class OutcomeStore:
    """Thread-safe, append-only store for experiment outcomes."""

    def __init__(self, path: Path = DEFAULT_STORE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[InterventionRecord] = []
        self._load()

    # Data access API

    def record(
        self,
        service: str,
        outcome: OutcomeType,
        risk_score: float,
        blast_radius: float,
        fragility: float,
        alpha: float,
        beta: float,
        rank_at_time: int,
        graph_state_hash: str | None = None,
        notes: str | None = None,
    ) -> InterventionRecord:
        """Record an experiment outcome and persist to disk."""
        rec = InterventionRecord(
            service=service,
            outcome=outcome,
            timestamp=datetime.utcnow().isoformat(),
            risk_score=round(risk_score, 6),
            blast_radius=round(blast_radius, 6),
            fragility=round(fragility, 6),
            alpha=round(alpha, 6),
            beta=round(beta, 6),
            br_contribution=round(alpha * blast_radius, 6),
            fr_contribution=round(beta * fragility, 6),
            rank_at_time=rank_at_time,
            graph_state_hash=graph_state_hash,
            notes=notes,
        )
        self._records.append(rec)
        self._save()
        logger.info(
            "Recorded outcome %s for service %s (rank %d, risk=%.4f)",
            outcome.value,
            service,
            rank_at_time,
            risk_score,
        )
        return rec

    def all(self) -> list[InterventionRecord]:
        """Return all recorded outcomes, oldest first."""
        return list(self._records)

    def for_service(self, service: str) -> list[InterventionRecord]:
        """Return all outcomes for a specific service."""
        return [r for r in self._records if r.service == service]

    def recent(self, n: int) -> list[InterventionRecord]:
        """Return the n most recent outcomes."""
        return list(self._records[-n:])

    def confirmed(self) -> list[InterventionRecord]:
        return [r for r in self._records if r.outcome == OutcomeType.WEAKNESS_CONFIRMED]

    def not_found(self) -> list[InterventionRecord]:
        return [r for r in self._records if r.outcome == OutcomeType.WEAKNESS_NOT_FOUND]

    def stats(self) -> dict:
        """Summary statistics over all recorded outcomes."""
        total = len(self._records)
        if total == 0:
            return {
                "total": 0,
                "confirmed": 0,
                "not_found": 0,
                "inconclusive": 0,
                "confirmation_rate": None,
            }
        confirmed = len(self.confirmed())
        not_found = len(self.not_found())
        inconclusive = total - confirmed - not_found
        return {
            "total": total,
            "confirmed": confirmed,
            "not_found": not_found,
            "inconclusive": inconclusive,
            "confirmation_rate": round(confirmed / total, 4),
        }

    # Serialization

    def _save(self) -> None:
        data = [asdict(r) for r in self._records]
        # Convert OutcomeType enum values to strings for JSON serialization
        for d in data:
            if isinstance(d.get("outcome"), OutcomeType):
                d["outcome"] = d["outcome"].value
        try:
            self.path.write_text(json.dumps(data, indent=2))
        except OSError as e:
            logger.error("Failed to persist outcomes to %s: %s", self.path, e)

    def _load(self) -> None:
        if not self.path.exists():
            self._records = []
            return
        try:
            data = json.loads(self.path.read_text())
            self._records = [
                InterventionRecord(**{**d, "outcome": OutcomeType(d["outcome"])}) for d in data
            ]
            logger.debug("Loaded %d outcome records from %s", len(self._records), self.path)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Could not load outcomes from %s: %s — starting fresh", self.path, e)
            self._records = []
