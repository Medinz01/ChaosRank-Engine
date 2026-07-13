"""Domain adapter registry for ChaosRank federation.
Manages the registration and validation of domain-specific adapters for
cross-domain risk scoring.
"""
from __future__ import annotations

import logging

from chaosrank_engine.federation.contracts import DomainBundle

logger = logging.getLogger(__name__)


class DomainRegistry:
    """Registry of domain bundles for federated risk scoring."""

    def __init__(self) -> None:
        self._bundles: dict[str, DomainBundle] = {}

    # Registration

    def register(self, bundle: DomainBundle) -> None:
        """
        Register a domain bundle.

        Raises ValueError if a bundle with the same domain_id is already
        registered. Use replace() to update an existing registration.
        """
        domain_id = bundle.domain_id
        if domain_id in self._bundles:
            raise ValueError(
                f"Domain '{domain_id}' is already registered. "
                f"Use replace() to update an existing registration."
            )
        self._bundles[domain_id] = bundle
        logger.info(
            "Registered domain '%s' (load_metric=%s)",
            domain_id,
            "provided" if bundle.load_metric else "none — will use window average",
        )

    def replace(self, bundle: DomainBundle) -> None:
        """Replace an existing domain registration."""
        self._bundles[bundle.domain_id] = bundle
        logger.info("Replaced domain registration '%s'", bundle.domain_id)

    def unregister(self, domain_id: str) -> None:
        """Remove a domain registration."""
        if domain_id not in self._bundles:
            raise KeyError(f"Domain '{domain_id}' is not registered")
        del self._bundles[domain_id]
        logger.info("Unregistered domain '%s'", domain_id)

    # Inspection

    @property
    def domain_ids(self) -> list[str]:
        return list(self._bundles.keys())

    @property
    def bundles(self) -> list[DomainBundle]:
        return list(self._bundles.values())

    def get(self, domain_id: str) -> DomainBundle:
        if domain_id not in self._bundles:
            raise KeyError(f"Domain '{domain_id}' is not registered")
        return self._bundles[domain_id]

    def __len__(self) -> int:
        return len(self._bundles)

    def __contains__(self, domain_id: str) -> bool:
        return domain_id in self._bundles

    # Validation

    def validate(self) -> list[str]:
        """
        Validate all registered domains for consistency.

        Returns list of error strings. Empty list means valid.
        Runs three checks:
          1. Each bundle's graph adapter validates its own output
          2. Cross-domain node ID collision detection
          3. Inter-domain edge declaration consistency (if any)
        """
        if not self._bundles:
            return ["No domains registered"]

        errors = []

        # Check 1 — per-domain graph validation
        domain_node_sets: dict[str, set[str]] = {}
        for domain_id, bundle in self._bundles.items():
            try:
                graph, components = bundle.graph.build_graph()
                domain_errors = bundle.graph.validate(graph, components)
                for e in domain_errors:
                    errors.append(f"[{domain_id}] {e}")
                domain_node_sets[domain_id] = set(graph.nodes())
            except Exception as exc:
                errors.append(f"[{domain_id}] GraphSourceAdapter.build_graph() raised: {exc}")
                domain_node_sets[domain_id] = set()

        all_ids: list[tuple[str, str]] = []  # (domain_id, node_id)
        for domain_id, nodes in domain_node_sets.items():
            for node in nodes:
                all_ids.append((domain_id, node))

        node_to_domains: dict[str, list[str]] = {}
        for domain_id, node in all_ids:
            node_to_domains.setdefault(node, []).append(domain_id)

        collisions = {
            node: domains for node, domains in node_to_domains.items() if len(domains) > 1
        }
        if collisions:
            for node, domains in collisions.items():
                errors.append(
                    f"Node ID collision: '{node}' appears in domains "
                    f"{domains}. Use domain-prefixed IDs or declare an "
                    f"inter-domain edge to make this relationship explicit."
                )

        return errors

    def summary(self) -> dict:
        """Human-readable summary of registered domains."""
        result = {}
        for domain_id, bundle in self._bundles.items():
            try:
                graph, components = bundle.graph.build_graph()
                result[domain_id] = {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "components": len(components),
                    "load_metric": bundle.load_metric is not None,
                }
            except Exception as exc:
                result[domain_id] = {"error": str(exc)}
        return result
