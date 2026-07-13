
import logging

import networkx as nx

logger = logging.getLogger(__name__)

DEFAULT_W_PR = 0.5
DEFAULT_W_OD = 0.5
DEFAULT_ASYNC_WEIGHT_FACTOR = 0.5

DEFAULT_W_BC = 0.2

ASYNC_SERVICE_PATTERNS = ("kafka", "sqs", "rabbitmq", "pubsub", "nats", "kinesis")


def compute_blast_radius(
    G: nx.DiGraph,
    w_pr: float = DEFAULT_W_PR,
    w_od: float = DEFAULT_W_OD,
    async_deps_provided: bool = False,
    async_weight_factor: float = DEFAULT_ASYNC_WEIGHT_FACTOR,
    use_betweenness: bool = False,
    w_bc: float | None = None,
) -> dict[str, float]:
    """Compute a blended blast radius score per service.

    Combines PageRank, In-Degree Centrality, and optional Betweenness Centrality
    into a weighted risk signal. All components are normalized to [0, 1].
    """
    if not 0.0 < async_weight_factor <= 1.0:
        raise ValueError(f"async_weight_factor must be in (0.0, 1.0], got {async_weight_factor}")

    if G.number_of_nodes() == 0:
        logger.warning("Empty graph — no blast radius scores to compute")
        return {}

    # Weight validation
    if use_betweenness:
        w_pr, w_od, w_bc = _resolve_betweenness_weights(w_pr, w_od, w_bc)
    else:
        if abs(w_pr + w_od - 1.0) > 1e-6:
            raise ValueError(f"w_pr + w_od must equal 1.0, got {w_pr + w_od:.6f}")
        w_bc = 0.0  # unused, kept for uniform blend call below

    _warn_async_blindspot(G, async_deps_provided)

    G_scored = _apply_async_weight(G, async_weight_factor)

    # PageRank
    try:
        pr = nx.pagerank(G_scored, weight="weight")
    except nx.PowerIterationFailedConvergence:
        logger.warning("PageRank failed to converge — falling back to uniform scores")
        pr = {n: 1.0 / G_scored.number_of_nodes() for n in G_scored.nodes()}

    # In-degree centrality
    if G_scored.number_of_nodes() > 1:
        od = nx.in_degree_centrality(G_scored)
    else:
        od = {n: 0.0 for n in G_scored.nodes()}

    # Betweenness centrality
    if use_betweenness:
        bc = _compute_betweenness(G_scored)
    else:
        bc = {}

    # Normalization
    pr_norm = _normalize(pr)
    od_norm = _normalize(od)
    bc_norm = _normalize(bc) if bc else {}

    # Blend
    scores: dict[str, float] = {}
    for node in G_scored.nodes():
        score = w_pr * pr_norm.get(node, 0.0) + w_od * od_norm.get(node, 0.0)
        if use_betweenness:
            score += w_bc * bc_norm.get(node, 0.0)
        scores[node] = score

    # Logging
    async_edge_count = sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "async")
    if async_edge_count:
        logger.info(
            "Blast radius computed: %d async edges scaled by async_weight_factor=%.2f",
            async_edge_count,
            async_weight_factor,
        )

    if use_betweenness:
        logger.info(
            "Blast radius computed for %d services "
            "(w_pr=%.2f, w_od=%.2f, w_bc=%.2f, betweenness=on)",
            len(scores),
            w_pr,
            w_od,
            w_bc,
        )
    else:
        logger.info(
            "Blast radius computed for %d services (w_pr=%.2f, w_od=%.2f)",
            len(scores),
            w_pr,
            w_od,
        )

    return scores


# Internal helpers


def _resolve_betweenness_weights(
    w_pr: float,
    w_od: float,
    w_bc: float | None,
) -> tuple[float, float, float]:
    """Validate or auto-adjust weights when betweenness is enabled.

    If w_bc is None:
        - Default w_bc to DEFAULT_W_BC (0.20).
        - Scale w_pr and w_od proportionally so the three weights sum to 1.0.
        - Emit a warning showing the adjusted values.

    If w_bc is provided:
        - Validate w_pr + w_od + w_bc == 1.0.
        - Raise ValueError if not.

    Returns (w_pr, w_od, w_bc) guaranteed to sum to 1.0.
    """
    if w_bc is None:
        w_bc = DEFAULT_W_BC
        total_pr_od = w_pr + w_od
        if total_pr_od < 1e-9:
            raise ValueError("w_pr + w_od must be positive")
        scale = 1.0 - w_bc
        w_pr_new = round(w_pr / total_pr_od * scale, 6)
        w_od_new = round(w_od / total_pr_od * scale, 6)
        # Absorb rounding residual into w_pr
        residual = round(1.0 - w_pr_new - w_od_new - w_bc, 6)
        w_pr_new = round(w_pr_new + residual, 6)
        logger.warning(
            "use_betweenness=True but w_bc not provided. "
            "Auto-adjusted weights: w_pr=%.4f, w_od=%.4f, w_bc=%.4f. "
            "Pass w_bc explicitly to silence this warning.",
            w_pr_new,
            w_od_new,
            w_bc,
        )
        return w_pr_new, w_od_new, w_bc

    total = w_pr + w_od + w_bc
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"w_pr + w_od + w_bc must equal 1.0 when use_betweenness=True, "
            f"got {total:.6f} (w_pr={w_pr}, w_od={w_od}, w_bc={w_bc})"
        )
    return w_pr, w_od, w_bc


def _compute_betweenness(G: nx.DiGraph) -> dict[str, float]:
    """Compute betweenness centrality with inverted edge weights.

    NetworkX betweenness_centrality treats weight as *distance* — higher
    weight means the edge is less likely to appear on shortest paths.
    Our edge weights are call frequencies (higher = stronger dependency),
    so we invert them (1 / weight) to make high-frequency edges shorter
    and therefore more likely to appear on critical paths.

    A scratch copy of G is used; the caller's graph is never mutated.
    Zero-weight edges are assigned a very large distance (1e9) so they are
    effectively excluded from path calculations.
    """
    G_bc = G.copy()
    for u, v, data in G_bc.edges(data=True):
        w = data.get("weight", 1)
        G_bc[u][v]["weight"] = 1.0 / w if w > 0 else 1e9

    return nx.betweenness_centrality(G_bc, weight="weight", normalized=True)


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalize a score dict to [0, 1] via min-max.

    If all values are identical (range == 0), returns 0.5 for all nodes.
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    span = hi - lo
    if span > 0:
        return {n: (v - lo) / span for n, v in scores.items()}
    return {n: 0.5 for n in scores}


def _apply_async_weight(G: nx.DiGraph, factor: float) -> nx.DiGraph:
    """Return a copy of G with async edge weights scaled by factor.

    If factor == 1.0 or no async edges exist, returns G unchanged (no copy needed).
    """
    has_async = any(d.get("edge_type") == "async" for _, _, d in G.edges(data=True))
    if not has_async or abs(factor - 1.0) < 1e-9:
        return G

    G_copy = G.copy()
    for u, v, data in G_copy.edges(data=True):
        if data.get("edge_type") == "async":
            G_copy[u][v]["weight"] = data.get("weight", 1) * factor
    return G_copy


def _warn_async_blindspot(G: nx.DiGraph, async_deps_provided: bool) -> None:
    async_nodes = [n for n in G.nodes() if any(p in n for p in ASYNC_SERVICE_PATTERNS)]

    if async_deps_provided:
        async_edge_count = sum(
            1 for _, _, data in G.edges(data=True) if data.get("edge_type") == "async"
        )
        logger.info(
            "Async deps provided — %d async edges merged into graph. "
            "Blast radius scores include async dependencies.",
            async_edge_count,
        )
    elif async_nodes:
        logger.warning(
            "Async messaging services detected in trace data. "
            "Blast radius scores may be incomplete for event-driven dependencies. "
            "Manually verify top-ranked services against known async dependency maps. "
            "Use --async-deps to provide a manifest. Detected: %s",
            ", ".join(sorted(async_nodes)),
        )
