"""Graph construction logic for the ChaosRank Engine.
Converts trace data into a NetworkX dependency graph for risk analysis.
"""
import logging
from pathlib import Path
import networkx as nx
from chaosrank_engine.parser.jaeger import parse_traces

logger = logging.getLogger(__name__)


def build_graph(
    traces_path: Path,
    min_call_frequency: int = 10,
    trace_format: str = "jaeger",
    otlp_format: str = "json",
) -> nx.DiGraph:
    """Construct a dependency graph from trace export files."""
    """Build a NetworkX DiGraph from a trace export file for engine benchmarks.

    Args:
        traces_path:        Path to trace export file.
        min_call_frequency: Drop edges with fewer calls. Default 10.
        trace_format:       "jaeger" | "otlp". Default "jaeger".
        otlp_format:        "json" | "protobuf". Only used when trace_format="otlp".
                            Default "json" (existing behaviour, no change).
    """
    if trace_format == "jaeger":
        edges = parse_traces(traces_path, min_call_frequency=min_call_frequency)
    else:
        # Reduced OTLP support for engine-only benchmarks to keep dependencies low
        raise ValueError(
            f"Unknown or unsupported trace format in engine: {trace_format!r}. "
            "Use full ChaosRank SDK for OTLP support."
        )

    G = nx.DiGraph()
    for (caller, callee), weight in edges.items():
        G.add_edge(caller, callee, weight=weight)

    logger.info("Built graph: %d services, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def reverse_graph(G: nx.DiGraph) -> nx.DiGraph:
    """Return a copy of the graph with all edges reversed (downstream -> upstream)."""
    return G.reverse(copy=True)
