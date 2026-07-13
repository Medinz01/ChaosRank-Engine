
import threading
import networkx as nx
from typing import Dict, Any, List

class WebhookState:
    """Thread-safe singleton to hold streaming graph edges and incidents."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.graph = nx.DiGraph()
        self.incidents: List[Dict[str, Any]] = []
        self.total_traces_processed = 0

    def add_edge(self, source: str, target: str, edge_type: str = "sync"):
        """Increment the weight of an edge. If it doesn't exist, create it."""
        with self.lock:
            self.total_traces_processed += 1
            if self.graph.has_edge(source, target):
                self.graph[source][target]["weight"] += 1.0
            else:
                self.graph.add_edge(source, target, weight=1.0, edge_type=edge_type)

    def add_incident(self, service: str, severity: str, timestamp: str):
        """Append a new incident payload."""
        with self.lock:
            self.incidents.append({
                "service": service,
                "severity": severity,
                "timestamp": timestamp
            })

    def get_state(self):
        """Retrieve the current aggregated state for the dashboard or ranking engine."""
        with self.lock:
            edges = []
            for u, v, data in self.graph.edges(data=True):
                edges.append({
                    "source": u,
                    "target": v,
                    "weight": data.get("weight", 1.0),
                    "edge_type": data.get("edge_type", "sync")
                })
            
            return {
                "graph": {
                    "edges": edges
                },
                "incidents": list(self.incidents),
                "total_traces_processed": self.total_traces_processed,
                "nodes": list(self.graph.nodes)
            }

    def clear(self):
        """Reset the streaming state."""
        with self.lock:
            self.graph.clear()
            self.incidents.clear()
            self.total_traces_processed = 0

# Global singleton instance
webhook_state = WebhookState()
