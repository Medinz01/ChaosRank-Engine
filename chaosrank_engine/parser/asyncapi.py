
import yaml
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def parse_asyncapi_specs(paths: list[Path]) -> list[dict[str, Any]]:
    """Parse multiple AsyncAPI yaml files and cross-reference pub/sub to build edges.
    
    Returns a list of dicts that can be mapped to EdgePayloads:
    [
        {
            "source": str,
            "target": str,
            "weight": float,
            "edge_type": "async",
            "channel": "asyncapi",
            "topic": str
        }
    ]
    """
    edges = []
    
    publishers = {} # channel -> list of publishers
    subscribers = {} # channel -> list of subscribers
    
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                spec = yaml.safe_load(f)
                
            service_name = spec.get("info", {}).get("title", path.stem).lower().replace(" ", "-")
            channels = spec.get("channels", {})
            
            for channel_name, channel_info in channels.items():
                if "publish" in channel_info:
                    subscribers.setdefault(channel_name, []).append(service_name)
                if "subscribe" in channel_info:
                    publishers.setdefault(channel_name, []).append(service_name)
                    
        except Exception as e:
            logger.error("Failed to parse AsyncAPI spec %s: %s", path, e)
            
    for channel_name in set(publishers.keys()).union(subscribers.keys()):
        channel_pubs = publishers.get(channel_name, [])
        channel_subs = subscribers.get(channel_name, [])
        
        for pub in channel_pubs:
            for sub in channel_subs:
                edges.append({
                    "source": sub,
                    "target": pub,
                    "weight": 1.0,
                    "edge_type": "async",
                    "channel": "asyncapi",
                    "topic": channel_name
                })
                
    logger.info("Parsed %d async edges from AsyncAPI specs", len(edges))
    return edges
