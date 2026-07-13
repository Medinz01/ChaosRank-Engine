
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def parse_kafka_topics(path: Path) -> list[dict[str, Any]]:
    """Parse a kafka-topics.json file and return async edge representations.
    
    Returns a list of dicts that can be mapped to EdgePayloads:
    [
        {
            "source": str,
            "target": str,
            "weight": float,
            "edge_type": "async",
            "channel": "kafka",
            "topic": str
        }
    ]
    """
    edges = []
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        topics = data.get("topics", [])
        for topic in topics:
            topic_name = topic.get("name")
            producer = topic.get("producer")
            consumers = topic.get("consumers", [])
            
            if not producer or not topic_name:
                continue
                
            for consumer in consumers:
                
                edges.append({
                    "source": consumer,
                    "target": producer,
                    "weight": 1.0,
                    "edge_type": "async",
                    "channel": "kafka",
                    "topic": topic_name
                })
                
        logger.info("Parsed %d async edges from Kafka topics", len(edges))
    except Exception as e:
        logger.error("Failed to parse Kafka topics: %s", e)
        
    return edges
