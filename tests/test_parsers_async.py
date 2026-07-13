
import json
import tempfile
import yaml
from pathlib import Path
from chaosrank_engine.parser.kafka import parse_kafka_topics
from chaosrank_engine.parser.asyncapi import parse_asyncapi_specs

def test_kafka_parser():
    kafka_json = {
        "topics": [
            {
                "name": "order-events",
                "producer": "frontend",
                "consumers": ["order-service"]
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(kafka_json, f)
        f_path = Path(f.name)
        
    try:
        edges = parse_kafka_topics(f_path)
        assert len(edges) == 1
        edge = edges[0]
        assert edge["source"] == "order-service"
        assert edge["target"] == "frontend"
        assert edge["weight"] == 1.0
        assert edge["topic"] == "order-events"
        assert edge["edge_type"] == "async"
    finally:
        f_path.unlink()

def test_asyncapi_parser():
    spec1 = {
        "info": {"title": "Service A"},
        "channels": {
            "user/signup": {
                "publish": {"message": {"payload": {}}}
            }
        }
    }
    
    spec2 = {
        "info": {"title": "Service B"},
        "channels": {
            "user/signup": {
                "subscribe": {"message": {"payload": {}}}
            }
        }
    }
    
    with tempfile.TemporaryDirectory() as td:
        dir_path = Path(td)
        with open(dir_path / "spec1.yaml", "w") as f:
            yaml.dump(spec1, f)
        with open(dir_path / "spec2.yaml", "w") as f:
            yaml.dump(spec2, f)
            
        edges = parse_asyncapi_specs([dir_path / "spec1.yaml", dir_path / "spec2.yaml"])
        assert len(edges) == 1
        edge = edges[0]
        
        assert edge["source"] == "service-a"
        assert edge["target"] == "service-b"
        assert edge["topic"] == "user/signup"
        assert edge["edge_type"] == "async"
