
import pytest
from fastapi.testclient import TestClient
from chaosrank_engine.api.main import app
from chaosrank_engine.api.webhook_manager import webhook_state

client = TestClient(app)

# Use the public access key for all webhook tests
AUTH = {}

@pytest.fixture(autouse=True)
def reset_webhook_state():
    """Ensure a clean state before each test."""
    webhook_state.clear()
    yield

def test_pagerduty_webhook():
    payload = {
        "event": {
            "data": {
                "title": "High CPU on api-gateway",
                "service": {"summary": "api-gateway"},
                "urgency": "high",
                "created_at": "2023-10-01T12:00:00Z"
            }
        }
    }
    response = client.post("/v1/webhooks/pagerduty", json=payload, headers=AUTH)
    assert response.status_code == 202
    
    state = webhook_state.get_state()
    assert len(state["incidents"]) == 1
    assert state["incidents"][0]["service"] == "api-gateway"
    assert state["incidents"][0]["severity"] == "high"

def test_datadog_webhook():
    payload = [
        {"source": "frontend", "target": "api-gateway", "edge_type": "sync"},
        {"source": "api-gateway", "target": "order-service", "edge_type": "sync"}
    ]
    response = client.post("/v1/webhooks/datadog", json=payload, headers=AUTH)
    assert response.status_code == 202
    
    state = webhook_state.get_state()
    assert len(state["graph"]["edges"]) == 2
    assert state["total_traces_processed"] == 2
    
    # Send again to verify weight increments
    client.post("/v1/webhooks/datadog", json=[{"source": "frontend", "target": "api-gateway", "edge_type": "sync"}], headers=AUTH)
    state = webhook_state.get_state()
    assert state["total_traces_processed"] == 3
    # Weight of frontend -> api-gateway should be 2.0
    for edge in state["graph"]["edges"]:
        if edge["source"] == "frontend" and edge["target"] == "api-gateway":
            assert edge["weight"] == 2.0

def test_jaeger_webhook():
    payload = {
        "data": [
            {
                "processes": {
                    "p1": {"serviceName": "frontend"},
                    "p2": {"serviceName": "backend"}
                },
                "spans": [
                    {"spanID": "s1", "processID": "p1", "references": []},
                    {"spanID": "s2", "processID": "p2", "references": [{"refType": "CHILD_OF", "spanID": "s1"}]}
                ]
            }
        ]
    }
    response = client.post("/v1/webhooks/jaeger", json=payload, headers=AUTH)
    assert response.status_code == 202
    
    state = webhook_state.get_state()
    assert len(state["graph"]["edges"]) == 1
    assert state["graph"]["edges"][0]["source"] == "frontend"
    assert state["graph"]["edges"][0]["target"] == "backend"

def test_clear_webhook_state():
    webhook_state.add_edge("a", "b")
    assert webhook_state.get_state()["total_traces_processed"] == 1
    response = client.post("/v1/webhooks/clear", headers=AUTH)
    assert response.status_code == 200
    assert webhook_state.get_state()["total_traces_processed"] == 0

