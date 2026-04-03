import requests
import json

url = "http://127.0.0.1:8081/v1/rank"
headers = {
    "X-ChaosRank-Key": "ysdk_9f3A7xL2Qp8vN6kT1mR4ZcH5uB0eWjD",
    "Content-Type": "application/json"
}

# Minimal payload with 2 edges as seen in the trace
payload = {
    "graph": {
        "edges": [
            {"source": "service-a", "target": "service-b", "weight": 10.0},
            {"source": "service-b", "target": "service-c", "weight": 5.0}
        ]
    },
    "incidents": {},
    "config": {
        "alpha": 0.6,
        "beta": 0.4,
        "decay_lambda": 0.1,
        "base_window": 5.0,
        "use_betweenness": False,
        "w_pr": 0.5,
        "w_od": 0.5,
        "async_weight_factor": 0.5,
        "async_deps_provided": False,
        "top_n": 0
    }
}

try:
    response = requests.post(url, headers=headers, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response Body:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
