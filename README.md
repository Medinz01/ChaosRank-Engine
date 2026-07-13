# ChaosRank Engine

This is the core of the ChaosRank project. It contains the mathematical models, adaptive scoring algorithms, and the FastAPI service layer that power the ChaosRank CLI.

## Overview

ChaosRank is a risk-driven chaos experiment scheduler. While the [ChaosRank CLI](https://github.com/Medinz01/chaosrank) handles trace parsing, graph building, and incident collection, the **ChaosRank Engine** acts as the central brain. It receives summarized graphs and incident histories to compute deterministic risk rankings.

## Features

- **Blast Radius Calculation**: Uses PageRank and In-degree blended centrality to determine the transitive impact of a service failure.
- **Fragility Scoring**: Normalizes historical incident frequency against load and topological importance.
- **Adaptive Weights**: Self-correcting Bayesian models that adjust risk factors based on the outcomes of previous chaos experiments.
- **Orchestration**: Graph merging and multi-domain federation capabilities.

## Installation and Usage

The easiest way to run the engine is via pip:

```bash
pip install chaosrank-engine

# Start the engine locally
uvicorn chaosrank_engine.api.main:app --host 0.0.0.0 --port 8080
```

By default, the engine exposes a REST API at port 8080. You can then point your ChaosRank CLI to this engine.

## Development

To set up a local development environment:

1. Clone the repository and install dependencies:
   ```bash
   pip install -e .
   ```
2. Run the test suite:
   ```bash
   pytest tests/ -v
   ```
3. Start the API in reload mode:
   ```bash
   uvicorn chaosrank_engine.api.main:app --reload
   ```

## License

This software is provided under the **Business Source License 1.1 (BSL 1.1)**.
