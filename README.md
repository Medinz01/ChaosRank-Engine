# ChaosRank Engine (Private) 🧠

This is the proprietary core of the ChaosRank project. It contains the mathematical models, adaptive scoring algorithms, and the FastAPI service layer.

## License
**Business Source License 1.1 (BSL 1.1)**

## Components
- **Scorer**: PageRank + In-degree blended centrality for Blast Radius.
- **Adaptive**: Bayesian weight adjustment based on experiment outcomes.
- **Orchestration**: Graph merging and federation.

## Development
1. Install dependencies: `pip install -e .`
2. Run tests: `pytest tests/ -v`
3. Start the API: `uvicorn chaosrank_engine.api.main:app --reload`
