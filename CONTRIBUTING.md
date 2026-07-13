# Contributing to ChaosRank Engine

Thanks for your interest in the core ChaosRank Engine!

---

## Setup

**Requirements:** Python 3.11+

### Local Development

1. Clone the repository and navigate to the directory:
```bash
git clone https://github.com/Medinz01/ChaosRank-Engine
cd ChaosRank-Engine
```

2. Install dependencies:
```bash
pip install -e ".[dev]"
```

---

## Running Tests

```bash
# Full suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=chaosrank_engine --cov-report=term-missing
```

All tests must pass before submitting a PR.

---

## Linting

```bash
ruff check chaosrank_engine/ tests/
```

ChaosRank Engine uses `ruff` for linting. Configuration is in `pyproject.toml`.
CI runs ruff on every push — fix warnings before submitting.

---

## Project Structure

```
ChaosRank-Engine/
├── chaosrank_engine/
│   ├── api/                  # FastAPI service layer, routes, models
│   ├── scorer/               # Core algorithms (Blast Radius, Fragility)
│   ├── federation/           # Multi-domain graph merging
│   └── adaptive/             # Bayesian weight adjustments
├── tests/                    # Test suite
└── pyproject.toml
```

---

## Submitting a PR

1. Fork the repo and create a branch: `git checkout -b your-feature`
2. Make your changes
3. Run `pytest tests/ -v` — all tests must pass
4. Run `ruff check chaosrank_engine/ tests/` — no warnings
5. Update `CHANGELOG.md` under `[Unreleased]`
6. Open a PR with a clear description of what changed and why
