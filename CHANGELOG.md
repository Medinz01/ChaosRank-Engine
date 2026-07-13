# Changelog

All notable changes to ChaosRank Engine will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
ChaosRank Engine follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.2] — 2026-07-13

### Changed
- **Open BSL Architecture Transition**: Completely removed API keys, rate limiters, and proprietary SaaS restrictions to prepare the engine for open self-hosted distribution on PyPI.
- **Code Quality**: Removed unused `fastapi.Depends` dependency injections to resolve `ruff` lint errors following the authorization removal.

---

## [0.2.1] — 2026-06-25

### Added
- **AWS Lambda Support**: Implemented writable `/tmp` storage support for serverless execution of the engine.
- **API Tiers (Deprecated)**: Implemented Public and Pro API tier logic, including `slowapi` rate limiting.

### Fixed
- Resolved `ruff` lint errors in `outcome_store` and adaptive routing modules.

---

## [0.2.0] — 2026-06-21

### Added
- **Initial Core Engine Release**: Extracted the proprietary algorithms (Blast Radius, Fragility, Adaptive Scoring, Federation) from the CLI into this dedicated backend REST API.
