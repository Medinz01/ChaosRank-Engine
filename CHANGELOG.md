# Changelog

All notable changes to ChaosRank Engine will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
ChaosRank Engine follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.3] — 2026-07-15

### Fixed
- **Small-N Normalization**: Fixed an issue in `fragility.py` where the Z-score normalization clamped the maximum fragility score below 1.0 for small microservice topologies (N < 10). The algorithm now dynamically scales the normalization bounds based on the sample's maximum absolute deviation, allowing the operational fragility signal to properly compete with the structural blast radius signal in small graphs.

---

## [0.2.2] — 2026-07-13

### Added
- **Webhooks & Live Streaming**: Introduced `WebhookManager` and the `/webhooks` API endpoints to support direct incident streaming from alerting systems (PagerDuty, Datadog, Opsgenie).
- **Async Topology Parsers**: Added native parsers for `asyncapi` and `kafka` topologies directly in the engine, allowing asynchronous edge processing without CLI translation.

### Changed
- **Federation & Adaptive Improvements**: Refined graph merging and correlation functions in `chaosrank_engine/federation/`, and optimized the Bayesian `adaptive` models by removing legacy outcome store dependencies.
- **Open BSL Architecture**: Completely removed API keys, `slowapi` rate limiters, and proprietary SaaS authentication. The engine is now completely open for self-hosted distribution.

---

## [0.2.1] — 2026-06-25

### Added
- **AWS Lambda Support**: Implemented writable `/tmp` storage support for serverless execution of the engine.
- **API Tiers**: Added Public and Pro API tier logic, including `slowapi` rate limiting. *(Note: This system has been deprecated and removed in 0.2.2)*.

---

## [0.2.0] — 2026-06-21

### Added
- **Initial Core Engine Release**: Extracted the proprietary algorithms (Blast Radius, Fragility, Adaptive Scoring, and Federation) from the CLI into this dedicated backend REST API.
