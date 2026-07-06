# Marine Ecosystem Monitoring System

**Status: (In Development) — planning/architecture phase. No hardware purchased, no implementation code written yet.** Everything in this repo is design documentation. See [DECISIONS.md](DECISIONS.md) for the locked design decisions this documentation is built against.

![Simulation results](diagrams/results-chart.svg)

| Precision | Recall | F1 |
|-----------|--------|-----|
| 0.937 | 0.808 | 0.868 |

Caveat: these numbers are from one simulation run with an unusually high injected anomaly rate (a storm runoff event covered most of the evaluation window), not a stable benchmark. See [simulation/README.md](simulation/README.md) for methodology.

## Motivation

Marine ecosystems generate continuous acoustic and environmental signal — biological activity, equipment/vessel noise, water-quality shifts, anomalous events — that's expensive to monitor with human observers or connectivity-dependent systems. This project designs a low-cost, edge-computed monitoring unit that combines passive acoustic monitoring with environmental sensing (temperature, pH, turbidity, salinity), correlates them on-device into a single feature space, and flags anomalies locally without depending on continuous high-bandwidth connectivity. The goal is a platform that's deployable on a solar-powered buoy or a fixed dock/pier mount using the same core hardware and software, reporting compact summaries over a low-bandwidth link while retaining full-resolution data for later retrieval.

## Documentation

- [DECISIONS.md](DECISIONS.md) — locked design decisions; the reference other docs are kept consistent with.
- [docs/architecture.md](docs/architecture.md) — full system architecture: sensor -> Pi -> processing -> telemetry -> storage/output flow, edge/offsite compute split, buoy/dock deployment profiles.
- [docs/hardware-spec.md](docs/hardware-spec.md) — component list and roles, interface considerations, enclosure/watertight considerations. Spec-level, no BOM yet.
- [docs/data-pipeline.md](docs/data-pipeline.md) — storage tiers, duty-cycle sampling flow, bulk retrieval process, SQLite schema sketch.
- [docs/ml-pipeline.md](docs/ml-pipeline.md) — 3-stage ML pipeline: feature extraction, unsupervised anomaly detection, future supervised classification.
- [docs/related-work.md](docs/related-work.md) — prior art and how this project's design differs.

## Diagrams

`/diagrams` — reserved for architecture/data-flow diagrams derived from the docs above (not yet created).

## Documentation map

Suggested reading order, with why each doc sits where it does:

1. [docs/architecture.md](docs/architecture.md) — start here: the whole-system picture (sensor -> Pi -> processing -> telemetry -> storage/output) everything else is a component or detail of.
2. [docs/data-pipeline.md](docs/data-pipeline.md) — the storage/data-flow half of that architecture in detail: duty-cycle sampling, storage tiers, SQLite schema.
3. [docs/ml-pipeline.md](docs/ml-pipeline.md) — what happens to the data data-pipeline.md describes once it reaches Stage 1/2 feature extraction and anomaly detection.
4. [simulation/README.md](simulation/README.md) — the ml-pipeline.md design as running code: synthetic data standing in for real hardware, exercising the same pipeline end-to-end.
5. [docs/hardware-spec.md](docs/hardware-spec.md) — the physical components the architecture and pipeline above assume, once you want to know what they'd run on.
6. [docs/related-work.md](docs/related-work.md) — how the above design choices relate to existing prior art, useful once the design itself is understood.
7. [DECISIONS.md](DECISIONS.md) — last as a reference, not reading material: the locked decisions everything above is kept consistent with, for looking something up rather than reading start to end.
