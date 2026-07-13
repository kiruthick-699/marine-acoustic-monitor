"""
edge/ -- production Pi-side implementation.

Unlike simulation/ (which stands in synthetic hardware to prove the
ml-pipeline design end-to-end), this package is the actual deployable
software: the same capture -> condition -> extract -> detect -> store ->
telemeter loop from docs/data-pipeline.md, wired against a Hardware
Abstraction Layer (edge/hal/) instead of synthetic generators.

Why this can be built and tested before any hardware is purchased
(DECISIONS.md: no hardware bought yet): edge/hal/mock.py implements every
HAL interface with realistic stand-in data (reusing simulation/data_generator
internally), so the entire orchestration -- scheduling, storage, telemetry
payload shape, calibration -- is exercised and unit-tested today. Swapping to
real hardware later is a matter of writing edge/hal/real/*.py drivers against
the same interfaces (edge/hal/interfaces.py) and flipping one config flag
(edge/config.yaml: hardware.mode) -- no change to capture_loop.py,
telemetry.py, calibration.py, or scheduler.py.

This package deliberately imports and reuses simulation/pipeline/* (signal
conditioning, feature extraction, anomaly detection, SQLite storage)
unchanged -- that code is not "simulation-only", only simulation/data_generator/
is. See docs/pi-implementation.md for the wiring/address plan and how to run
this in mock mode today vs. real mode once hardware exists.
"""
