# Hardware Specification

Status: planning/architecture phase. No hardware purchased. This document is spec-level — component roles and interface requirements, not a bill of materials. Consistent with [DECISIONS.md](../DECISIONS.md) and [architecture.md](architecture.md).

## Component list and role

**Hydrophone (acoustic sensor)**
Primary sensing input. Captures underwater acoustic signal in scheduled duty-cycle windows (record N seconds every M minutes). Feeds the on-device FFT + feature extraction stage (MFCCs, spectral centroid, ZCR, RMS energy, spectral flatness) and, at full resolution, the local raw-audio store.

**Environmental sensors: temperature, pH, turbidity, salinity**
Secondary sensing inputs, sampled alongside each acoustic window. Values (plus rate-of-change) are normalized and concatenated with acoustic features into the joint feature vector used for anomaly detection. Also logged independently as structured environmental readings in SQLite, and included in the compact telemetry payload.

**IMU**
Provides orientation/motion data. On a buoy, distinguishes wave/mooring motion from genuine environmental or acoustic anomalies, and supports system health logging (e.g. detecting tilt/capsize risk, impact events). Not part of the core acoustic anomaly-detection feature vector unless later shown useful; treated as system-health/context sensor.

**Raspberry Pi (edge compute)**
Central compute node. Runs capture scheduling, FFT/feature extraction, unsupervised anomaly detection, local storage writes (flat audio files + SQLite WAL), and telemetry payload assembly/transmission. Only compute element in the system — no other onboard processor.

**Telemetry module (LoRa or low-bandwidth cellular)**
Transmits compact payloads only: feature summaries, environmental readings, anomaly alerts, on the same duty cycle as sampling. Does not carry raw audio. Choice between LoRa and low-bandwidth cellular is a deployment-specific decision (range, existing gateway infrastructure, cellular coverage at site), not fixed at the architecture level.

**Power system (solar panel + battery for buoy; solar+battery or shore power for dock)**
Sized to sustain duty-cycle sampling, edge compute bursts, and telemetry transmission between charge cycles. See [architecture.md](architecture.md) for the deployment-profile power notes.

## Key interface considerations

**Hydrophone / ADC**
Hydrophone output is analog and low-level, requiring a dedicated ADC (or ADC-equipped preamp) ahead of the Pi's GPIO — the Pi has no native high-quality analog audio input. ADC must support a sample rate sufficient for the frequency range of interest (bioacoustic signals of interest typically extend well above what a basic audio codec assumes) and sufficient bit depth to preserve dynamic range for both quiet ambient and loud transient events. Interface to the Pi is most likely I2S or SPI, not USB, to minimize latency and power overhead — final bus choice deferred to implementation.

**Environmental sensor bus**
Temperature, pH, turbidity, and salinity sensors are typically I2C or analog-with-ADC devices. I2C is preferred where available for multi-sensor sharing over one bus (fewer GPIO pins, simpler wiring), with attention to address collisions if multiple sensors share default addresses. Sensors without native I2C output route through the same ADC used for auxiliary analog inputs, or a separate low-channel-count ADC if the hydrophone ADC is fully committed.

**IMU bus**
IMUs are standard I2C or SPI devices; low pin/power overhead, straightforward to share the same I2C bus as environmental sensors if address space allows.

**Power draw implications of duty-cycle sampling**
System is not continuously active at full draw — draw profile has two states: (1) sleep/idle between wake windows, minimal draw, sensors and compute powered down or in low-power mode; (2) active window, higher draw during capture + FFT/feature extraction + anomaly detection + telemetry transmission. Telemetry transmission is typically the highest instantaneous draw event in the cycle (especially cellular over LoRa). Duty cycle (N seconds every M minutes) is the primary lever for average power consumption and must be balanced against solar charge input and battery capacity for the buoy profile; sizing itself belongs in a future power budget, not this document.

## Enclosure / watertight considerations (buoy reference platform)

- Pi, ADC, telemetry module, and power electronics require a sealed, waterproof enclosure rated for continuous partial or full submersion exposure (splash, spray, and wave wash at minimum; full submersion resistance if mounted low on the buoy).
- Hydrophone and environmental sensors are the only elements intentionally in direct water contact — their cabling/connectors into the sealed electronics enclosure are the primary watertight risk point and require waterproof (marine-rated) connectors or a properly potted cable entry, not just a sealed box.
- Enclosure must accommodate thermal dissipation for the Pi under active-window load without compromising the seal (passive conduction to enclosure body, not fan-based active cooling).
- Biofouling on the hydrophone and environmental sensor surfaces will affect data quality over time; enclosure/mounting design should keep sensing surfaces accessible for cleaning at maintenance visits, without requiring the sealed electronics enclosure to be opened.
- Antenna (telemetry module) either mounted external to the sealed enclosure with a waterproof bulkhead feedthrough, or the enclosure material/design must not attenuate the telemetry signal if the antenna stays internal.
- Solar panel is inherently external and exposed; its wiring into the sealed enclosure is a second watertight cable-entry point distinct from the sensor entries above.
- Dock/pier profile has materially relaxed watertight requirements (typically above waterline, less wave exposure) but should use the same enclosure design as the buoy profile to keep hardware interchangeable across deployment profiles, per [architecture.md](architecture.md).
