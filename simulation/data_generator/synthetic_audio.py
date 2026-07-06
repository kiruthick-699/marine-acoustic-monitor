"""
Synthetic hydrophone audio generator.

Produces synthetic underwater audio resembling what a real hydrophone would
capture at a monitoring buoy/dock site: ambient ocean noise, optionally with
an injected biological call (whistle + click train, dolphin/odontocete-like)
or a vessel noise event (tonal propeller/engine harmonics + broadband
cavitation noise). This exists so the on-device feature-extraction and
anomaly-detection pipeline (see docs/ml-pipeline.md) can be built and tested
before any real hardware or field recordings exist -- see DECISIONS.md,
project status is planning/no hardware yet.

None of this claims to be a physically exact underwater acoustics simulator.
It is a simplified generator whose events are spectrally and temporally
distinct enough to exercise a feature-extraction + anomaly-detection
pipeline meaningfully, with the specific choices explained inline.
"""

from typing import Optional, Tuple

import numpy as np
from scipy import signal


def generate_ambient_background(
    duration_s: float, sample_rate: int, color: str = "pink"
) -> np.ndarray:
    """
    Generate ambient ocean background noise.

    Real underwater ambient noise (wind, waves, distant shipping, biological
    chorus) is not flat/white -- its energy is concentrated at lower
    frequencies and falls off as frequency increases, roughly following a
    1/f (pink) or 1/f^2 (brown/red) power spectral density. Plain white
    noise sounds unnaturally flat and "hissy" compared to a real ocean
    recording, so we shape white noise's spectrum instead of using it
    directly.

    Method: generate white noise, take its FFT, scale each frequency bin's
    magnitude by 1/sqrt(f) (which gives 1/f power, i.e. pink) or 1/f (which
    gives 1/f^2 power, i.e. brown), then inverse-FFT back to the time
    domain. This is a standard, cheap way to produce colored noise without
    designing a recursive filter.

    Args:
        duration_s: length of the background clip, in seconds.
        sample_rate: samples per second (Hz).
        color: "pink" (1/f power) or "brown" (1/f^2 power). Pink is the
            closer match to typical shallow-water ambient noise spectra;
            brown is offered as an alternative for deeper/quieter sites.

    Returns:
        1D numpy array (float32) of audio samples, scaled to a quiet
        "noise floor" amplitude so later additive events have a
        predictable baseline to sit on top of.
    """
    if color not in ("pink", "brown"):
        raise ValueError(f"Unknown noise color: {color!r}. Use 'pink' or 'brown'.")

    n_samples = int(duration_s * sample_rate)

    white = np.random.normal(0, 1, n_samples)

    # rfft/rfftfreq: real-input FFT, only the non-negative frequencies
    # (0 .. Nyquist) are needed since the input is real-valued.
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

    # avoid a divide-by-zero at DC (freq = 0 Hz); reuse the first nonzero
    # bin's frequency so the DC component gets a large-but-finite scale
    # like its neighbor, rather than blowing up.
    if len(freqs) > 1:
        freqs[0] = freqs[1]
    else:
        freqs[0] = 1.0

    if color == "pink":
        scale = 1.0 / np.sqrt(freqs)
    else:  # brown
        scale = 1.0 / freqs

    shaped_spectrum = spectrum * scale
    colored = np.fft.irfft(shaped_spectrum, n=n_samples)

    # normalize to unit peak, then scale down to a quiet noise floor --
    # ambient background should be clearly quieter than any injected
    # biological/vessel event, matching how a real hydrophone recording
    # has a low-level background under any louder foreground event.
    colored = colored / (np.max(np.abs(colored)) + 1e-12)
    colored *= 0.15

    return colored.astype(np.float32)


def inject_biological_call(
    audio: np.ndarray,
    sample_rate: int,
    onset_s: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Add a synthetic biological vocalization into an existing audio buffer.

    Many marine biological sounds are frequency-modulated tonal sweeps --
    pitch rising and/or falling smoothly over tens to hundreds of
    milliseconds, as in dolphin whistles. Many others -- odontocete
    echolocation, snapping shrimp -- are short broadband clicks repeated in
    a train. This function synthesizes a simplified version of both: one
    upward FM "whistle" chirp, followed by a short click train. It stands in
    for a generic biological event, not any specific species' real call.

    Args:
        audio: existing 1D audio buffer to inject into. The input is not
            modified; a modified copy is returned.
        sample_rate: samples per second (Hz), must match `audio`'s rate.
        onset_s: time (seconds) within `audio` at which the call begins.
            If None, a random onset is chosen that leaves room for the
            full call to fit before the buffer ends.

    Returns:
        (modified_audio, onset_s): the audio with the call added, and the
        actual onset time used. Returning the resolved onset matters when
        it was chosen randomly, so the caller can record it as ground
        truth for later evaluation.
    """
    out = audio.copy()
    n_total = len(out)
    total_duration_s = n_total / sample_rate

    call_duration_s = 0.35  # whistle + click train together, a short event

    if onset_s is None:
        latest_onset = max(total_duration_s - call_duration_s - 0.05, 0.0)
        onset_s = float(np.random.uniform(0.0, latest_onset)) if latest_onset > 0 else 0.0

    onset_sample = int(onset_s * sample_rate)

    # --- whistle: upward FM chirp, loosely dolphin-whistle-like ---
    whistle_duration_s = 0.2
    t = np.linspace(
        0, whistle_duration_s, int(whistle_duration_s * sample_rate), endpoint=False
    )
    # Sweep 4 kHz -> 12 kHz: within the range typical of dolphin whistles,
    # and well above the low-frequency energy dominating ambient noise and
    # vessel noise (see inject_vessel_noise_event), so this event is
    # spectrally distinct and easy for feature extraction (e.g. spectral
    # centroid) to separate from the other signal types.
    whistle = signal.chirp(t, f0=4000, f1=12000, t1=whistle_duration_s, method="linear")
    # A Hann envelope ramps the whistle up and back down smoothly. Real
    # calls don't start/stop with a hard edge, and a hard edge would itself
    # inject broadband "click" energy we don't intend here.
    whistle *= np.hanning(len(whistle))
    whistle *= 0.6  # louder than the ambient floor so it's detectable

    # --- click train: short broadband impulses, echolocation-like ---
    n_clicks = 5
    click_width_s = 0.001  # ~1 ms: short and broadband by design
    click_spacing_s = 0.03
    click_train = np.zeros(
        int((n_clicks * click_spacing_s + click_width_s) * sample_rate)
    )
    click_width_samples = max(int(click_width_s * sample_rate), 1)
    for i in range(n_clicks):
        start = int(i * click_spacing_s * sample_rate)
        # A short burst of noise stands in for a broadband click. Real
        # clicks are impulsive and spectrally broad -- noise approximates
        # that better than a single pure tone would.
        click_train[start : start + click_width_samples] += np.random.normal(
            0, 1, click_width_samples
        )
    click_train *= 0.5

    call = np.concatenate([whistle, click_train])
    call_len = len(call)

    end_sample = min(onset_sample + call_len, n_total)
    usable_len = end_sample - onset_sample
    if usable_len > 0:
        out[onset_sample:end_sample] += call[:usable_len]

    return out, onset_s


def inject_vessel_noise_event(
    audio: np.ndarray,
    sample_rate: int,
    onset_s: float,
    duration_s: float,
) -> np.ndarray:
    """
    Add a synthetic vessel-passage noise event into an existing audio buffer.

    Commercial vessel underwater noise is dominated by low-frequency tonal
    peaks from propeller shaft rotation and blade-rate harmonics, riding on
    top of broadband cavitation/machinery noise. The 60/90/120 Hz tonal
    components below are representative peak locations drawn from published
    commercial vessel underwater noise characterization studies -- dominant
    tonal energy clustering roughly in the 60-120 Hz band for many
    commercial vessel classes -- not a specific vessel's measured spectrum.

    Args:
        audio: existing 1D audio buffer to inject into. The input is not
            modified; a modified copy is returned.
        sample_rate: samples per second (Hz), must match `audio`'s rate.
        onset_s: time (seconds) within `audio` at which the vessel noise
            begins to ramp up (start of approach).
        duration_s: total duration (seconds) of the vessel passage event,
            covering approach, closest point of approach, and departure.

    Returns:
        modified_audio: the audio with the vessel event added.
    """
    out = audio.copy()
    n_total = len(out)
    onset_sample = int(onset_s * sample_rate)
    n_event = int(duration_s * sample_rate)
    end_sample = min(onset_sample + n_event, n_total)
    usable_len = end_sample - onset_sample
    if usable_len <= 0:
        return out

    t = np.arange(usable_len) / sample_rate

    # Tonal components at 60/90/120 Hz: representative shaft/blade-rate
    # harmonic peak locations for commercial vessel classes, per real
    # vessel noise characterization studies referenced above. Weighted
    # descending (1.0 / 0.7 / 0.5) since real harmonic series typically
    # decrease in level at higher harmonics, not because these exact ratios
    # were measured for a specific vessel.
    tonal = (
        np.sin(2 * np.pi * 60 * t)
        + 0.7 * np.sin(2 * np.pi * 90 * t)
        + 0.5 * np.sin(2 * np.pi * 120 * t)
    )
    tonal /= 2.2  # rough normalization across the three summed components

    # Broadband component: cavitation and machinery noise is noise-like
    # rather than tonal, so unshaped white noise stands in for it here
    # (unlike the ambient background, which is deliberately colored).
    broadband = np.random.normal(0, 1, usable_len)
    broadband /= np.max(np.abs(broadband)) + 1e-12

    vessel_signal = 0.7 * tonal + 0.3 * broadband

    # Amplitude envelope: rises then falls, modeling underwater received
    # level as a vessel approaches (decreasing range -> increasing level),
    # passes its closest point of approach, then departs (increasing range
    # -> decreasing level). A Hann window gives a smooth rise-peak-fall
    # shape with no hard edges, matching that qualitative passage shape.
    envelope = np.hanning(usable_len)

    event = vessel_signal * envelope * 0.8  # dominant over ambient during passage

    out[onset_sample:end_sample] += event
    return out


def generate_duty_cycle_sample(
    duration_s: float = 30,
    sample_rate: int = 22050,
    inject_anomaly: Optional[str] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Generate one full duty-cycle capture window of synthetic hydrophone
    audio, matching the "record N seconds every M minutes" sampling model
    from DECISIONS.md.

    This is the top-level entry point simulation scripts/tests should call.
    It builds the ambient background and optionally injects one anomaly
    event ("biological" or "vessel"), returning both the audio and ground
    truth metadata so a downstream feature-extraction/anomaly-detection run
    on this sample can be scored against a known answer.

    Args:
        duration_s: length of the capture window, in seconds. Default 30s
            as a representative baseline duty-cycle window length.
        sample_rate: samples per second (Hz). Default 22050 Hz is more than
            double the ~12 kHz top of the synthetic whistle sweep used in
            `inject_biological_call`, satisfying the Nyquist rate with
            headroom while keeping data volume modest compared to full
            44.1/48 kHz audio rates.
        inject_anomaly: one of:
            - None: pure ambient background, no anomaly (negative example).
            - "biological": inject one biological call at a random onset.
            - "vessel": inject one vessel noise event at a random onset,
              lasting roughly a third of the window's duration.

    Returns:
        (audio, metadata): `audio` is the 1D synthetic audio array (float32)
        for the window. `metadata` is a dict of ground truth for evaluation:
            {
                "duration_s": float,
                "sample_rate": int,
                "anomaly_injected": bool,
                "anomaly_type": str or None,
                "onset_s": float or None,
                "event_duration_s": float or None,
            }
    """
    if inject_anomaly not in (None, "biological", "vessel"):
        raise ValueError(
            f"inject_anomaly must be None, 'biological', or 'vessel', got {inject_anomaly!r}"
        )

    audio = generate_ambient_background(duration_s, sample_rate)

    metadata = {
        "duration_s": duration_s,
        "sample_rate": sample_rate,
        "anomaly_injected": inject_anomaly is not None,
        "anomaly_type": inject_anomaly,
        "onset_s": None,
        "event_duration_s": None,
    }

    if inject_anomaly == "biological":
        audio, onset_s = inject_biological_call(audio, sample_rate, onset_s=None)
        metadata["onset_s"] = onset_s
        # matches call_duration_s in inject_biological_call (whistle + click train)
        metadata["event_duration_s"] = 0.35

    elif inject_anomaly == "vessel":
        event_duration_s = min(duration_s / 3.0, duration_s)
        latest_onset = max(duration_s - event_duration_s, 0.0)
        onset_s = float(np.random.uniform(0.0, latest_onset)) if latest_onset > 0 else 0.0
        audio = inject_vessel_noise_event(audio, sample_rate, onset_s, event_duration_s)
        metadata["onset_s"] = onset_s
        metadata["event_duration_s"] = event_duration_s

    return audio, metadata
