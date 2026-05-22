"""A library of spectral templates for the D1' construction.

The default ``build_D1prime`` uses a 1-Hz slow-wave \(e_S\) and an eye-movement
\(e_A\). To verify that the dependent-mixing collapse is not specific to that
choice, this module provides:

* pure sinusoids at multiple frequencies and random phases,
* Gabor wavelets at multiple centre frequencies,
* random structured patterns (low-pass white noise of varying smoothness),

each as a ``(name, template)`` pair so a downstream experiment can sweep.
"""
from __future__ import annotations

import numpy as np

from artifact_limits import SAMPLING_FREQ, SEGMENT_LEN


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def sinusoid_template(freq_hz: float, phase: float = 0.0,
                       T: int = SEGMENT_LEN, fs: int = SAMPLING_FREQ) -> np.ndarray:
    t = np.arange(T) / fs
    return _unit(np.sin(2 * np.pi * freq_hz * t + phase).astype(np.float32))


def gabor_template(freq_hz: float, sigma_s: float = 0.25, phase: float = 0.0,
                    T: int = SEGMENT_LEN, fs: int = SAMPLING_FREQ) -> np.ndarray:
    t = np.arange(T) / fs - (T / fs) / 2.0
    env = np.exp(-0.5 * (t / sigma_s) ** 2)
    car = np.sin(2 * np.pi * freq_hz * t + phase)
    return _unit((env * car).astype(np.float32))


def random_structured_template(seed: int, smoothness: float = 0.02,
                                T: int = SEGMENT_LEN,
                                fs: int = SAMPLING_FREQ) -> np.ndarray:
    """Low-pass white noise: Gaussian iid then heavy low-pass filter."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(T)
    freqs = np.fft.rfftfreq(T, 1.0 / fs)
    spec = np.fft.rfft(raw)
    spec = spec * np.exp(-smoothness * freqs ** 2)
    return _unit(np.fft.irfft(spec, T).astype(np.float32))


def template_library(seed: int = 0) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Returns a list of (name, e_S, e_A) triples for the D1' sweep.

    Each pair is informed by a different family of coupling: sinusoid pairs at
    matched frequency but different phase model in-phase oscillation; Gabors
    capture transient burst coupling; random structured patterns capture
    diffuse low-frequency shared drift.
    """
    rng = np.random.default_rng(seed)
    out: list[tuple[str, np.ndarray, np.ndarray]] = []

    for f in (0.5, 1.0, 2.0, 4.0, 8.0):
        phase = float(rng.uniform(0, 2 * np.pi))
        out.append((f"sin_{f}Hz_phase_match",
                     sinusoid_template(f, phase=phase),
                     sinusoid_template(f, phase=phase)))
        out.append((f"sin_{f}Hz_phase_opposite",
                     sinusoid_template(f, phase=phase),
                     sinusoid_template(f, phase=phase + np.pi / 2)))

    for f in (1.0, 4.0, 12.0):
        out.append((f"gabor_{f}Hz",
                     gabor_template(f, sigma_s=0.25),
                     gabor_template(f, sigma_s=0.25, phase=np.pi / 4)))

    for k in range(3):
        out.append((f"random_struct_{k}",
                     random_structured_template(seed=seed + 100 + k),
                     random_structured_template(seed=seed + 200 + k)))

    return out
