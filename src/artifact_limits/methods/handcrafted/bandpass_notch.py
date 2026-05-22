"""H1 — bandpass + notch baseline (no artifact-targeted operation)."""
from __future__ import annotations

import numpy as np
import scipy.signal as sp

from artifact_limits import SAMPLING_FREQ
from artifact_limits.methods.base import DenoiserBase


class BandpassNotch(DenoiserBase):
    name = "bandpass_notch"

    def __init__(self, low: float = 0.5, high: float = 45.0, line: float = 50.0,
                 fs: int = SAMPLING_FREQ):
        self.fs = fs
        nyq = fs / 2.0
        self.b_bp, self.a_bp = sp.butter(4, [low / nyq, min(high, nyq - 1) / nyq], "bandpass")
        self.b_n, self.a_n = sp.iirnotch(line, Q=30, fs=fs)

    def fit(self, s_pool: np.ndarray, a_pool: np.ndarray) -> "BandpassNotch":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        y = sp.filtfilt(self.b_bp, self.a_bp, x, axis=-1)
        y = sp.filtfilt(self.b_n, self.a_n, y, axis=-1)
        return y.astype(np.float32)
