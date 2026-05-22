"""H2 — Discrete Wavelet Transform thresholding (Donoho–Johnstone universal)."""
from __future__ import annotations

import numpy as np
import pywt

from artifact_limits.methods.base import DenoiserBase


class DWTThresholding(DenoiserBase):
    name = "dwt"

    def __init__(self, wavelet: str = "db4", level: int = 5, mode: str = "soft",
                 threshold_scale: float = 1.0):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.threshold_scale = float(threshold_scale)

    def fit(self, s_pool, a_pool) -> "DWTThresholding":
        return self

    def _denoise_one(self, x: np.ndarray) -> np.ndarray:
        coeffs = pywt.wavedec(x, self.wavelet, level=self.level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        T = self.threshold_scale * sigma * np.sqrt(2 * np.log(max(len(x), 2)))
        new_coeffs = [coeffs[0]] + [pywt.threshold(c, T, mode=self.mode) for c in coeffs[1:]]
        rec = pywt.waverec(new_coeffs, self.wavelet)
        return rec[: x.shape[0]]

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros_like(x, dtype=np.float32)
        for i in range(x.shape[0]):
            out[i] = self._denoise_one(x[i]).astype(np.float32)
        return out
