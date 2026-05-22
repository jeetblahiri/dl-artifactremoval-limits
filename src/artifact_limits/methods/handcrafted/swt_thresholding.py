"""H3 — Stationary Wavelet Transform thresholding (translation-invariant)."""
from __future__ import annotations

import numpy as np
import pywt

from artifact_limits.methods.base import DenoiserBase


def _pad_to_pow2(x: np.ndarray) -> tuple[np.ndarray, int]:
    n = x.shape[-1]
    p = 1 << int(np.ceil(np.log2(n)))
    if p == n:
        return x, n
    pad = p - n
    return np.pad(x, ((0, 0), (0, pad)), mode="reflect"), n


class SWTThresholding(DenoiserBase):
    name = "swt"

    def __init__(self, wavelet: str = "db4", level: int = 4, mode: str = "soft",
                 threshold_scale: float = 1.0):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.threshold_scale = float(threshold_scale)

    def fit(self, s_pool, a_pool) -> "SWTThresholding":
        return self

    def _denoise_one(self, x: np.ndarray, T_full: int) -> np.ndarray:
        coeffs = pywt.swt(x, self.wavelet, level=self.level)
        new = []
        for cA, cD in coeffs:
            sigma = np.median(np.abs(cD)) / 0.6745
            T = self.threshold_scale * sigma * np.sqrt(2 * np.log(max(len(cD), 2)))
            new.append((cA, pywt.threshold(cD, T, mode=self.mode)))
        rec = pywt.iswt(new, self.wavelet)
        return rec[:T_full]

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        T_full = x.shape[-1]
        x_pad, _ = _pad_to_pow2(x)
        out = np.zeros_like(x, dtype=np.float32)
        for i in range(x.shape[0]):
            out[i] = self._denoise_one(x_pad[i], T_full).astype(np.float32)
        return out
