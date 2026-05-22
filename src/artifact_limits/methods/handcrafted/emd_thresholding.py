"""H4 — Empirical Mode Decomposition + thresholding of high-frequency IMFs."""
from __future__ import annotations

import numpy as np

from artifact_limits.methods.base import DenoiserBase

try:                                                            # pragma: no cover - optional
    from PyEMD import EMD
    _HAS_PYEMD = True
except Exception:                                                # pragma: no cover - fallback
    EMD = None
    _HAS_PYEMD = False


class EMDThresholding(DenoiserBase):
    name = "emd"

    def __init__(self, n_imfs_to_threshold: int = 3, max_imfs: int = 8,
                 threshold_scale: float = 1.0):
        self.k = n_imfs_to_threshold
        self.max_imfs = max_imfs
        self.threshold_scale = float(threshold_scale)

    def fit(self, s_pool, a_pool) -> "EMDThresholding":
        return self

    def _denoise_one(self, xi: np.ndarray) -> np.ndarray:
        if not _HAS_PYEMD:
            return xi
        imfs = EMD().emd(xi, max_imf=self.max_imfs)
        if imfs.ndim != 2 or imfs.shape[0] == 0:
            return xi
        for j in range(min(self.k, imfs.shape[0])):
            sigma = np.median(np.abs(imfs[j])) / 0.6745
            T = self.threshold_scale * sigma * np.sqrt(2 * np.log(max(len(xi), 2)))
            imfs[j] = np.sign(imfs[j]) * np.maximum(np.abs(imfs[j]) - T, 0.0)
        return imfs.sum(axis=0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros_like(x, dtype=np.float32)
        for i in range(x.shape[0]):
            out[i] = self._denoise_one(x[i]).astype(np.float32)
        return out
