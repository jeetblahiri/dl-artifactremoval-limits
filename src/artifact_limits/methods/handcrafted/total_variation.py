"""H5 — Total Variation denoising via Chambolle's algorithm."""
from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_tv_chambolle

from artifact_limits.methods.base import DenoiserBase


class TotalVariation(DenoiserBase):
    name = "total_variation"

    def __init__(self, weight: float = 0.1, max_iter: int = 200):
        self.weight = float(weight)
        self.max_iter = int(max_iter)

    def fit(self, s_pool, a_pool) -> "TotalVariation":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros_like(x, dtype=np.float32)
        for i in range(x.shape[0]):
            out[i] = denoise_tv_chambolle(x[i], weight=self.weight,
                                          max_num_iter=self.max_iter).astype(np.float32)
        return out
