"""H6 — Savitzky-Golay local polynomial smoothing."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from artifact_limits.methods.base import DenoiserBase


class SavGol(DenoiserBase):
    name = "savgol"

    def __init__(self, window: int = 21, order: int = 3):
        self.window = int(window) | 1   # force odd
        self.order = int(order)

    def fit(self, s_pool, a_pool) -> "SavGol":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return savgol_filter(np.asarray(x), self.window, self.order, axis=-1).astype(np.float32)
