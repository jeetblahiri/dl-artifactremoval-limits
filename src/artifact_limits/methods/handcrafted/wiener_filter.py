"""H_W — Wiener filter (the theoretical floor materialised as a method)."""
from __future__ import annotations

import numpy as np

from artifact_limits.methods.base import DenoiserBase
from artifact_limits.theory.wiener_floor import apply_wiener, estimate_wiener_floor


class WienerFilter(DenoiserBase):
    name = "wiener_filter"

    def __init__(self, noise_var: float = 0.0):
        self.noise_var = float(noise_var)
        self.W: np.ndarray | None = None
        self.floor = None

    def fit(self, s_pool: np.ndarray, a_pool: np.ndarray) -> "WienerFilter":
        self.floor = estimate_wiener_floor(s_pool, a_pool, noise_var=self.noise_var)
        self.W = self.floor.W
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("WienerFilter must be .fit() before .transform().")
        return apply_wiener(np.asarray(x), self.W)
