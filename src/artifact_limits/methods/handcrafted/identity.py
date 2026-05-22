"""H0 — identity (no-denoising lower anchor)."""
from __future__ import annotations

import numpy as np

from artifact_limits.methods.base import DenoiserBase


class Identity(DenoiserBase):
    name = "identity"

    def fit(self, s_pool: np.ndarray, a_pool: np.ndarray) -> "Identity":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)
