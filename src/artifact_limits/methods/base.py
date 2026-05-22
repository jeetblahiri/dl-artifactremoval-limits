"""Abstract base class for every denoiser in the methods registry."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DenoiserBase(ABC):
    """Every method must conform to (fit, transform).

    fit:        called once on training pools (or training mixtures, for DL).
    transform:  applies the trained method to a batch of test mixtures.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, s_pool: np.ndarray, a_pool: np.ndarray) -> "DenoiserBase":
        ...

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, s_pool: np.ndarray, a_pool: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.fit(s_pool, a_pool).transform(x)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
