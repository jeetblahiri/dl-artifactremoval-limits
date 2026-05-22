"""Inference for autocorrelated time series.

Two utilities aimed at the E5 in-vivo audit:

* ``block_bootstrap_ci`` --- non-overlapping moving-block bootstrap that
  resamples *blocks* of consecutive windows rather than individual windows,
  so the resampled series preserves short-range autocorrelation.

* ``circular_shift_null`` --- generates a null distribution of the estimator
  under the hypothesis ``MI(X; Y) = 0`` by *circularly shifting* one of the
  channels by a random offset. Unlike iid permutation this preserves each
  channel's autocorrelation while destroying cross-channel synchrony.

Both are estimator-agnostic; pass any function ``estimator(fx, fy) -> float``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def block_bootstrap_ci(fx: np.ndarray, fy: np.ndarray,
                        estimator: Callable[[np.ndarray, np.ndarray], float],
                        n_boot: int = 200,
                        block_len: int = 3,
                        alpha: float = 0.05,
                        seed: int = 0) -> tuple[float, float, float, np.ndarray]:
    """Moving-block bootstrap on paired feature matrices.

    ``fx``, ``fy`` are (n_windows, d) feature matrices for the two channels.
    Blocks of ``block_len`` consecutive windows are resampled with replacement
    until the bootstrap sample length matches the original. Returns the point
    estimate, the lower/upper percentile bounds at ``alpha``, and the full
    bootstrap distribution.
    """
    rng = np.random.default_rng(seed)
    fx = np.asarray(fx)
    fy = np.asarray(fy)
    n = fx.shape[0]
    if n < block_len:
        # Fall back to iid bootstrap if block doesn't fit.
        block_len = 1
    n_blocks = int(np.ceil(n / block_len))

    point = float(estimator(fx, fy))
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, max(n - block_len + 1, 1), size=n_blocks)
        idx_parts = [np.arange(s, s + block_len) for s in starts]
        idx = np.concatenate(idx_parts)[:n]
        boots[b] = estimator(fx[idx], fy[idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi), boots


def circular_shift_null(fx: np.ndarray, fy: np.ndarray,
                         estimator: Callable[[np.ndarray, np.ndarray], float],
                         n_perm: int = 200,
                         min_shift: int = 1,
                         seed: int = 0) -> np.ndarray:
    """Circular-shift null: shift ``fy`` by a random offset and recompute MI.

    Preserves ``fy``'s autocorrelation; destroys ``(fx, fy)`` cross-channel
    synchrony. Shift uniformly in ``[min_shift, n - min_shift]``.
    """
    rng = np.random.default_rng(seed)
    fx = np.asarray(fx)
    fy = np.asarray(fy)
    n = fx.shape[0]
    if n <= 2 * min_shift:
        # Too few windows for non-trivial shift; fall back to permutation.
        out = np.empty(n_perm, dtype=float)
        for b in range(n_perm):
            out[b] = estimator(fx, fy[rng.permutation(n)])
        return out
    out = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        k = int(rng.integers(min_shift, n - min_shift + 1))
        shifted = np.roll(fy, k, axis=0)
        out[b] = estimator(fx, shifted)
    return out
