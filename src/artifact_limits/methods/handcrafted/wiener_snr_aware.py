r"""SNR-aware Wiener-filter variants.

The default ``WienerFilter`` is built from \(\Sigma_S\) and \(\Sigma_A\) at unit
artifact scale; under the EEGdenoiseNet protocol the test mixture is
\(x = s + \lambda a\) with \(\lambda^2 = 10^{-\mathrm{SNR}/10}\). Two natural
variants close that gap:

* ``WienerFilterKnownSNR`` --- one filter per SNR level, used at matched SNR;
  the per-SNR Bayes-optimal linear estimator. Requires knowing the test SNR.
* ``WienerFilterSNRMarginal`` --- one filter optimised under the SNR mixture
  distribution. Does not need to know the test SNR.

Both reduce the SNR-marginalisation cost of the default filter and isolate the
``non-Gaussian'' gap from the ``SNR-averaging'' gap in the headline excess.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from artifact_limits import SNR_DB_LEVELS
from artifact_limits.methods.base import DenoiserBase
from artifact_limits.theory.wiener_floor import estimate_wiener_floor


def _lam_sq(snr_db: float) -> float:
    """For per-segment unit-variance pools, λ² = 10^{-SNR/10}."""
    return float(10.0 ** (-float(snr_db) / 10.0))


class WienerFilterKnownSNR(DenoiserBase):
    r"""Per-SNR Wiener filter applied at matched test-time SNR.

    At ``fit`` time we precompute one (T, T) Wiener matrix per SNR level
    using \(\Sigma_S + \lambda^2 \Sigma_A\). At ``transform`` time the caller
    supplies a per-segment ``snr_db`` vector; we apply the matching filter.

    If ``snr_db`` is not supplied to ``transform``, we use the closest level
    from ``snr_db_levels`` to the empirical per-segment SNR (estimated from x
    against s_pool variance) --- this is informative for the reference value
    but should be treated as oracle.
    """
    name = "wiener_known_snr"

    def __init__(self, snr_db_levels: Iterable[float] = SNR_DB_LEVELS,
                 noise_var: float = 0.0):
        self.snr_db_levels = tuple(float(s) for s in snr_db_levels)
        self.noise_var = float(noise_var)
        self.W_by_snr: dict[float, np.ndarray] = {}

    def fit(self, s_pool: np.ndarray, a_pool: np.ndarray) -> "WienerFilterKnownSNR":
        self.W_by_snr.clear()
        for snr in self.snr_db_levels:
            lam = float(np.sqrt(_lam_sq(snr)))
            floor = estimate_wiener_floor(s_pool, a_pool,
                                          noise_var=self.noise_var,
                                          artifact_scale=lam)
            self.W_by_snr[float(snr)] = floor.W.astype(np.float32)
        return self

    def transform(self, x: np.ndarray, snr_db: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if snr_db is None:
            # Fall back to applying the average filter (poor man's marginal).
            W_avg = np.mean(list(self.W_by_snr.values()), axis=0)
            return (x @ W_avg.T).astype(np.float32)
        snr_db = np.asarray(snr_db, dtype=np.float64)
        out = np.zeros_like(x, dtype=np.float32)
        # Group test segments by nearest SNR level for batched matmuls.
        levels = np.array(list(self.W_by_snr.keys()))
        for level in levels:
            mask = np.isclose(snr_db, level)
            if not mask.any():
                continue
            W = self.W_by_snr[float(level)]
            out[mask] = (x[mask] @ W.T).astype(np.float32)
        # Any segment whose snr did not match a level uses the closest.
        unmatched = ~np.isin(snr_db, levels)
        if unmatched.any():
            for i in np.where(unmatched)[0]:
                level = float(levels[np.argmin(np.abs(levels - snr_db[i]))])
                W = self.W_by_snr[level]
                out[i] = (x[i] @ W.T).astype(np.float32)
        return out


class WienerFilterSNRMarginal(DenoiserBase):
    r"""Wiener filter optimised over the SNR-mixture distribution.

    For SNR levels \(s_1,\dots,s_k\) sampled uniformly at training/test time,
    we set \(\Sigma_X = \Sigma_S + \overline{\lambda^2}\,\Sigma_A\) where
    \(\overline{\lambda^2} = \tfrac{1}{k}\sum_i \lambda^2(s_i)\). The
    resulting filter does *not* need the test SNR and is the proper
    apples-to-apples linear comparator for SNR-marginal deep methods.
    """
    name = "wiener_snr_marginal"

    def __init__(self, snr_db_levels: Iterable[float] = SNR_DB_LEVELS,
                 noise_var: float = 0.0):
        self.snr_db_levels = tuple(float(s) for s in snr_db_levels)
        self.noise_var = float(noise_var)
        self.W: np.ndarray | None = None

    def fit(self, s_pool: np.ndarray, a_pool: np.ndarray) -> "WienerFilterSNRMarginal":
        mean_lam_sq = float(np.mean([_lam_sq(s) for s in self.snr_db_levels]))
        floor = estimate_wiener_floor(s_pool, a_pool,
                                      noise_var=self.noise_var,
                                      artifact_scale=float(np.sqrt(mean_lam_sq)))
        self.W = floor.W.astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("WienerFilterSNRMarginal must be fit before transform.")
        x = np.asarray(x, dtype=np.float64)
        return (x @ self.W.T).astype(np.float32)
