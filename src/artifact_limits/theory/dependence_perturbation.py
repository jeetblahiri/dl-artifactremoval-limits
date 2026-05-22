"""D1' construction wrapper that also reports the induced I(S; A).

Most of the heavy lifting is in `artifact_limits.data.build_D1prime`; this module
adds the empirical I(S̃; Ã) estimate used as the x-axis of Figure 3.
"""
from __future__ import annotations

import numpy as np

from artifact_limits.data.build_D1prime import construct_D1prime
from artifact_limits.theory.mi_estimators import gaussian_copula_mi, ksg_mi


def _segment_feature(seg: np.ndarray) -> np.ndarray:
    """Compact 4-D summary of a segment for MI estimation: variance, kurtosis,
    band-power [0–4Hz, 30–80Hz] proxies via PSD bin sums.
    """
    seg = np.asarray(seg, dtype=np.float64)
    n, T = seg.shape
    var = seg.var(axis=-1)
    mu = seg.mean(axis=-1, keepdims=True)
    kurt = ((seg - mu) ** 4).mean(axis=-1) / (var ** 2 + 1e-12)
    spec = np.abs(np.fft.rfft(seg, axis=-1)) ** 2
    F = spec.shape[1]
    lo = spec[:, :max(1, F // 10)].sum(axis=-1)
    hi = spec[:, max(1, 3 * F // 4):].sum(axis=-1)
    return np.stack([var, kurt, lo, hi], axis=1)


def perturb_with_mi(s_pool: np.ndarray,
                    a_pool: np.ndarray,
                    alpha: float,
                    seed: int = 42,
                    mi_method: str = "gauss_copula") -> dict:
    out = construct_D1prime(s_pool, a_pool, alpha=alpha, seed=seed)
    fs = _segment_feature(out["s_tilde"])
    fa = _segment_feature(out["a_tilde"])
    if mi_method == "ksg":
        mi = ksg_mi(fs, fa, k=4)
    else:
        mi = gaussian_copula_mi(fs, fa)
    out["I_SA_estimate"] = float(mi)
    return out
