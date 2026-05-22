"""Information-theoretic diagnostics for denoiser outputs.

For each segment-level summary, estimate:
- I(ŝ; s)   signal preservation
- I(ŝ; a)   artifact leakage
- I(ŝ; z)   latent leakage (when z is available, e.g. D1'(α))

We use per-segment summary statistics (variance, kurtosis, band-power
proxies) rather than raw waveforms, because the KSG estimator scales poorly
beyond a few dimensions.
"""
from __future__ import annotations

import numpy as np

from artifact_limits.theory.dependence_perturbation import _segment_feature
from artifact_limits.theory.mi_estimators import gaussian_copula_mi, ksg_mi


def _feat(x: np.ndarray) -> np.ndarray:
    return _segment_feature(x)


def info_diagnostics(s_hat: np.ndarray,
                     s: np.ndarray,
                     a: np.ndarray,
                     z: np.ndarray | None = None,
                     estimator: str = "ksg",
                     k: int = 4) -> dict:
    """Compute info-theoretic diagnostics for a method's outputs.

    s_hat, s, a: (N, T). z: (N,) optional. Returns a flat dict.
    """
    est = ksg_mi if estimator == "ksg" else gaussian_copula_mi
    f_hat = _feat(s_hat)
    f_s = _feat(s)
    f_a = _feat(a)
    res = {
        "I_shat_s": float(est(f_hat, f_s, k=k) if estimator == "ksg" else est(f_hat, f_s)),
        "I_shat_a": float(est(f_hat, f_a, k=k) if estimator == "ksg" else est(f_hat, f_a)),
    }
    if z is not None:
        z_col = np.asarray(z).reshape(-1, 1)
        res["I_shat_z"] = float(est(f_hat, z_col, k=k) if estimator == "ksg" else est(f_hat, z_col))
    return res
