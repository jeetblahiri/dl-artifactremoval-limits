"""Wiener floor and Wiener filter for the EEGdenoiseNet additive-independent model.

Under X = S + A + N with S ⊥ A ⊥ N, the linear MMSE estimator is

    W = Σ_S (Σ_S + Σ_A + σ²I)^{-1}            (so ŝ = W x)

with expected squared error

    D_W = tr(Σ_S (Σ_A + σ²I) (Σ_S + Σ_A + σ²I)^{-1}).

This module exposes both as functions and provides convenience routines to
report the floor in RRMSE-T units (the field's metric).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _empirical_cov(x: np.ndarray, bias: bool = True) -> np.ndarray:
    """Per-feature zero-mean covariance over rows.

    np.cov treats columns as observations by default; we transpose so rows = obs.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.cov(x.T, bias=bias)


@dataclass
class WienerFloor:
    D_W: float
    D_W_per_freq: np.ndarray
    Sigma_S: np.ndarray
    Sigma_A: np.ndarray
    rrmse_floor: float           # √(D_W / tr(Σ_S))
    W: np.ndarray                # Wiener-filter matrix, shape (T, T); ŝ = x @ W.T
    noise_var: float
    n_segments_S: int
    n_segments_A: int


def _silence_blas_matmul_warnings():
    """Apple Accelerate occasionally emits spurious divide/overflow warnings on
    small matmuls. Suppress them locally; the numerical results are unaffected.
    """
    import warnings
    warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
    warnings.filterwarnings("ignore", message="overflow encountered in matmul")
    warnings.filterwarnings("ignore", message="invalid value encountered in matmul")


_silence_blas_matmul_warnings()


def estimate_wiener_floor(s_pool: np.ndarray,
                          a_pool: np.ndarray,
                          noise_var: float = 0.0,
                          ridge: float = 1e-8,
                          artifact_scale: float = 1.0) -> WienerFloor:
    """Estimate D_W and the Wiener filter matrix from the training pools.

    Args:
        s_pool: (N_s, T) clean EEG segments.
        a_pool: (N_a, T) artifact segments (same kind, e.g. all EOG).
        noise_var: additive sensor noise variance σ². 0.0 if not modelled.
        ridge: ridge added to Σ_X for numerical stability.
        artifact_scale: scalar multiplier on the artifact (after segment-level
            normalization this is λ from the SNR protocol, so for unit-variance
            pools λ² = 10^(-SNR_dB/10)).

    Returns:
        WienerFloor dataclass.
    """
    s = np.asarray(s_pool, dtype=np.float64)
    a = np.asarray(a_pool, dtype=np.float64)
    T = s.shape[1]
    if a.shape[1] != T:
        raise ValueError("s_pool and a_pool must have matching T.")

    Sigma_S = _empirical_cov(s, bias=True)
    Sigma_A = (artifact_scale ** 2) * _empirical_cov(a, bias=True)
    Sigma_N = noise_var * np.eye(T)
    Sigma_X = Sigma_S + Sigma_A + Sigma_N + ridge * np.eye(T)

    # W = Σ_S Σ_X^{-1}; ŝ = W x. For (T,T) symmetric Σ_X we solve via Cholesky.
    L = np.linalg.cholesky(Sigma_X)
    # Σ_S Σ_X^{-1} = (Σ_X^{-1} Σ_S^T)^T → solve(L, Σ_S^T) twice
    rhs = np.linalg.solve(L, Sigma_S.T)
    inv_Sx_Ss = np.linalg.solve(L.T, rhs)
    W = inv_Sx_Ss.T

    # D_W = tr(Σ_S (Σ_A + σ²I) Σ_X^{-1}).
    A_plus_N = Sigma_A + Sigma_N
    rhs2 = np.linalg.solve(L, A_plus_N.T)
    M = np.linalg.solve(L.T, rhs2)                # = Σ_X^{-1} (Σ_A + σ²I)
    D_W = float(np.trace(Sigma_S @ M))

    # Per-frequency contribution (rough — diagonal of W under DFT basis).
    # Compute on a circulant approximation: average per-tap diagonal in DFT.
    F = np.fft.fft(np.eye(T)) / np.sqrt(T)
    W_freq_diag = np.real(np.diag(F.conj() @ W @ F))
    sS_freq_diag = np.real(np.diag(F.conj() @ Sigma_S @ F))
    aN_freq_diag = np.real(np.diag(F.conj() @ A_plus_N @ F))
    sX_freq_diag = sS_freq_diag + aN_freq_diag
    sX_freq_diag = np.where(sX_freq_diag <= 0, 1e-12, sX_freq_diag)
    D_W_per_freq = sS_freq_diag * aN_freq_diag / sX_freq_diag
    D_W_per_freq = np.real(D_W_per_freq[: T // 2 + 1])

    rrmse_floor = float(np.sqrt(D_W / max(np.trace(Sigma_S), 1e-12)))

    return WienerFloor(
        D_W=D_W,
        D_W_per_freq=D_W_per_freq,
        Sigma_S=Sigma_S,
        Sigma_A=Sigma_A,
        rrmse_floor=rrmse_floor,
        W=W,
        noise_var=noise_var,
        n_segments_S=s.shape[0],
        n_segments_A=a.shape[0],
    )


def wiener_filter_predict(x: np.ndarray,
                          s_pool: np.ndarray,
                          a_pool: np.ndarray,
                          noise_var: float = 0.0) -> np.ndarray:
    """One-shot: estimate W from the pools and apply to a batch of mixtures."""
    floor = estimate_wiener_floor(s_pool, a_pool, noise_var=noise_var)
    return apply_wiener(x, floor.W)


def apply_wiener(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """ŝ = x @ W^T."""
    x64 = np.asarray(x, dtype=np.float64)
    return (x64 @ W.T).astype(x.dtype if hasattr(x, "dtype") else np.float32)


def closed_form_rrmse_floor(floor: WienerFloor) -> float:
    """RRMSE-T-equivalent of D_W given Σ_S."""
    return floor.rrmse_floor


def estimate_floor_per_snr(s_pool: np.ndarray,
                            a_pool: np.ndarray,
                            snr_db_levels,
                            noise_var: float = 0.0) -> dict[float, WienerFloor]:
    """Closed-form Wiener floor for each SNR level under the EEGdenoiseNet
    mixing protocol on per-segment-normalised pools.

    For unit-variance s and a, the per-segment SNR scale is λ² = 10^(-SNR/10);
    Σ_A_eff = λ² Σ_A. We return one ``WienerFloor`` per SNR.
    """
    out: dict[float, WienerFloor] = {}
    for snr in snr_db_levels:
        lam = float(10 ** (-float(snr) / 20.0))   # λ such that λ² = 10^(-SNR/10)
        out[float(snr)] = estimate_wiener_floor(s_pool, a_pool,
                                                noise_var=noise_var,
                                                artifact_scale=lam)
    return out
