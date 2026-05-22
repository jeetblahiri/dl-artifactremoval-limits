"""Reconstruction metrics — the field's standard set plus floor-normalised variants.

All metric functions accept arrays of shape (N, T) for (estimate, target) and
return either a per-segment vector (length N) or a single scalar depending on
the `reduce` flag.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _to_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    return x


def rrmse_t(s_hat: np.ndarray, s: np.ndarray, reduce: str = "none") -> np.ndarray:
    """Temporal RRMSE = ||ŝ - s||₂ / ||s||₂ per segment."""
    s_hat = _to_2d(s_hat)
    s = _to_2d(s)
    err = np.sqrt(np.mean((s_hat - s) ** 2, axis=-1))
    nrm = np.sqrt(np.mean(s ** 2, axis=-1))
    val = err / (nrm + 1e-12)
    if reduce == "mean":
        return float(val.mean())
    return val


def rrmse_f(s_hat: np.ndarray, s: np.ndarray, reduce: str = "none") -> np.ndarray:
    """Spectral RRMSE on magnitude spectrum."""
    S_hat = np.abs(np.fft.rfft(_to_2d(s_hat), axis=-1))
    S = np.abs(np.fft.rfft(_to_2d(s), axis=-1))
    err = np.sqrt(np.mean((S_hat - S) ** 2, axis=-1))
    nrm = np.sqrt(np.mean(S ** 2, axis=-1))
    val = err / (nrm + 1e-12)
    if reduce == "mean":
        return float(val.mean())
    return val


def cc(s_hat: np.ndarray, s: np.ndarray, reduce: str = "none") -> np.ndarray:
    """Pearson correlation per segment."""
    s_hat = _to_2d(s_hat)
    s = _to_2d(s)
    sh = s_hat - s_hat.mean(axis=-1, keepdims=True)
    ss = s - s.mean(axis=-1, keepdims=True)
    num = (sh * ss).sum(axis=-1)
    den = np.sqrt((sh ** 2).sum(axis=-1) * (ss ** 2).sum(axis=-1)) + 1e-12
    val = num / den
    if reduce == "mean":
        return float(val.mean())
    return val


def snr_improvement(x: np.ndarray, s_hat: np.ndarray, s: np.ndarray,
                    reduce: str = "none") -> np.ndarray:
    """SNR improvement in dB between input mixture x and estimate ŝ."""
    x = _to_2d(x)
    s_hat = _to_2d(s_hat)
    s = _to_2d(s)
    snr_in = 10 * np.log10(np.mean(s ** 2, axis=-1) / (np.mean((x - s) ** 2, axis=-1) + 1e-12))
    snr_out = 10 * np.log10(np.mean(s ** 2, axis=-1) / (np.mean((s_hat - s) ** 2, axis=-1) + 1e-12))
    val = snr_out - snr_in
    if reduce == "mean":
        return float(val.mean())
    return val


def summarise_per_snr(s_hat: np.ndarray, s: np.ndarray, x: np.ndarray,
                      snr_db: np.ndarray) -> dict:
    """Per-segment metrics grouped by SNR level → dict of dicts."""
    snr_db = np.asarray(snr_db)
    out: dict[str, dict[float, float]] = {"rrmse_t": {}, "rrmse_f": {}, "cc": {}, "snr_imp": {}}
    for level in np.unique(snr_db):
        m = snr_db == level
        if not m.any():
            continue
        out["rrmse_t"][float(level)] = float(rrmse_t(s_hat[m], s[m]).mean())
        out["rrmse_f"][float(level)] = float(rrmse_f(s_hat[m], s[m]).mean())
        out["cc"][float(level)] = float(cc(s_hat[m], s[m]).mean())
        out["snr_imp"][float(level)] = float(snr_improvement(x[m], s_hat[m], s[m]).mean())
    return out


def excess_over_floor(metric_per_snr: dict[float, float],
                      floor_per_snr: dict[float, float]) -> dict[float, float]:
    """Floor-normalised excess: (metric − floor) / floor per SNR cell."""
    return {snr: (val - floor_per_snr[snr]) / (floor_per_snr[snr] + 1e-12)
            for snr, val in metric_per_snr.items() if snr in floor_per_snr}
