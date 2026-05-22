"""Statistical tests used by the experiments.

- Paired Wilcoxon signed-rank.
- Cliff's δ effect size.
- Benjamini–Hochberg FDR correction.
- Bootstrap percentile confidence intervals.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_pair(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 5 or np.allclose(a, b):
        return {"stat": 0.0, "pvalue": 1.0, "n": int(a.size)}
    stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return {"stat": float(stat), "pvalue": float(p), "n": int(a.size)}


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's δ ∈ [−1, 1]; positive means a > b stochastically."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return 0.0
    # Memory-efficient via sorted-b and searchsorted (still O(n log n)).
    b_sorted = np.sort(b)
    n_b = b_sorted.size
    greater = np.searchsorted(b_sorted, a, side="left")
    less = n_b - np.searchsorted(b_sorted, a, side="right")
    delta = (greater.sum() - less.sum()) / (a.size * n_b)
    return float(delta)


def fdr_bh(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg FDR: returns boolean mask of rejected nulls."""
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    if not below.any():
        return np.zeros(n, dtype=bool)
    cutoff = np.max(np.where(below)[0])
    reject = np.zeros(n, dtype=bool)
    reject[order[: cutoff + 1]] = True
    return reject


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean and (alpha/2, 1-alpha/2) bootstrap percentile CI."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(v, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)
