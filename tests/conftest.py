"""Shared test fixtures."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def gaussian_pools():
    """Two zero-mean Gaussian pools with known covariances on a small T.

    Returns (s_pool, a_pool, Sigma_S, Sigma_A).
    """
    rng = np.random.default_rng(0)
    T = 32
    Sigma_S = np.eye(T) * 1.0
    # AR(1)-like artifact covariance
    rho = 0.6
    idx = np.arange(T)
    Sigma_A = rho ** np.abs(idx[:, None] - idx[None, :])
    n_s = 4000
    n_a = 4000
    s_pool = rng.multivariate_normal(np.zeros(T), Sigma_S, size=n_s).astype(np.float32)
    a_pool = rng.multivariate_normal(np.zeros(T), Sigma_A, size=n_a).astype(np.float32)
    return s_pool, a_pool, Sigma_S, Sigma_A
