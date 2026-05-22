"""MI estimators: calibration on Gaussian joints with known I(X;Y) = -½ log(1-ρ²)."""
import numpy as np

from artifact_limits.theory.mi_estimators import gaussian_copula_mi, ksg_mi


def _gauss_pair(n: int, rho: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    x = z[:, 0]
    y = rho * x + np.sqrt(max(1 - rho ** 2, 1e-9)) * z[:, 1]
    return x, y


def test_gaussian_copula_mi_calibrated_on_known_joint():
    rho = 0.7
    x, y = _gauss_pair(8000, rho, seed=0)
    expected = -0.5 * np.log(1 - rho ** 2)
    est = gaussian_copula_mi(x, y)
    assert abs(est - expected) < 0.05


def test_ksg_mi_increases_with_rho():
    e_low = ksg_mi(*_gauss_pair(2000, 0.1, seed=0), k=4)
    e_high = ksg_mi(*_gauss_pair(2000, 0.8, seed=0), k=4)
    assert e_high > e_low


def test_mi_is_zero_for_independent_samples():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    y = rng.standard_normal(2000)
    assert ksg_mi(x, y, k=4) < 0.1
    assert gaussian_copula_mi(x, y) < 0.05
