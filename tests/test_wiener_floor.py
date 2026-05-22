"""Wiener floor: empirical RRMSE matches the closed form on Gaussian data."""
import numpy as np

from artifact_limits.theory.wiener_floor import (
    apply_wiener, estimate_wiener_floor, wiener_filter_predict,
)


def test_wiener_floor_closed_form_matches_empirical(gaussian_pools):
    s_pool, a_pool, Sigma_S, Sigma_A = gaussian_pools
    floor = estimate_wiener_floor(s_pool, a_pool, noise_var=0.0)

    rng = np.random.default_rng(1)
    n_test = 5000
    s = rng.multivariate_normal(np.zeros(Sigma_S.shape[0]), Sigma_S, size=n_test).astype(np.float32)
    a = rng.multivariate_normal(np.zeros(Sigma_A.shape[0]), Sigma_A, size=n_test).astype(np.float32)
    x = s + a
    s_hat = apply_wiener(x, floor.W)
    empirical = float(np.mean((s_hat - s) ** 2) * Sigma_S.shape[0])
    # Both should match D_W to within finite-sample noise (~few %).
    rel_err = abs(empirical - floor.D_W) / floor.D_W
    assert rel_err < 0.10, f"D_W mismatch: {empirical:.4f} vs {floor.D_W:.4f}"


def test_wiener_filter_predict_alias(gaussian_pools):
    s_pool, a_pool, *_ = gaussian_pools
    x = s_pool[:64] + a_pool[:64]
    floor = estimate_wiener_floor(s_pool, a_pool)
    direct = apply_wiener(x, floor.W)
    one_shot = wiener_filter_predict(x, s_pool, a_pool)
    np.testing.assert_allclose(direct, one_shot, rtol=1e-5)


def test_artifact_scale_monotone(gaussian_pools):
    """Increasing the artifact scale should monotonically raise D_W."""
    s_pool, a_pool, *_ = gaussian_pools
    floors = [estimate_wiener_floor(s_pool, a_pool, artifact_scale=s).D_W
              for s in [0.1, 0.5, 1.0, 2.0]]
    assert all(b > a for a, b in zip(floors, floors[1:]))


def test_per_snr_floor_decreases_with_snr(gaussian_pools):
    """Higher SNR (less artifact) → lower per-SNR D_W."""
    from artifact_limits.theory.wiener_floor import estimate_floor_per_snr
    s_pool, a_pool, *_ = gaussian_pools
    floors = estimate_floor_per_snr(s_pool, a_pool, [-7, -4, -1, 2])
    rrmse = [floors[snr].rrmse_floor for snr in [-7, -4, -1, 2]]
    assert all(b < a for a, b in zip(rrmse, rrmse[1:]))
