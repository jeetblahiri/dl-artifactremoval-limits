"""Interface conformance for every handcrafted method in the registry."""
import numpy as np
import pytest

from artifact_limits.methods.registry import HANDCRAFTED_IDS, build_method


@pytest.mark.parametrize("method_id", HANDCRAFTED_IDS)
def test_method_runs_end_to_end(method_id, gaussian_pools):
    s_pool, a_pool, _, _ = gaussian_pools
    method = build_method(method_id)
    method.fit(s_pool, a_pool)
    x = s_pool[:32] + a_pool[:32]
    s_hat = method.transform(x)
    assert s_hat.shape == x.shape
    assert np.isfinite(s_hat).all(), f"{method_id} produced non-finite output"


def test_identity_returns_input_unchanged(gaussian_pools):
    s_pool, a_pool, _, _ = gaussian_pools
    method = build_method("identity").fit(s_pool, a_pool)
    x = s_pool[:8] + a_pool[:8]
    np.testing.assert_allclose(method.transform(x), x)


def test_wiener_filter_beats_identity(gaussian_pools):
    s_pool, a_pool, _, _ = gaussian_pools
    wf = build_method("wiener_filter").fit(s_pool, a_pool)
    id_ = build_method("identity")
    x = s_pool[:128] + a_pool[:128]
    err_id = float(np.mean((id_.transform(x) - s_pool[:128]) ** 2))
    err_wf = float(np.mean((wf.transform(x) - s_pool[:128]) ** 2))
    assert err_wf < err_id
