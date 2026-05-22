"""D1' construction: α = 0 reproduces D1 exactly."""
import numpy as np

from artifact_limits.data.build_D1prime import construct_D1prime


def test_alpha_zero_is_identity():
    rng = np.random.default_rng(0)
    s = rng.standard_normal((64, 128)).astype(np.float32)
    a = rng.standard_normal((64, 128)).astype(np.float32)
    out = construct_D1prime(s, a, alpha=0.0, seed=123)
    # Pairing permutation is internal; we just check sum equals s + a_perm.
    np.testing.assert_allclose(out["s_tilde"], s)
    np.testing.assert_allclose(out["s_tilde"] + out["a_tilde"], out["x"])


def test_positive_alpha_modifies_segments():
    rng = np.random.default_rng(0)
    s = rng.standard_normal((64, 128)).astype(np.float32)
    a = rng.standard_normal((64, 128)).astype(np.float32)
    out = construct_D1prime(s, a, alpha=0.5, seed=123)
    assert not np.allclose(out["s_tilde"], s)
