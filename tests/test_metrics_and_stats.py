"""Sanity tests for evaluation metrics and statistics."""
import numpy as np

from artifact_limits.evaluation.reconstruction_metrics import cc, rrmse_t, snr_improvement
from artifact_limits.evaluation.stats import cliffs_delta, fdr_bh, wilcoxon_pair


def test_rrmse_zero_on_perfect_estimate():
    s = np.random.default_rng(0).standard_normal((16, 32))
    np.testing.assert_allclose(rrmse_t(s, s), 0.0, atol=1e-7)


def test_cc_one_on_identity():
    s = np.random.default_rng(0).standard_normal((16, 32))
    np.testing.assert_allclose(cc(s, s), 1.0, atol=1e-6)


def test_snr_improvement_positive_when_estimate_closer():
    rng = np.random.default_rng(0)
    s = rng.standard_normal((8, 32))
    a = rng.standard_normal((8, 32))
    x = s + a
    s_hat = s + 0.1 * a       # closer to s than x
    assert (snr_improvement(x, s_hat, s).mean()) > 0


def test_cliffs_delta_extremes():
    a = np.full(20, 1.0)
    b = np.full(20, 0.0)
    assert cliffs_delta(a, b) == 1.0
    assert cliffs_delta(b, a) == -1.0


def test_fdr_rejects_only_small_pvalues():
    p = np.array([0.001, 0.02, 0.5, 0.8])
    rej = fdr_bh(p, q=0.05)
    assert rej[0] and not rej[3]


def test_wilcoxon_pair_returns_pvalue():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a + 0.5
    out = wilcoxon_pair(a, b)
    assert 0.0 <= out["pvalue"] <= 1.0
