"""Theorem 1 simulator: X is γ-invariant; Var[S − γZ] varies with γ."""
import numpy as np

from artifact_limits.theory.identifiability_sim import simulate


def test_X_is_invariant_under_gamma():
    run = simulate(n=4000, alpha=0.9, beta=0.4)
    # Var[(S − γZ) + (A + γZ)] = Var[S + A] regardless of γ
    for g in run.gammas:
        Xp = (run.S - g * run.Z) + (run.A + g * run.Z)
        np.testing.assert_allclose(np.mean(Xp), np.mean(run.X), atol=1e-6)
        np.testing.assert_allclose(np.var(Xp), np.var(run.X), rtol=1e-6)


def test_S_marginal_changes_with_gamma():
    run = simulate(n=4000)
    distinct = np.unique(np.round(run.var_S_alt, 5))
    assert distinct.size >= 3, "Var[S − γZ] should vary across γ"
