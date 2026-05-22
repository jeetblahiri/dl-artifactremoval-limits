"""Controlled simulation of Theorem 1 (identifiability failure).

Construct S = S0 + αZ, A = A0 + βZ with S0, A0, Z mutually independent. Then
verify that for every γ, (S', A') = (S − γZ, A + γZ) yields the same X = S + A
distribution, i.e. X is information-theoretically blind to γ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IdentifiabilityRun:
    gammas: np.ndarray
    var_S_alt: np.ndarray        # Var[S − γZ]
    var_A_alt: np.ndarray        # Var[A + γZ]
    var_X: float                 # Var[X], independent of γ
    moment_X_per_gamma: np.ndarray   # 4th central moment of X under each γ
    S0: np.ndarray
    A0: np.ndarray
    Z: np.ndarray
    S: np.ndarray
    A: np.ndarray
    X: np.ndarray
    alpha: float
    beta: float


def simulate(n: int = 8000,
             alpha: float = 0.8,
             beta: float = 0.6,
             gammas: np.ndarray | None = None,
             seed: int = 0) -> IdentifiabilityRun:
    rng = np.random.default_rng(seed)
    if gammas is None:
        gammas = np.linspace(-1.0, 1.5, 9)

    S0 = rng.standard_normal(n)
    A0 = rng.standard_normal(n) * 0.8
    Z = rng.standard_normal(n) * 1.2

    S = S0 + alpha * Z
    A = A0 + beta * Z
    X = S + A

    var_S_alt = np.array([np.var(S - g * Z) for g in gammas])
    var_A_alt = np.array([np.var(A + g * Z) for g in gammas])
    var_X = float(np.var(X))
    # Higher moments of X are invariant under γ — verify by computing X under each γ
    # using the alternative decomposition; the sum is unchanged but compute anyway.
    moment_X_per_gamma = np.array([float(np.mean((((S - g * Z) + (A + g * Z))
                                                   - X.mean()) ** 4))
                                   for g in gammas])
    return IdentifiabilityRun(gammas=gammas,
                              var_S_alt=var_S_alt,
                              var_A_alt=var_A_alt,
                              var_X=var_X,
                              moment_X_per_gamma=moment_X_per_gamma,
                              S0=S0, A0=A0, Z=Z, S=S, A=A, X=X,
                              alpha=alpha, beta=beta)
