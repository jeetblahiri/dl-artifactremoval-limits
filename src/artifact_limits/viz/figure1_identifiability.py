"""Figure 1 — identifiability failure under shared latents.

Three panels:
  A. Schematic time series of (S, A) vs (S', A') = (S − γZ, A + γZ), both summing to X.
  B. Marginals of S, S' (different) and X, X' (identical).
  C. Var[S − γZ] vs γ — no observation-level statistic distinguishes the family.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from artifact_limits.theory.identifiability_sim import simulate
from artifact_limits.viz._io import save_figure
from artifact_limits.viz.style import ACCENT_COLOR, DEEP_COLOR, HANDCRAFTED_COLOR, apply_style


def main() -> None:
    apply_style()
    run = simulate(n=8000, alpha=0.8, beta=0.6, seed=0)
    gammas = run.gammas

    fig, axs = plt.subplots(1, 3, figsize=(8.5, 2.5))

    # Panel A — schematic traces (first 200 samples of the simulation).
    t = np.arange(200)
    g_demo = 0.5
    S_prime = run.S - g_demo * run.Z
    A_prime = run.A + g_demo * run.Z
    axs[0].plot(t, run.S[:200], color=HANDCRAFTED_COLOR, label="$S$")
    axs[0].plot(t, run.A[:200], color=DEEP_COLOR, label="$A$", alpha=0.7)
    axs[0].plot(t, S_prime[:200], color=HANDCRAFTED_COLOR, ls=":", label="$S' = S-\\gamma Z$")
    axs[0].plot(t, A_prime[:200], color=DEEP_COLOR, ls=":", label="$A' = A+\\gamma Z$", alpha=0.7)
    axs[0].plot(t, run.X[:200], color="black", lw=0.8, label="$X = S+A$")
    axs[0].set_xlabel("sample index")
    axs[0].set_title("A. Two decompositions, same X")
    axs[0].legend(loc="upper right", ncol=1, fontsize=5.5)

    # Panel B — marginal histograms.
    bins = np.linspace(-6, 6, 60)
    axs[1].hist(run.S, bins=bins, alpha=0.5, color=HANDCRAFTED_COLOR, label="$S$")
    axs[1].hist(run.S - g_demo * run.Z, bins=bins, alpha=0.5, color=ACCENT_COLOR,
                label="$S - \\gamma Z$")
    axs[1].set_title("B. $S$ and $S'$ marginals differ")
    axs[1].set_xlabel("amplitude")
    axs[1].legend(fontsize=6)

    # Panel C — Var[S − γZ] vs γ.
    axs[2].plot(gammas, run.var_S_alt, "-o", color=HANDCRAFTED_COLOR, label="Var$[S - \\gamma Z]$")
    axs[2].plot(gammas, run.var_A_alt, "-o", color=DEEP_COLOR, label="Var$[A + \\gamma Z]$")
    axs[2].axhline(run.var_X, ls="--", color="black", label="Var$[X]$ (invariant)")
    axs[2].set_xlabel("$\\gamma$")
    axs[2].set_title("C. $X$ invariant; $S', A'$ vary")
    axs[2].legend(fontsize=6)

    plt.tight_layout()
    pdf = save_figure(fig, "figure1_identifiability")
    print(f"[figure1] written: {pdf}; var(X) = {run.var_X:.4f} (γ-invariant by construction).")


if __name__ == "__main__":
    main()
