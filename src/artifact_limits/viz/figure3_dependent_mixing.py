"""Figure 3 — dependent-mixing collapse (E4 result)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from artifact_limits.viz._io import load_consolidated, save_figure
from artifact_limits.viz.style import apply_style, method_color


def main() -> None:
    apply_style()
    data = load_consolidated("E4")
    alphas = np.array(data["alphas"])
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.5, 3.0))

    for entry in data["methods"]:
        name = entry["method_id"]
        color = method_color(name)
        axA.plot(alphas, entry["rrmse_t_per_alpha"], "-o", color=color, label=name)
        axB.plot(alphas, entry["leak_z_per_alpha"], "-o", color=color, label=name)

    axA.set_xlabel(r"dependence strength $\alpha$")
    axA.set_ylabel("RRMSE-T")
    axA.set_title("A. RRMSE rises with $\\alpha$")

    axB.set_xlabel(r"$\alpha$")
    axB.set_ylabel(r"$I(\hat s; z)$")
    axB.set_title("B. Latent leakage")
    axA.legend(fontsize=5.5, ncol=2)

    plt.tight_layout()
    pdf = save_figure(fig, "figure3_dependent_mixing")
    print(f"[figure3] written: {pdf}")


if __name__ == "__main__":
    main()
