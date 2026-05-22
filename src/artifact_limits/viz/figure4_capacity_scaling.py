"""Figure 4 — capacity scaling for SimpleCNN (E3)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from artifact_limits.viz._io import load_consolidated, save_figure
from artifact_limits.viz.style import DEEP_COLOR, apply_style


def main() -> None:
    apply_style()
    data = load_consolidated("E3")
    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    for entry in data["families"]:
        sizes = np.array(entry["param_counts"])
        rrmse = np.array(entry["rrmse_t_mean"])
        lo = np.array(entry["rrmse_t_lo"])
        hi = np.array(entry["rrmse_t_hi"])
        ax.plot(sizes, rrmse, "-o", label=entry["family"], color=DEEP_COLOR)
        ax.fill_between(sizes, lo, hi, color=DEEP_COLOR, alpha=0.2)
    floor = float(data["wiener_floor_rrmse_t"])
    ax.axhline(floor, ls="--", color="black", label=f"Wiener floor $D_W$={floor:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("parameter count")
    ax.set_ylabel("RRMSE-T")
    ax.set_title("Capacity scaling saturates at the floor")
    ax.legend()

    plt.tight_layout()
    pdf = save_figure(fig, "figure4_capacity_scaling")
    print(f"[figure4] written: {pdf}")


if __name__ == "__main__":
    main()
