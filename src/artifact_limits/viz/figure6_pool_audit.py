"""Figure 6 — EEGdenoiseNet pool audit (E0)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from artifact_limits.viz._io import load_consolidated, save_figure
from artifact_limits.viz.style import ACCENT_COLOR, DEEP_COLOR, HANDCRAFTED_COLOR, apply_style


def main() -> None:
    apply_style()
    data = load_consolidated("E0")
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.0))

    hf = np.array(data["clean_pool"]["high_freq_power"])
    axs[0].hist(hf, bins=40, color=DEEP_COLOR)
    axs[0].set_title("A. 50–80 Hz power in clean pool")
    axs[0].set_xlabel("power (a.u.)")

    lf = np.array(data["clean_pool"]["low_freq_power"])
    axs[1].hist(lf, bins=40, color=HANDCRAFTED_COLOR)
    axs[1].set_title("B. <1 Hz power in clean pool")
    axs[1].set_xlabel("power (a.u.)")

    obs = np.array(data["mi_observed"])
    null = np.array(data["mi_null"])
    axs[2].scatter(np.full(obs.size, 0), obs, color=ACCENT_COLOR, label="observed")
    axs[2].scatter(np.full(null.size, 1) + np.random.normal(0, 0.05, null.size),
                   null, color="gray", alpha=0.4, label="permutation null")
    axs[2].set_xticks([0, 1])
    axs[2].set_xticklabels(["observed", "null"])
    axs[2].set_ylabel("MI (nats)")
    axs[2].set_title("C. Cross-pool MI vs null")
    axs[2].legend()

    plt.tight_layout()
    pdf = save_figure(fig, "figure6_pool_audit")
    print(f"[figure6] written: {pdf}")


if __name__ == "__main__":
    main()
