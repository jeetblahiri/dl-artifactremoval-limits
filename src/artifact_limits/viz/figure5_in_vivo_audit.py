"""Figure 5 — In vivo dependence audit on Sleep-EDF.

Panel A: pooled fixed-effects I(EEG; EOG) and I(EEG; EMG) with 95 % bootstrap
         CI and permutation-null violin.
Panel B: per-subject scatter, showing across-subject heterogeneity.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from artifact_limits.viz._io import load_consolidated, save_figure
from artifact_limits.viz.style import ACCENT_COLOR, DEEP_COLOR, HANDCRAFTED_COLOR, apply_style


def _violin_overlay(ax, x_pos, samples, color, width=0.4):
    parts = ax.violinplot([samples], positions=[x_pos], widths=width, showmedians=False,
                          showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.25)


def main() -> None:
    apply_style()
    data = load_consolidated("E5")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    labels = list(data["I_dep"].keys())
    means = [data["I_dep"][lbl]["mean"] for lbl in labels]
    los   = [data["I_dep"][lbl]["lo"] for lbl in labels]
    his   = [data["I_dep"][lbl]["hi"] for lbl in labels]
    nulls = data.get("null_samples", {})

    colors_by_label = {"EOG": HANDCRAFTED_COLOR, "EMG": DEEP_COLOR}
    x = np.arange(len(labels))
    for i, lbl in enumerate(labels):
        c = colors_by_label.get(lbl, ACCENT_COLOR)
        if lbl in nulls and nulls[lbl]:
            _violin_overlay(axA, i, np.asarray(nulls[lbl]), c)
    axA.bar(x, means,
            yerr=[np.asarray(means) - np.asarray(los),
                  np.asarray(his) - np.asarray(means)],
            capsize=4,
            color=[colors_by_label.get(lbl, ACCENT_COLOR) for lbl in labels],
            edgecolor="black", linewidth=0.8, alpha=0.85)
    axA.set_xticks(x)
    axA.set_xticklabels(labels)
    axA.set_ylabel("Mutual information (nats)")
    axA.set_title("A. Pooled (fixed-effects) residual MI")
    axA.axhline(0, color="black", lw=0.4)

    # Panel B — per subject jittered scatter
    summary = data.get("per_subject_summary", {})
    if summary:
        x_pos = np.arange(len(labels))
        rng = np.random.default_rng(0)
        for i, lbl in enumerate(labels):
            vals = np.asarray(summary.get(lbl, []), dtype=float)
            jitter = rng.normal(0, 0.07, size=vals.size)
            axB.scatter(x_pos[i] + jitter, vals, s=22,
                        color=colors_by_label.get(lbl, ACCENT_COLOR),
                        edgecolor="black", linewidth=0.4, zorder=3)
            if vals.size:
                axB.hlines(np.median(vals), x_pos[i] - 0.2, x_pos[i] + 0.2,
                           color="black", lw=1.0)
        axB.set_xticks(x_pos)
        axB.set_xticklabels(labels)
        axB.set_ylabel("Per-subject MI (nats)")
        axB.set_title("B. Across-subject heterogeneity")
        axB.axhline(0, color="black", lw=0.4)
    else:
        axB.text(0.5, 0.5, "Per-subject MI unavailable",
                 ha="center", va="center", transform=axB.transAxes)
        axB.axis("off")

    plt.tight_layout()
    pdf = save_figure(fig, "figure5_in_vivo_audit")
    print(f"[figure5] written: {pdf}")


if __name__ == "__main__":
    main()
