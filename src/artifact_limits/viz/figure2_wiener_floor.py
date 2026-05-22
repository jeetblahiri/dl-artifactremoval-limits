"""Figure 2 — methods clustered around the Wiener floor.

Three panels:
  A. Bar chart of methods on EEGdenoiseNet (RRMSE-T with bootstrap CI).
  B. Floor-normalised excess (RRMSE - D_W) / D_W.
  C. Non-Gaussian gain budget: where each method sits between the absolute
     pool-marginal floor (D_W) and the Wiener filter's empirical MSE — the
     fraction of the sub-Wiener region a method captures.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from artifact_limits.viz._io import load_consolidated, save_figure
from artifact_limits.viz.style import apply_style, method_color


def _non_gaussian_gain(rrmse: np.ndarray, floor: float, wiener_emp: float) -> np.ndarray:
    """Position of each method between D_W (1.0) and the Wiener filter (0.0).

    Methods at or above the Wiener filter map to ≤0 (no non-Gaussian gain).
    Methods at the floor map to 1.0 (entire budget exploited). The budget is
    (wiener_emp − D_W) / wiener_emp in RRMSE units; we normalise by it.
    """
    budget = max(wiener_emp - floor, 1e-12)
    return (wiener_emp - rrmse) / budget


def main() -> None:
    apply_style()
    data = load_consolidated("E2")
    methods = data["methods"]
    rrmse = np.array([m["rrmse_t_mean"] for m in methods])
    rrmse_lo = np.array([m["rrmse_t_lo"] for m in methods])
    rrmse_hi = np.array([m["rrmse_t_hi"] for m in methods])
    names = [m["method_id"] for m in methods]
    colors = [method_color(n) for n in names]
    floor = float(data.get("wiener_floor_rrmse_t", np.nan))

    # Empirical Wiener filter RRMSE (the method itself sits in the methods list).
    wiener_idx = names.index("wiener_filter") if "wiener_filter" in names else None
    wiener_emp = float(rrmse[wiener_idx]) if wiener_idx is not None else floor

    order = np.argsort(rrmse)[::-1]
    rrmse, rrmse_lo, rrmse_hi = rrmse[order], rrmse_lo[order], rrmse_hi[order]
    names_o = [names[i] for i in order]
    colors_o = [colors[i] for i in order]

    fig, axs = plt.subplots(1, 3, figsize=(11.5, 0.32 * len(methods) + 1.5),
                             gridspec_kw={"width_ratios": [1.6, 1.0, 1.2]})
    axA, axB, axC = axs
    y = np.arange(len(methods))

    axA.barh(y, rrmse, color=colors_o,
             xerr=[rrmse - rrmse_lo, rrmse_hi - rrmse], capsize=2)
    axA.axvline(floor, ls="--", color="black", label=f"Wiener floor $D_W$ = {floor:.3f}")
    axA.axvline(wiener_emp, ls=":", color="grey",
                label=f"Wiener filter (emp.) = {wiener_emp:.3f}")
    axA.set_yticks(y)
    axA.set_yticklabels(names_o)
    axA.set_xlabel("RRMSE-T (mean across SNR × kind, 5 seeds)")
    axA.set_title("A. Methods on EEGdenoiseNet")
    axA.legend(loc="lower right", fontsize=6)

    excess = (rrmse - floor) / (floor + 1e-12)
    axB.barh(y, excess, color=colors_o)
    axB.axvline(0, color="black", lw=0.6)
    axB.set_yticks(y)
    axB.set_yticklabels([])
    axB.set_xlabel("(RRMSE − $D_W$) / $D_W$")
    axB.set_title("B. Excess above the floor")

    ng_gain = _non_gaussian_gain(rrmse, floor, wiener_emp)
    axC.barh(y, np.clip(ng_gain, -0.05, None), color=colors_o)
    axC.axvline(0, color="grey", lw=0.6, label="Wiener filter")
    axC.axvline(1, ls="--", color="black", label="Floor $D_W$")
    axC.set_yticks(y)
    axC.set_yticklabels([])
    axC.set_xlabel("non-Gaussian gain budget exploited")
    axC.set_title("C. Sub-Wiener (non-Gaussian) gain")
    axC.legend(loc="lower right", fontsize=6)

    plt.tight_layout()
    pdf = save_figure(fig, "figure2_wiener_floor")
    print(f"[figure2] written: {pdf};  floor={floor:.4f}  wiener_emp={wiener_emp:.4f}")


if __name__ == "__main__":
    main()
