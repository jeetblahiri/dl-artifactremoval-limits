"""Paper-quality matplotlib defaults."""
from __future__ import annotations

import matplotlib as mpl

HANDCRAFTED_COLOR = "#2E7D32"
DEEP_COLOR = "#1565C0"
WIENER_COLOR = "#B00020"
ACCENT_COLOR = "#F57C00"


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 8.0,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.6,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def method_color(method_id: str) -> str:
    from artifact_limits.methods.registry import DEEP_IDS, HANDCRAFTED_IDS
    if method_id == "wiener_filter":
        return WIENER_COLOR
    if method_id in HANDCRAFTED_IDS:
        return HANDCRAFTED_COLOR
    if method_id in DEEP_IDS:
        return DEEP_COLOR
    return ACCENT_COLOR
