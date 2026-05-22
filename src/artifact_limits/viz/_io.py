"""Shared helpers for figure scripts: locate cached results, write outputs."""
from __future__ import annotations

import json
from pathlib import Path

from artifact_limits.data.paths import figures_dir, results_dir


def load_consolidated(exp_id: str) -> dict:
    p = results_dir() / exp_id / "consolidated.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Run the experiment first (`make reproduce-{exp_id.lower()}`)"
            " and then `python results/consolidate.py`.")
    return json.loads(p.read_text())


def save_figure(fig, name: str) -> Path:
    out_dir = figures_dir()
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    return pdf
