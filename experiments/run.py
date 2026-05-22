"""Unified experiment runner.

Usage: ``python -m experiments.run E2 --seed 42``

Each experiment module exposes a single ``run(seed, cfg, out_dir) -> dict``
entrypoint; the JSON it returns lands in ``results/<exp>/seed_<seed>/results.json``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import click
import yaml

from artifact_limits.data.paths import results_dir

EXPERIMENTS = {
    "E0": ("experiments.e0_pool_audit", "e0.yaml"),
    "E1": ("experiments.e1_reproduction_and_wiener_floor", "e1.yaml"),
    "E2": ("experiments.e2_methods_comparison", "e2.yaml"),
    "E3": ("experiments.e3_capacity_scaling", "e3.yaml"),
    "E4": ("experiments.e4_dependent_mixing_perturbation", "e4.yaml"),
    "E4t": ("experiments.e4_template_robustness", "e4.yaml"),
    "E5": ("experiments.e5_in_vivo_dependence", "e5.yaml"),
    "E6": ("experiments.e6_oracle_mmse", "e6.yaml"),
}


def _load_cfg(default_yaml: str, override: str | None) -> dict:
    cfg_dir = Path(__file__).parent / "config"
    p = Path(override) if override else cfg_dir / default_yaml
    return yaml.safe_load(p.read_text())


def _dispatch(mod_name: str):
    import importlib
    return importlib.import_module(mod_name)


@click.command()
@click.argument("exp_id", type=click.Choice(list(EXPERIMENTS.keys())))
@click.option("--seed", default=42, type=int)
@click.option("--config", default=None, type=str, help="Path to a YAML override.")
def run(exp_id: str, seed: int, config: str | None) -> None:
    module_name, default_yaml = EXPERIMENTS[exp_id]
    cfg = _load_cfg(default_yaml, config)
    out = results_dir() / exp_id / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)

    module = _dispatch(module_name)
    t0 = time.perf_counter()
    payload = module.run(seed=seed, cfg=cfg, out_dir=out)
    payload["wall_seconds"] = time.perf_counter() - t0
    payload["config"] = cfg

    out_file = out / "results.json"
    out_file.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[{exp_id}] wrote {out_file} ({payload['wall_seconds']:.1f} s)")


if __name__ == "__main__":
    run()
