"""E4-aux: template-robustness sweep for the D1' construction.

For each template pair (e_S, e_A) in ``template_library``, we re-build the
dependent-mixing variant at the full α grid and evaluate every method that
was trained in E4. The reported summary per method is the slope of the
RRMSE-T-vs-α curve (linear regression). The intended takeaway is that the
sign and rough magnitude of the slope survives the template choice, even
though absolute degradation rates differ.

This script is inference-only: it uses methods already trained on D1 (α=0),
so it adds essentially no wall-clock cost relative to E4 itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from artifact_limits.data.build_D1prime import construct_D1prime
from artifact_limits.data.d1prime_templates import template_library
from artifact_limits.data.load_D1 import load_D1
from artifact_limits.evaluation.reconstruction_metrics import rrmse_t
from artifact_limits.methods import build_method
from experiments.shared.data_loaders import make_test_mixtures
from experiments.shared.train_loops import repeat_seeds


def run(seed: int, cfg: dict, out_dir: Path) -> dict:
    next(repeat_seeds([seed]))
    d1 = load_D1()
    rng = np.random.default_rng(seed)
    kind = cfg.get("artifact_kind", "EOG")
    (mix,) = make_test_mixtures(d1, rng, kinds=(kind,))

    alphas = list(cfg["alpha_grid"])
    templates = template_library(seed=seed)
    methods = list(cfg.get("handcrafted", [])) + list(cfg.get("deep", []))
    deep_set = set(cfg.get("deep", []))
    deep_kwargs = cfg.get("deep_kwargs", {})

    # Train every method once on D1 (α=0).
    fitted = {}
    for mid in methods:
        if mid in deep_set:
            method = build_method(mid, seed=seed, **deep_kwargs)
        else:
            method = build_method(mid)
        method.fit(d1.s_train, d1.a_train[kind])
        fitted[mid] = method

    per_template = []
    for name, e_S, e_A in templates:
        per_alpha_per_method: dict[str, list[float]] = {}
        for a in alphas:
            pert = construct_D1prime(mix.s, mix.a, alpha=a,
                                     seed=seed + int(a * 1000),
                                     e_S=e_S, e_A=e_A)
            for mid, method in fitted.items():
                s_hat = method.transform(pert["x"])
                per_alpha_per_method.setdefault(mid, []).append(
                    float(rrmse_t(s_hat, pert["s_tilde"]).mean()))
        slopes = {}
        for mid, rrmse_curve in per_alpha_per_method.items():
            xs = np.asarray(alphas, dtype=float)
            ys = np.asarray(rrmse_curve, dtype=float)
            slope = float(np.polyfit(xs, ys, 1)[0])
            slopes[mid] = {"rrmse_t_per_alpha": rrmse_curve,
                           "slope": slope}
        per_template.append({"template": name, "methods": slopes})

    # Aggregate: mean / std of slope across templates per method.
    slope_by_method: dict[str, list[float]] = {}
    for t in per_template:
        for mid, m in t["methods"].items():
            slope_by_method.setdefault(mid, []).append(m["slope"])
    aggregate = {
        mid: {"mean_slope": float(np.mean(s)),
              "std_slope": float(np.std(s)),
              "n_templates": int(len(s))}
        for mid, s in slope_by_method.items()
    }

    return {
        "experiment": "E4t",
        "seed": seed,
        "alphas": alphas,
        "kind": kind,
        "n_templates": len(templates),
        "per_template": per_template,
        "aggregate_slope": aggregate,
    }
