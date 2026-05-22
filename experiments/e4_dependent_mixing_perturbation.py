"""E4 — Train on D1 (α=0); evaluate on D1'(α) across an α-grid.

Reports per-α RRMSE-T against the modified target s̃ and the latent-leakage
I(ŝ; z). Deep methods are trained once on α=0 and then frozen across α.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from artifact_limits.data.build_D1prime import construct_D1prime
from artifact_limits.data.load_D1 import load_D1
from artifact_limits.evaluation.info_diagnostics import info_diagnostics
from artifact_limits.evaluation.reconstruction_metrics import rrmse_t
from artifact_limits.methods import build_method
from experiments.shared.data_loaders import make_test_mixtures
from experiments.shared.train_loops import repeat_seeds


def _fit_once(method_id: str, s_pool, a_pool, deep_set, seed, deep_kwargs):
    if method_id in deep_set:
        method = build_method(method_id, seed=seed, **deep_kwargs)
    else:
        method = build_method(method_id)
    method.fit(s_pool, a_pool)
    return method


def run(seed: int, cfg: dict, out_dir: Path) -> dict:
    next(repeat_seeds([seed]))
    d1 = load_D1()
    rng = np.random.default_rng(seed)
    kind = cfg.get("artifact_kind", "EOG")
    (mix,) = make_test_mixtures(d1, rng, kinds=(kind,))

    # Pre-build per-α dependent-mixing test sets from the same test base.
    alphas = list(cfg["alpha_grid"])
    perturbed: dict[float, dict] = {}
    for a in alphas:
        perturbed[a] = construct_D1prime(mix.s, mix.a, alpha=a, seed=seed + int(a * 1000))

    deep_set = set(cfg.get("deep", []))
    methods = list(cfg.get("handcrafted", [])) + list(cfg.get("deep", []))
    deep_kwargs = cfg.get("deep_kwargs", {})

    results = []
    for method_id in methods:
        m = _fit_once(method_id, d1.s_train, d1.a_train[kind],
                      deep_set, seed, deep_kwargs)
        rrmse_per_alpha = []
        leak_per_alpha = []
        for a in alphas:
            pert = perturbed[a]
            s_hat = m.transform(pert["x"])
            rrmse_per_alpha.append(float(rrmse_t(s_hat, pert["s_tilde"]).mean()))
            try:
                diag = info_diagnostics(s_hat, pert["s_tilde"], pert["a_tilde"],
                                        z=pert["z"], estimator="gauss_copula")
                leak_per_alpha.append(diag.get("I_shat_z", 0.0))
            except Exception:
                leak_per_alpha.append(0.0)
        results.append({
            "method_id": method_id,
            "rrmse_t_per_alpha": rrmse_per_alpha,
            "leak_z_per_alpha": leak_per_alpha,
        })

    return {
        "experiment": "E4",
        "seed": seed,
        "alphas": alphas,
        "kind": kind,
        "methods": results,
    }
