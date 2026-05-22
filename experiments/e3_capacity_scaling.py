"""E3 — Capacity scaling on SimpleCNN."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from artifact_limits.data.load_D1 import load_D1
from artifact_limits.evaluation.stats import bootstrap_ci
from artifact_limits.methods import build_method
from artifact_limits.theory.wiener_floor import estimate_wiener_floor
from experiments.shared.data_loaders import make_test_mixtures
from experiments.shared.train_loops import repeat_seeds, score_method


def run(seed: int, cfg: dict, out_dir: Path) -> dict:
    next(repeat_seeds([seed]))
    d1 = load_D1()
    rng = np.random.default_rng(seed)
    kind = cfg.get("artifact_kinds", ["EOG"])[0]
    (mix,) = make_test_mixtures(d1, rng, kinds=(kind,))
    floor = estimate_wiener_floor(d1.s_train, d1.a_train[kind])
    deep_kwargs = cfg.get("deep_kwargs", {})
    family = cfg.get("family", "simple_cnn")

    param_counts = []
    means, los, his = [], [], []
    for widths in cfg["width_sweep"]:
        method = build_method(family, widths=widths, seed=seed, **deep_kwargs)
        method.fit(d1.s_train, d1.a_train[kind])
        s_hat = method.transform(mix.x)
        scores = score_method(s_hat, mix.s, mix.x, mix.snr_db)
        rrmse_t_pool = np.array(list(scores["rrmse_t"].values()))
        mean, lo, hi = bootstrap_ci(rrmse_t_pool, seed=seed)
        param_counts.append(int(method.n_parameters()))
        means.append(mean)
        los.append(lo)
        his.append(hi)

    return {
        "experiment": "E3",
        "seed": seed,
        "families": [{
            "family": family,
            "param_counts": param_counts,
            "rrmse_t_mean": means,
            "rrmse_t_lo": los,
            "rrmse_t_hi": his,
        }],
        "wiener_floor_rrmse_t": float(floor.rrmse_floor),
    }
