"""E1 — Reproduce SOTA DL on EEGdenoiseNet and locate the Wiener floor.

Outputs per (method, artifact kind) the per-SNR RRMSE-T/F, CC, SNR-improvement,
and the floor-normalised excess. The floor is computed on the *training* pool.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from artifact_limits import SNR_DB_LEVELS
from artifact_limits.data.load_D1 import load_D1
from artifact_limits.evaluation.reconstruction_metrics import excess_over_floor
from artifact_limits.methods import build_method
from artifact_limits.theory.wiener_floor import estimate_floor_per_snr, estimate_wiener_floor
from experiments.shared.data_loaders import make_test_mixtures
from experiments.shared.train_loops import per_method_payload, repeat_seeds, score_method


def _floor_rrmse_per_snr(s_pool, a_pool, snr_levels) -> dict[float, float]:
    floors = estimate_floor_per_snr(s_pool, a_pool, snr_levels)
    return {snr: floor.rrmse_floor for snr, floor in floors.items()}


def run(seed: int, cfg: dict, out_dir: Path) -> dict:
    next(repeat_seeds([seed]))
    d1 = load_D1()
    rng = np.random.default_rng(seed)
    kinds = tuple(cfg.get("artifact_kinds", ("EOG", "EMG")))
    mixtures = make_test_mixtures(d1, rng, kinds=kinds)
    snr_levels = tuple(SNR_DB_LEVELS)

    # Both: (a) a pool-marginal floor (the back-of-envelope number) and
    #       (b) per-SNR floors (the right number against which excess is computed).
    floor_per_kind = {}
    floor_rrmse = {}
    floor_rrmse_per_snr_kind: dict[str, dict[float, float]] = {}
    for kind in kinds:
        floor_per_kind[kind] = estimate_wiener_floor(d1.s_train, d1.a_train[kind])
        floor_rrmse[kind] = floor_per_kind[kind].rrmse_floor
        floor_rrmse_per_snr_kind[kind] = _floor_rrmse_per_snr(
            d1.s_train, d1.a_train[kind], snr_levels)

    methods = ["wiener_filter"] + list(cfg.get("methods", []))
    deep_kwargs = cfg.get("deep_kwargs", {})

    method_payloads = []
    for method_id in methods:
        per_kind: dict[str, dict] = {}
        for mix in mixtures:
            if method_id == "wiener_filter":
                method = build_method(method_id)
            elif method_id in {"identity", "bandpass_notch", "dwt", "swt", "emd",
                               "total_variation", "savgol"}:
                method = build_method(method_id)
            else:
                method = build_method(method_id, seed=seed, **deep_kwargs)
            method.fit(d1.s_train, d1.a_train[mix.kind])
            s_hat = method.transform(mix.x)
            scores = score_method(s_hat, mix.s, mix.x, mix.snr_db)
            scores["rrmse_t_excess"] = excess_over_floor(
                scores["rrmse_t"], floor_rrmse_per_snr_kind[mix.kind])
            per_kind[mix.kind] = scores
        method_payloads.append(per_method_payload(method_id, per_kind))

    return {
        "experiment": "E1",
        "seed": seed,
        "methods": method_payloads,
        "wiener_floor_rrmse_t_per_kind": floor_rrmse,
        "wiener_floor_rrmse_t": float(np.mean(list(floor_rrmse.values()))),
        "wiener_floor_per_snr_per_kind": floor_rrmse_per_snr_kind,
    }
