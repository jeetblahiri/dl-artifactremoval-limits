"""E2 — Methods comparison inside EEGdenoiseNet."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from artifact_limits import SNR_DB_LEVELS
from artifact_limits.data.load_D1 import load_D1
from artifact_limits.evaluation.reconstruction_metrics import excess_over_floor, rrmse_t
from artifact_limits.evaluation.stats import bootstrap_ci
from artifact_limits.methods import build_method
from artifact_limits.methods.tune import tune_handcrafted
from artifact_limits.theory.wiener_floor import estimate_floor_per_snr, estimate_wiener_floor
from experiments.shared.data_loaders import make_test_mixtures
from experiments.shared.train_loops import per_method_payload, repeat_seeds, score_method


def _tune_handcrafted_per_kind(method_id: str, d1, kinds, n_trials: int, seed: int) -> dict:
    """Run a fresh Optuna sweep per artifact kind so each kind gets its own
    tuned parameters (the validation distribution differs)."""
    best_per_kind: dict[str, dict] = {}
    for k in kinds:
        res = tune_handcrafted(method_id,
                               d1.s_train, d1.a_train[k],
                               d1.s_val,   d1.a_val[k],
                               n_trials=n_trials, seed=seed)
        best_per_kind[k] = {"params": res.best_params, "val_score": res.best_score,
                            "n_trials": res.n_trials}
    return best_per_kind


def run(seed: int, cfg: dict, out_dir: Path) -> dict:
    next(repeat_seeds([seed]))
    d1 = load_D1()
    rng = np.random.default_rng(seed)
    kinds = tuple(cfg.get("artifact_kinds", ("EOG", "EMG")))
    mixtures = make_test_mixtures(d1, rng, kinds=kinds)
    snr_levels = tuple(SNR_DB_LEVELS)

    floor_per_kind = {k: estimate_wiener_floor(d1.s_train, d1.a_train[k]) for k in kinds}
    floor_per_snr_kind = {
        k: {snr: fl.rrmse_floor
            for snr, fl in estimate_floor_per_snr(d1.s_train, d1.a_train[k], snr_levels).items()}
        for k in kinds
    }

    deep_kwargs = cfg.get("deep_kwargs", {})
    handcrafted = list(cfg.get("handcrafted", []))
    deep = list(cfg.get("deep", []))
    optuna_trials = int(cfg.get("optuna_trials", 20))

    # Tune handcrafted methods per artifact kind on val data only.
    tuned_handcrafted: dict[str, dict] = {}
    for method_id in handcrafted:
        tuned_handcrafted[method_id] = _tune_handcrafted_per_kind(
            method_id, d1, kinds, optuna_trials, seed)

    all_methods = handcrafted + deep
    payloads = []
    for method_id in all_methods:
        per_kind: dict[str, dict] = {}
        seg_rrmse_pool: list[float] = []
        for mix in mixtures:
            if method_id in deep:
                method = build_method(method_id, seed=seed, **deep_kwargs)
            elif method_id in tuned_handcrafted:
                method = build_method(method_id,
                                      **tuned_handcrafted[method_id][mix.kind]["params"])
            else:
                method = build_method(method_id)
            method.fit(d1.s_train, d1.a_train[mix.kind])
            s_hat = method.transform(mix.x)
            scores = score_method(s_hat, mix.s, mix.x, mix.snr_db)
            scores["rrmse_t_excess"] = excess_over_floor(
                scores["rrmse_t"], floor_per_snr_kind[mix.kind])
            per_kind[mix.kind] = scores
            # Bootstrap CI is drawn from the per-segment RRMSE pool, not the
            # 4-SNR cell-means, so the CI reflects segment-level variance.
            seg_rrmse_pool.extend(rrmse_t(s_hat, mix.s).tolist())
        payload = per_method_payload(method_id, per_kind)
        mean, lo, hi = bootstrap_ci(np.array(seg_rrmse_pool), n_boot=1000, seed=seed)
        payload.update({"rrmse_t_mean": mean, "rrmse_t_lo": lo, "rrmse_t_hi": hi})
        payloads.append(payload)

    return {
        "experiment": "E2",
        "seed": seed,
        "methods": payloads,
        "wiener_floor_rrmse_t_per_kind": {k: float(v.rrmse_floor) for k, v in floor_per_kind.items()},
        "wiener_floor_per_snr_per_kind": floor_per_snr_kind,
        "wiener_floor_rrmse_t": float(np.mean([v.rrmse_floor for v in floor_per_kind.values()])),
        "optuna_tuned_handcrafted": tuned_handcrafted,
        "optuna_n_trials": optuna_trials,
    }
