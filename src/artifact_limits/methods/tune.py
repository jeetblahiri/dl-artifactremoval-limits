"""Optuna-based hyperparameter tuning for handcrafted denoisers.

Each registered method has its own *search space*, identified by `method_id`.
The optimiser fits on `s_train`/`a_train`, scores on `(x_val, s_val)` mixtures
formed at the standard EEGdenoiseNet SNR grid, and returns the params that
minimise mean per-segment RRMSE-T.

We use 20 trials per method (PREREGISTRATION §5). Search uses the official
validation split only; the test split is never touched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from artifact_limits import SNR_DB_LEVELS
from artifact_limits.data.load_D1 import make_stratified_mixture
from artifact_limits.evaluation.reconstruction_metrics import rrmse_t
from artifact_limits.methods import build_method


@dataclass
class TuneResult:
    method_id: str
    best_params: dict
    best_score: float                  # mean RRMSE-T at the best params
    history: list[tuple[dict, float]]  # full trial log
    n_trials: int


def _objective_factory(method_id: str, params: dict,
                       s_train: np.ndarray, a_train: np.ndarray,
                       x_val: np.ndarray, s_val: np.ndarray) -> float:
    method = build_method(method_id, **params)
    method.fit(s_train, a_train)
    s_hat = method.transform(x_val)
    return float(rrmse_t(s_hat, s_val).mean())


def _suggest_params(trial, method_id: str) -> dict:
    if method_id == "wiener_filter":
        return {
            "noise_var": trial.suggest_float("noise_var", 0.0, 0.5),
        }
    if method_id == "bandpass_notch":
        return {
            "low":  trial.suggest_float("low",  0.1, 2.0),
            "high": trial.suggest_float("high", 30.0, 50.0),
        }
    if method_id == "dwt":
        return {
            "wavelet": trial.suggest_categorical(
                "wavelet", ["db4", "db6", "db8", "sym4", "sym6", "coif2"]),
            "level":           trial.suggest_int("level", 3, 6),
            "mode":            trial.suggest_categorical("mode", ["soft", "hard"]),
            "threshold_scale": trial.suggest_float("threshold_scale", 0.1, 2.0),
        }
    if method_id == "swt":
        return {
            "wavelet": trial.suggest_categorical(
                "wavelet", ["db4", "db6", "db8", "sym4", "sym6"]),
            "level":           trial.suggest_int("level", 2, 5),
            "mode":            trial.suggest_categorical("mode", ["soft", "hard"]),
            "threshold_scale": trial.suggest_float("threshold_scale", 0.1, 2.0),
        }
    if method_id == "emd":
        return {
            "n_imfs_to_threshold": trial.suggest_int("n_imfs_to_threshold", 1, 6),
            "max_imfs":             trial.suggest_int("max_imfs", 4, 10),
            "threshold_scale":      trial.suggest_float("threshold_scale", 0.1, 2.0),
        }
    if method_id == "total_variation":
        return {
            "weight":  trial.suggest_float("weight", 1e-3, 5.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 50, 400),
        }
    if method_id == "savgol":
        window = trial.suggest_int("window", 5, 51, step=2)        # forced odd
        order_hi = min(window - 1, 6)
        return {
            "window": int(window),
            "order":  trial.suggest_int("order", 1, order_hi),
        }
    if method_id == "identity":
        return {}
    raise ValueError(f"No Optuna search space defined for method_id={method_id!r}")


def _scale_threshold_for_dwt(s_pool, x_val, s_val):
    """Hook for a future custom threshold scale — currently unused."""
    return None


def tune_handcrafted(method_id: str,
                     s_train: np.ndarray,
                     a_train: np.ndarray,
                     s_val: np.ndarray,
                     a_val: np.ndarray,
                     n_trials: int = 20,
                     seed: int = 42,
                     snr_db_levels=SNR_DB_LEVELS) -> TuneResult:
    """Optuna search over a method's hyperparameter space (val-only)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rng = np.random.default_rng(seed)
    val_mix = make_stratified_mixture(s_val, a_val,
                                      snr_db_levels=snr_db_levels, rng=rng)
    x_val_mix = val_mix["x"]
    s_val_target = val_mix["s"]

    history: list[tuple[dict, float]] = []

    def objective(trial) -> float:
        params = _suggest_params(trial, method_id)
        try:
            score = _objective_factory(method_id, params, s_train, a_train,
                                       x_val_mix, s_val_target)
        except Exception as e:
            # Mark bad trials with a large but finite score so Optuna can prune.
            history.append((params, float("nan")))
            raise optuna.exceptions.TrialPruned() from e
        history.append((params, score))
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    if method_id == "identity":
        # Trivially no params — record the identity score once and return.
        score = _objective_factory("identity", {}, s_train, a_train,
                                   x_val_mix, s_val_target)
        return TuneResult("identity", {}, score, [({}, score)], 1)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   catch=(Exception,))
    return TuneResult(
        method_id=method_id,
        best_params=dict(study.best_params),
        best_score=float(study.best_value),
        history=history,
        n_trials=int(n_trials),
    )


def tune_all_handcrafted(method_ids,
                         s_train: np.ndarray,
                         a_train: np.ndarray,
                         s_val: np.ndarray,
                         a_val: np.ndarray,
                         n_trials: int = 20,
                         seed: int = 42) -> dict[str, TuneResult]:
    out: dict[str, TuneResult] = {}
    for mid in method_ids:
        out[mid] = tune_handcrafted(mid, s_train, a_train, s_val, a_val,
                                    n_trials=n_trials, seed=seed)
    return out
