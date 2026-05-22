"""Convenience wrappers for fitting + scoring a registered method."""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np

from artifact_limits.evaluation.reconstruction_metrics import summarise_per_snr
from artifact_limits.methods import build_method


def fit_and_predict(method_id: str,
                    s_pool: np.ndarray,
                    a_pool: np.ndarray,
                    x_test: np.ndarray,
                    method_kwargs: dict | None = None) -> tuple[np.ndarray, float, "object"]:
    method_kwargs = dict(method_kwargs or {})
    method = build_method(method_id, **method_kwargs)
    t0 = time.perf_counter()
    method.fit(s_pool, a_pool)
    s_hat = method.transform(x_test)
    elapsed = time.perf_counter() - t0
    return s_hat, elapsed, method


def score_method(s_hat: np.ndarray, s: np.ndarray, x: np.ndarray,
                 snr_db: np.ndarray) -> dict:
    return summarise_per_snr(s_hat, s, x, snr_db)


def per_method_payload(method_id: str,
                       per_kind_results: dict[str, dict]) -> dict:
    """Flatten (kind → per-snr metrics) into a serialisable result block."""
    payload = {"method_id": method_id, "per_kind": {}}
    rrmse_t_all = []
    for kind, scores in per_kind_results.items():
        payload["per_kind"][kind] = scores
        rrmse_t_all.extend(scores["rrmse_t"].values())
    payload["rrmse_t_mean"] = float(np.mean(rrmse_t_all)) if rrmse_t_all else float("nan")
    payload["rrmse_t_std"] = float(np.std(rrmse_t_all)) if rrmse_t_all else float("nan")
    return payload


def repeat_seeds(seeds: Iterable[int]):
    """Iterator with a quick reseed helper for numpy + torch."""
    import numpy as np
    try:
        import torch
        has_torch = True
    except Exception:
        has_torch = False
    for s in seeds:
        np.random.seed(s)
        if has_torch:
            torch.manual_seed(s)
        yield s
