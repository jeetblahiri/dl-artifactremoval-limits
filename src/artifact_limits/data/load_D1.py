"""Canonical loader for EEGdenoiseNet (D1).

Provides the (s_pool, a_pool, x, s_target) interface required by the methods
layer, plus a mixture-generator with the field's standard SNR-stratified
training distribution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from artifact_limits import SNR_DB_LEVELS
from artifact_limits.data.download_eegdenoisenet import ensure_dataset
from artifact_limits.data.paths import splits_dir

ArtifactKind = Literal["EOG", "EMG"]


@dataclass(frozen=True)
class D1Split:
    s_train: np.ndarray
    s_val: np.ndarray
    s_test: np.ndarray
    a_train: dict[str, np.ndarray]   # by artifact kind
    a_val: dict[str, np.ndarray]
    a_test: dict[str, np.ndarray]


def _slice(arr: np.ndarray, idx: list[int]) -> np.ndarray:
    return np.ascontiguousarray(arr[np.asarray(idx, dtype=np.int64)])


def _per_segment_normalize(arr: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std per segment — the EEGdenoiseNet preprocessing convention."""
    arr = np.asarray(arr, dtype=np.float32)
    mu = arr.mean(axis=-1, keepdims=True)
    sigma = arr.std(axis=-1, keepdims=True)
    return (arr - mu) / (sigma + 1e-9)


def load_D1(artifact_kinds: Iterable[str] = ("EOG", "EMG"),
            normalize: bool = True) -> D1Split:
    paths = ensure_dataset()
    eeg = np.load(paths["EEG"]).astype(np.float32)
    eog = np.load(paths["EOG"]).astype(np.float32) if "EOG" in artifact_kinds else None
    emg = np.load(paths["EMG"]).astype(np.float32) if "EMG" in artifact_kinds else None
    if normalize:
        eeg = _per_segment_normalize(eeg)
        if eog is not None:
            eog = _per_segment_normalize(eog)
        if emg is not None:
            emg = _per_segment_normalize(emg)

    split = json.loads((splits_dir() / "eegdenoisenet_split.json").read_text())

    a_train: dict[str, np.ndarray] = {}
    a_val: dict[str, np.ndarray] = {}
    a_test: dict[str, np.ndarray] = {}
    if eog is not None:
        a_train["EOG"] = _slice(eog, split["EOG"]["train"])
        a_val["EOG"] = _slice(eog, split["EOG"]["val"])
        a_test["EOG"] = _slice(eog, split["EOG"]["test"])
    if emg is not None:
        a_train["EMG"] = _slice(emg, split["EMG"]["train"])
        a_val["EMG"] = _slice(emg, split["EMG"]["val"])
        a_test["EMG"] = _slice(emg, split["EMG"]["test"])

    return D1Split(
        s_train=_slice(eeg, split["EEG"]["train"]),
        s_val=_slice(eeg, split["EEG"]["val"]),
        s_test=_slice(eeg, split["EEG"]["test"]),
        a_train=a_train,
        a_val=a_val,
        a_test=a_test,
    )


def _snr_scale(s: np.ndarray, a: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale a so that 10 log10(||s||² / ||λa||²) = snr_db, per-segment.

    Returns λ a.
    """
    s_pow = np.mean(s ** 2, axis=-1, keepdims=True)
    a_pow = np.mean(a ** 2, axis=-1, keepdims=True)
    lam = np.sqrt(s_pow / (a_pow + 1e-12) / (10 ** (snr_db / 10.0)))
    return lam * a


def make_mixture(s_pool: np.ndarray,
                 a_pool: np.ndarray,
                 snr_db_levels: Iterable[float] = SNR_DB_LEVELS,
                 rng: np.random.Generator | None = None,
                 pairing: Literal["random", "sequential"] = "random") -> dict:
    """Construct paired (x, s, λa) at given SNR levels (uniform sampling)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = s_pool.shape[0]
    if a_pool.shape[0] < n:
        # Resample with replacement to length n.
        idx_a = rng.integers(0, a_pool.shape[0], size=n)
    else:
        if pairing == "random":
            idx_a = rng.permutation(a_pool.shape[0])[:n]
        else:
            idx_a = np.arange(n)
    a = a_pool[idx_a].astype(np.float32)
    snr_db_levels = list(snr_db_levels)
    snrs = rng.choice(snr_db_levels, size=n).astype(np.float32)
    a_scaled = _snr_scale(s_pool, a, snrs[:, None])
    x = s_pool + a_scaled
    return {"x": x.astype(np.float32),
            "s": s_pool.astype(np.float32),
            "a": a_scaled.astype(np.float32),
            "snr_db": snrs}


def make_stratified_mixture(s_pool: np.ndarray, a_pool: np.ndarray,
                            snr_db_levels: Iterable[float] = SNR_DB_LEVELS,
                            rng: np.random.Generator | None = None) -> dict:
    """One mixture per (segment, SNR) → flat batch with explicit SNR label per row."""
    if rng is None:
        rng = np.random.default_rng(0)
    snr_db_levels = list(snr_db_levels)
    parts = []
    for snr in snr_db_levels:
        n = s_pool.shape[0]
        idx_a = (rng.permutation(a_pool.shape[0])[:n]
                 if a_pool.shape[0] >= n
                 else rng.integers(0, a_pool.shape[0], size=n))
        a = a_pool[idx_a].astype(np.float32)
        a_scaled = _snr_scale(s_pool, a, float(snr))
        parts.append({"x": s_pool + a_scaled, "s": s_pool, "a": a_scaled,
                      "snr_db": np.full(n, snr, dtype=np.float32)})
    out = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    return out
