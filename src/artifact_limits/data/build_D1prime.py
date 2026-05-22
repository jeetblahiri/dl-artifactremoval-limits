"""Construct the dependent-mixing variant D1' described in §4.3.

For α ∈ [0,1], pool of clean segments s_i and artifact segments a_j, and fixed
unit-norm spectral templates e_S, e_A ∈ R^T:

    z_k ~ N(0,1)
    s̃_k = s_i + α z_k ||s_i|| e_S
    ã_k = a_j + α z_k ||a_j|| e_A
    x_k = s̃_k + ã_k

`α = 0` reproduces D1 (numerical equality).
"""
from __future__ import annotations

import argparse
import json
from typing import Iterable

import h5py
import numpy as np

from artifact_limits import SAMPLING_FREQ, SEGMENT_LEN
from artifact_limits.data.load_D1 import load_D1, make_stratified_mixture
from artifact_limits.data.paths import processed_dir


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def slow_wave_template(T: int = SEGMENT_LEN, fs: int = SAMPLING_FREQ, freq_hz: float = 1.0) -> np.ndarray:
    """A unit-norm 1-Hz sinusoid; surrogate for the slow-wave cortical pattern e_S."""
    t = np.arange(T) / fs
    e = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    return _unit(e)


def eye_movement_template(T: int = SEGMENT_LEN, fs: int = SAMPLING_FREQ) -> np.ndarray:
    """Low-frequency saccade-like pattern: a step convolved with a 2-Hz lobe."""
    t = np.arange(T) / fs
    step = (t > (T / fs) * 0.5).astype(np.float32) - 0.5
    lobe = np.exp(-((t - (T / fs) * 0.5) ** 2) / (2 * 0.1 ** 2)).astype(np.float32)
    e = step + 0.5 * lobe
    e = e - e.mean()
    return _unit(e)


def construct_D1prime(s_pool: np.ndarray,
                      a_pool: np.ndarray,
                      alpha: float,
                      seed: int = 42,
                      e_S: np.ndarray | None = None,
                      e_A: np.ndarray | None = None) -> dict:
    """Apply the dependent-mixing perturbation to a paired test set.

    Returns dict with keys x, s_tilde, a_tilde, z, alpha, I_SA_estimate (filled by caller).
    The pool sizes are aligned by permutation: result length = len(s_pool).
    """
    rng = np.random.default_rng(seed)
    T = s_pool.shape[1]
    if e_S is None:
        e_S = slow_wave_template(T)
    if e_A is None:
        e_A = eye_movement_template(T)

    n = s_pool.shape[0]
    if a_pool.shape[0] >= n:
        idx_a = rng.permutation(a_pool.shape[0])[:n]
    else:
        idx_a = rng.integers(0, a_pool.shape[0], size=n)
    a_paired = a_pool[idx_a].astype(np.float32)
    s = s_pool.astype(np.float32)

    z = rng.standard_normal(n).astype(np.float32)
    s_norms = np.linalg.norm(s, axis=-1, keepdims=True)
    a_norms = np.linalg.norm(a_paired, axis=-1, keepdims=True)

    s_tilde = s + alpha * z[:, None] * s_norms * e_S[None, :]
    a_tilde = a_paired + alpha * z[:, None] * a_norms * e_A[None, :]
    x = s_tilde + a_tilde

    return {"x": x.astype(np.float32),
            "s_tilde": s_tilde.astype(np.float32),
            "a_tilde": a_tilde.astype(np.float32),
            "z": z.astype(np.float32),
            "alpha": float(alpha),
            "e_S": e_S, "e_A": e_A}


def build_alpha_grid(alphas: Iterable[float], seed: int = 42,
                     out_filename: str = "D1prime.h5") -> str:
    """Materialise (x, s_tilde, a_tilde, z) per α and per artifact kind to HDF5."""
    d1 = load_D1()
    rng = np.random.default_rng(seed)
    mix_eog = make_stratified_mixture(d1.s_test, d1.a_test["EOG"], rng=rng)
    mix_emg = make_stratified_mixture(d1.s_test, d1.a_test["EMG"], rng=rng)

    out_path = processed_dir() / out_filename
    with h5py.File(out_path, "w") as f:
        for art_name, mix in (("EOG", mix_eog), ("EMG", mix_emg)):
            g = f.create_group(art_name)
            g.create_dataset("snr_db", data=mix["snr_db"])
            for a_str in alphas:
                a = float(a_str)
                d = construct_D1prime(mix["s"], mix["a"], alpha=a, seed=seed + int(a * 1000))
                sub = g.create_group(f"alpha_{a:.4f}")
                sub.create_dataset("x", data=d["x"], compression="gzip", compression_opts=4)
                sub.create_dataset("s_tilde", data=d["s_tilde"], compression="gzip", compression_opts=4)
                sub.create_dataset("a_tilde", data=d["a_tilde"], compression="gzip", compression_opts=4)
                sub.create_dataset("z", data=d["z"])
                sub.attrs["alpha"] = a
        f.attrs["seed"] = seed
        f.attrs["alphas"] = json.dumps([float(a) for a in alphas])
    print(f"[build_D1prime] wrote {out_path}")
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-grid", default="0,0.05,0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    alphas = [float(a) for a in args.alpha_grid.split(",") if a.strip()]
    build_alpha_grid(alphas, seed=args.seed)


if __name__ == "__main__":
    main()
