"""E0 — Pool audit: does P_S ⊥ P_A hold inside EEGdenoiseNet?

We compute three diagnostic quantities:
  * 50–80 Hz power in the clean EEG pool (high-frequency residual muscle indicator);
  * <1 Hz power in the clean EEG pool (residual eye / drift indicator);
  * cross-pool MI between random (s_i, a_j) pairings against a permutation null.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from artifact_limits import SAMPLING_FREQ
from artifact_limits.data.load_D1 import load_D1
from artifact_limits.theory.dependence_perturbation import _segment_feature
from artifact_limits.theory.mi_estimators import gaussian_copula_mi
from experiments.shared.train_loops import repeat_seeds


def _band_power(x: np.ndarray, lo: float, hi: float, fs: int = SAMPLING_FREQ) -> np.ndarray:
    freqs = np.fft.rfftfreq(x.shape[-1], 1.0 / fs)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    mask = (freqs >= lo) & (freqs <= hi)
    return spec[:, mask].sum(axis=-1)


def run(seed: int, cfg: dict, out_dir: Path) -> dict:
    next(repeat_seeds([seed]))
    d1 = load_D1()

    s = d1.s_train
    a_eog = d1.a_train["EOG"]
    a_emg = d1.a_train["EMG"]

    hf = _band_power(s, *cfg["high_freq_band"])
    lf = _band_power(s, *cfg["low_freq_band"])

    rng = np.random.default_rng(seed)
    n_samples = min(2000, s.shape[0])
    sel_s = rng.choice(s.shape[0], n_samples, replace=False)
    sel_eog = rng.choice(a_eog.shape[0], n_samples, replace=True)
    sel_emg = rng.choice(a_emg.shape[0], n_samples, replace=True)
    fs = _segment_feature(s[sel_s])
    fa_eog = _segment_feature(a_eog[sel_eog])
    fa_emg = _segment_feature(a_emg[sel_emg])

    mi_obs = [
        gaussian_copula_mi(fs, fa_eog),
        gaussian_copula_mi(fs, fa_emg),
    ]
    mi_null = []
    for _ in range(cfg["n_perm"]):
        idx = rng.permutation(n_samples)
        mi_null.append(gaussian_copula_mi(fs, fa_eog[idx]))
        mi_null.append(gaussian_copula_mi(fs, fa_emg[idx]))

    result = {
        "experiment": "E0",
        "seed": seed,
        "clean_pool": {
            "high_freq_power": hf.tolist(),
            "low_freq_power": lf.tolist(),
            "high_freq_band": cfg["high_freq_band"],
            "low_freq_band": cfg["low_freq_band"],
        },
        "mi_observed": mi_obs,
        "mi_null": mi_null,
        "summary": {
            "median_hf_power": float(np.median(hf)),
            "frac_above_3sigma_hf": float(np.mean(hf > (hf.mean() + 3 * hf.std()))),
            "median_lf_power": float(np.median(lf)),
        },
    }
    return result
