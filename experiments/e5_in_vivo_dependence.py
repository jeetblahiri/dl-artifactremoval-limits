"""E5 — In vivo dependence audit on Sleep-EDF.

For each subject and each artifact pair (EEG, EOG) / (EEG, EMG):
  1. Bandpass 0.5–45 Hz.
  2. Linearly regress out the trivial coupling.
  3. KSG MI on 10-s window features (variance, kurtosis, band-power).
  4. Bootstrap CI; permutation null.

Per-subject MI is reported; aggregate fixed-effects MI is computed by pooling
the window-level feature matrices across subjects and re-running the estimator
on the pool.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LinearRegression

from artifact_limits.data.load_D3_audit import load_D3_audit
from artifact_limits.evaluation.block_inference import (
    block_bootstrap_ci,
    circular_shift_null,
)
from artifact_limits.theory.dependence_perturbation import _segment_feature
from artifact_limits.theory.mi_estimators import gaussian_copula_mi, permutation_null


def _bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 45.0) -> np.ndarray:
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.99)
    sos = butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def _regress_out(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return y − a x − b (linear residual)."""
    reg = LinearRegression().fit(x.reshape(-1, 1), y)
    return y - reg.predict(x.reshape(-1, 1))


def _windowed_features(x: np.ndarray, fs: float, win_s: float) -> np.ndarray:
    w = int(win_s * fs)
    n = x.size // w
    segs = x[: n * w].reshape(n, w)
    return _segment_feature(segs)


def _analyse_subject(subject: str, duration_s: float, win_s: float,
                     n_perm: int, seed: int,
                     block_len_windows: int = 3) -> dict:
    """Per-subject MI with block-bootstrap CIs and circular-shift null.

    ``block_len_windows`` is in *windows*; with 10-s windows and block_len=3
    that's a 30-second block, comparable to autocorrelation timescales in
    bandpassed EEG.
    """
    rec = load_D3_audit(subject=subject, duration_s=duration_s)
    fs = rec.fs
    eeg = _bandpass(rec.eeg, fs)
    eog = _bandpass(rec.eog, fs)
    emg = _bandpass(rec.emg, fs)

    eeg_res_eog = _regress_out(eog, eeg)
    eeg_res_emg = _regress_out(emg, eeg)

    feats = {
        "EOG": (_windowed_features(eeg_res_eog, fs, win_s),
                _windowed_features(eog, fs, win_s)),
        "EMG": (_windowed_features(eeg_res_emg, fs, win_s),
                _windowed_features(emg, fs, win_s)),
    }

    I_dep: dict[str, dict] = {}
    null_samples: dict[str, list[float]] = {}
    for label, (fx, fy) in feats.items():
        mean_mi, lo, hi, _ = block_bootstrap_ci(
            fx, fy, estimator=gaussian_copula_mi,
            n_boot=200, block_len=block_len_windows, seed=seed)
        I_dep[label] = {"mean": float(mean_mi), "lo": float(lo), "hi": float(hi)}
        nulls_circ = circular_shift_null(
            fx, fy, estimator=gaussian_copula_mi,
            n_perm=int(n_perm), seed=seed)
        null_samples[label] = nulls_circ.tolist()

    return {
        "subject": subject,
        "is_synthetic": bool(rec.is_synthetic),
        "I_dep": I_dep,
        "null_samples": null_samples,
        "features": {label: (fx, fy) for label, (fx, fy) in feats.items()},
    }


def run(seed: int, cfg: dict, out_dir: Path) -> dict:
    subjects = list(cfg.get("subjects") or [cfg.get("subject", "SC4001")])
    duration_s = float(cfg.get("duration_s", 600.0))
    win_s = float(cfg["window_s"])
    n_perm = int(cfg.get("n_perm", 200))

    per_subject: list[dict] = []
    pooled_feats: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"EOG": [], "EMG": []}
    for subj in subjects:
        out = _analyse_subject(subj, duration_s, win_s, n_perm, seed)
        for label in ("EOG", "EMG"):
            pooled_feats[label].append(out["features"][label])
        # Drop the per-feature arrays before persisting (serialisation cost).
        feats_for_report = out.pop("features")
        out["n_windows"] = {label: int(feats_for_report[label][0].shape[0])
                            for label in feats_for_report}
        per_subject.append(out)

    # Across-subject fixed-effects MI: pool feature matrices then estimate once.
    # Use block bootstrap to respect within-subject autocorrelation; the
    # pooled matrix's "block" still groups consecutive windows from the same
    # subject (subjects are concatenated in order).
    aggregate: dict[str, dict] = {}
    aggregate_null: dict[str, list[float]] = {}
    for label, pairs in pooled_feats.items():
        if not pairs:
            continue
        fx = np.concatenate([p[0] for p in pairs], axis=0)
        fy = np.concatenate([p[1] for p in pairs], axis=0)
        mean_mi, lo, hi, _ = block_bootstrap_ci(
            fx, fy, estimator=gaussian_copula_mi,
            n_boot=200, block_len=3, seed=seed)
        aggregate[label] = {"mean": float(mean_mi), "lo": float(lo), "hi": float(hi),
                            "n_windows": int(fx.shape[0])}
        aggregate_null[label] = circular_shift_null(
            fx, fy, estimator=gaussian_copula_mi,
            n_perm=int(n_perm), seed=seed).tolist()

    return {
        "experiment": "E5",
        "seed": seed,
        "subjects": subjects,
        "per_subject": per_subject,
        "I_dep": aggregate,
        "null_samples": aggregate_null,
        # Convenience: per-subject mean MI as a flat array per label.
        "per_subject_summary": {
            label: [s["I_dep"][label]["mean"] for s in per_subject]
            for label in ("EOG", "EMG")
        },
    }
