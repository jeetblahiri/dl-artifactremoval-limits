"""Oracle estimate of the non-linear MMSE on EEGdenoiseNet.

We train a deliberately over-parameterised denoiser on the largest synthetic
mixture set we can afford, then report its held-out test loss as an empirical
upper bound on \(D_{\mathrm{MMSE}}\). Anything a measurable estimator can
achieve sits at or above this number; in particular,

    \(D_{\mathrm{MMSE}} \;\le\; \hat D^{\mathrm{up}}_{\mathrm{MMSE}}\;,\)

so the gap between the empirical Wiener filter and \(\hat D^{\mathrm{up}}_{\mathrm{MMSE}}\)
is a *lower* bound on the non-Gaussian gain budget realisable in practice.

The implementation reuses the existing IC-U-Net architecture with a wider
channel ladder and an extended training schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from artifact_limits import SNR_DB_LEVELS
from artifact_limits.data.load_D1 import D1Split, make_stratified_mixture
from artifact_limits.evaluation.reconstruction_metrics import rrmse_t
from artifact_limits.methods.deep.ic_unet import ICUNet


@dataclass
class OracleResult:
    rrmse_t_mean: float            # \hat D^{up}_{MMSE} (RRMSE-T units)
    rrmse_t_per_kind: dict[str, float]
    n_train_segments: int
    n_test_segments: int
    n_parameters: int
    epochs_trained: int


def fit_oracle(d1: D1Split,
                kinds: Iterable[str] = ("EOG", "EMG"),
                channels: tuple[int, ...] = (32, 64, 128, 256, 512),
                epochs: int = 150,
                batch_size: int = 64,
                lr: float = 1e-3,
                patience: int = 20,
                rng_seed: int = 0) -> OracleResult:
    """Train a wide IC-U-Net per artifact kind and return an MMSE upper bound."""
    rrmse_per_kind: dict[str, float] = {}
    n_params = 0
    n_train_total = 0
    n_test_total = 0
    for kind in kinds:
        # Build the oracle as a TorchDenoiserBase subclass (ICUNet handles it).
        model = ICUNet(channels=channels,
                       epochs=epochs, batch_size=batch_size, lr=lr,
                       patience=patience, seed=rng_seed)
        model.fit(d1.s_train, d1.a_train[kind])
        n_params = max(n_params, model.n_parameters())
        # Held-out evaluation on the test pool's SNR-stratified mixture.
        rng = np.random.default_rng(rng_seed)
        mix = make_stratified_mixture(d1.s_test, d1.a_test[kind],
                                      snr_db_levels=SNR_DB_LEVELS, rng=rng)
        s_hat = model.transform(mix["x"])
        rrmse_per_kind[kind] = float(rrmse_t(s_hat, mix["s"]).mean())
        n_test_total += mix["s"].shape[0]
        n_train_total += d1.s_train.shape[0]

    return OracleResult(
        rrmse_t_mean=float(np.mean(list(rrmse_per_kind.values()))),
        rrmse_t_per_kind=rrmse_per_kind,
        n_train_segments=n_train_total,
        n_test_segments=n_test_total,
        n_parameters=n_params,
        epochs_trained=epochs,
    )
