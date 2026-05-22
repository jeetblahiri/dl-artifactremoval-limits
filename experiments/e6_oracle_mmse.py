"""E6 --- Oracle non-linear MMSE upper bound on EEGdenoiseNet.

Train a wide IC-U-Net (\(\sim\!10^7\) parameters) on the training pool with an
extended schedule and report held-out RRMSE-T. The number is an empirical
upper bound on \(D_{\mathrm{MMSE}}\) for the synthetic-mixing law \(Q\).
"""
from __future__ import annotations

from pathlib import Path

from artifact_limits.data.load_D1 import load_D1
from artifact_limits.theory.oracle_mmse import fit_oracle
from experiments.shared.train_loops import repeat_seeds


def run(seed: int, cfg: dict, out_dir: Path) -> dict:
    next(repeat_seeds([seed]))
    d1 = load_D1()
    kinds = tuple(cfg.get("artifact_kinds", ("EOG", "EMG")))
    channels = tuple(cfg.get("channels", [32, 64, 128, 256, 512]))
    epochs = int(cfg.get("epochs", 80))
    batch_size = int(cfg.get("batch_size", 64))
    lr = float(cfg.get("lr", 1.0e-3))
    patience = int(cfg.get("patience", 15))
    out = fit_oracle(d1,
                     kinds=kinds,
                     channels=channels,
                     epochs=epochs,
                     batch_size=batch_size,
                     lr=lr,
                     patience=patience,
                     rng_seed=seed)
    return {
        "experiment": "E6",
        "seed": seed,
        "channels": list(channels),
        "epochs_configured": epochs,
        "epochs_used": out.epochs_trained,
        "n_parameters": out.n_parameters,
        "n_train_segments": out.n_train_segments,
        "n_test_segments": out.n_test_segments,
        "rrmse_t_per_kind": out.rrmse_t_per_kind,
        "rrmse_t_mean": out.rrmse_t_mean,
    }
