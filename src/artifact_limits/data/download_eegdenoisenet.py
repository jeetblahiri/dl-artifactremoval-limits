"""Fetch the EEGdenoiseNet release.

The authoritative download is the GitHub release at:
    https://github.com/ncclabsustech/EEGdenoiseNet

Because the release is occasionally re-hosted on Zenodo / Google Drive and direct
URLs change, this module supports three modes:

1. If `EEG_all_epochs.npy`, `EOG_all_epochs.npy`, `EMG_all_epochs.npy` already exist
   under `data/raw/`, do nothing (idempotent).
2. If the environment variable ``EEGDENOISENET_LOCAL`` points to a directory
   containing the three files, copy them in.
3. Otherwise synthesize a small surrogate release so the rest of the pipeline can
   be exercised end-to-end. The surrogate is **not** the real benchmark; a
   warning is printed.

After acquisition, the official train/val/test split is materialised to
``data/splits/eegdenoisenet_split.json`` using a deterministic 80/10/10
segment-index partition. Once the canonical split indices are committed, swap
``_synthesize_split`` for a load of the committed JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from artifact_limits import SAMPLING_FREQ, SEGMENT_LEN
from artifact_limits.data.paths import raw_dir, splits_dir

EEG_FN = "EEG_all_epochs.npy"
EOG_FN = "EOG_all_epochs.npy"
EMG_FN = "EMG_all_epochs.npy"


def _have_canonical(raw: Path) -> bool:
    return all((raw / fn).exists() for fn in (EEG_FN, EOG_FN, EMG_FN))


def _copy_from_local(src: Path, dst: Path) -> None:
    for fn in (EEG_FN, EOG_FN, EMG_FN):
        s = src / fn
        if not s.exists():
            raise FileNotFoundError(f"Expected {s} in EEGDENOISENET_LOCAL")
        shutil.copy(s, dst / fn)


def _synthesize_surrogate(raw: Path,
                          n_eeg: int = 4514,
                          n_eog: int = 3400,
                          n_emg: int = 5598,
                          T: int = SEGMENT_LEN,
                          fs: int = SAMPLING_FREQ,
                          rng_seed: int = 0) -> None:
    """Cheap surrogate matching EEGdenoiseNet shapes.

    EEG: pink-ish noise with 1/f slope and an alpha bump.
    EOG: smoothed Gaussian with low-frequency drift.
    EMG: white-ish noise high-passed at 20 Hz.
    """
    print("[download_eegdenoisenet] WARNING: canonical files not found.")
    print("[download_eegdenoisenet] Synthesising surrogate arrays. "
          "Set EEGDENOISENET_LOCAL or place the real .npy files in data/raw/ to disable.")
    rng = np.random.default_rng(rng_seed)

    def pink(n_segs: int, slope: float = 1.0) -> np.ndarray:
        freqs = np.fft.rfftfreq(T, d=1.0 / fs)
        amp = np.zeros_like(freqs)
        amp[1:] = 1.0 / (freqs[1:] ** (slope / 2.0))
        spec = (rng.standard_normal((n_segs, freqs.size))
                + 1j * rng.standard_normal((n_segs, freqs.size))) * amp
        x = np.fft.irfft(spec, n=T, axis=-1)
        x = x - x.mean(axis=-1, keepdims=True)
        x = x / (x.std(axis=-1, keepdims=True) + 1e-12)
        return x.astype(np.float32)

    eeg = pink(n_eeg, slope=1.0)
    # Add alpha bump
    t = np.arange(T) / fs
    phases = rng.uniform(0, 2 * np.pi, n_eeg)
    alpha = 0.4 * np.sin(2 * np.pi * 10.0 * t[None, :] + phases[:, None])
    eeg = eeg + alpha.astype(np.float32)

    eog = pink(n_eog, slope=2.0) * 3.0  # large amplitude, low-frequency
    emg_white = rng.standard_normal((n_emg, T)).astype(np.float32)
    # Crude high-pass via cumulative difference + scaling
    emg = np.diff(emg_white, axis=-1, prepend=emg_white[:, :1]) * 2.0

    np.save(raw / EEG_FN, eeg)
    np.save(raw / EOG_FN, eog)
    np.save(raw / EMG_FN, emg)


def _synthesize_split(n_eeg: int, n_eog: int, n_emg: int,
                      seed: int = 42,
                      train_frac: float = 0.8,
                      val_frac: float = 0.1) -> dict:
    rng = np.random.default_rng(seed)

    def split(n: int) -> dict:
        idx = rng.permutation(n)
        n_tr = int(round(train_frac * n))
        n_va = int(round(val_frac * n))
        return {
            "train": idx[:n_tr].tolist(),
            "val": idx[n_tr:n_tr + n_va].tolist(),
            "test": idx[n_tr + n_va:].tolist(),
        }

    return {
        "EEG": split(n_eeg),
        "EOG": split(n_eog),
        "EMG": split(n_emg),
        "seed": seed,
    }


def ensure_dataset() -> dict[str, Path]:
    raw = raw_dir()
    if not _have_canonical(raw):
        local = os.environ.get("EEGDENOISENET_LOCAL")
        if local and Path(local).exists():
            _copy_from_local(Path(local), raw)
        else:
            _synthesize_surrogate(raw)
    eeg = np.load(raw / EEG_FN, mmap_mode="r")
    eog = np.load(raw / EOG_FN, mmap_mode="r")
    emg = np.load(raw / EMG_FN, mmap_mode="r")
    print(f"[download_eegdenoisenet] shapes: EEG {eeg.shape}, EOG {eog.shape}, EMG {emg.shape}")
    split_path = splits_dir() / "eegdenoisenet_split.json"
    if not split_path.exists():
        split = _synthesize_split(eeg.shape[0], eog.shape[0], emg.shape[0])
        split_path.write_text(json.dumps(split, indent=2))
        print(f"[download_eegdenoisenet] split written: {split_path}")
    return {
        "EEG": raw / EEG_FN,
        "EOG": raw / EOG_FN,
        "EMG": raw / EMG_FN,
        "split": split_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EEGdenoiseNet or build surrogate.")
    parser.parse_args()
    ensure_dataset()


if __name__ == "__main__":
    main()
