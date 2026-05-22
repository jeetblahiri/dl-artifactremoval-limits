"""Optional loader for Sleep-EDF subject SC4001 (E5 audit only).

If MNE-Python is available *and* the recording is downloadable from PhysioNet,
this module fetches the EDF and returns aligned (EEG, EOG, EMG) arrays. If the
download fails (no network, etc.), it produces a synthetic surrogate so the rest
of the pipeline can be run; a warning is printed and the result is flagged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from artifact_limits.data.paths import raw_dir


@dataclass
class D3Audit:
    eeg: np.ndarray           # (n_samples,)
    eog: np.ndarray
    emg: np.ndarray
    fs: float
    is_synthetic: bool
    annotations: dict | None  # sleep-stage annotation per window (optional)


def _synthetic_audit(duration_s: float = 300.0,
                     fs: float = 100.0,
                     seed: int = 42) -> D3Audit:
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    z = rng.standard_normal(n)                  # latent state (autonomic)
    z = np.convolve(z, np.ones(int(2 * fs)) / (2 * fs), mode="same")

    eeg = rng.standard_normal(n) * 0.5
    eeg = eeg + 0.3 * np.sin(2 * np.pi * 10.0 * t + 0.5 * z)   # alpha modulated by latent
    eog = rng.standard_normal(n) * 0.5
    eog = eog + 0.5 * z                                        # share latent
    emg = rng.standard_normal(n)
    emg = emg + 0.2 * np.abs(z)                                # share latent (rectified)
    return D3Audit(eeg=eeg.astype(np.float32),
                   eog=eog.astype(np.float32),
                   emg=emg.astype(np.float32),
                   fs=fs, is_synthetic=True, annotations=None)


def _candidate_edf_paths(subject: str) -> list:
    """All EDF filenames Sleep-EDF uses for a SC subject (E0 and similar)."""
    cands = []
    for suffix in ("E0", "F0", "G0", "H0"):
        cands.append(raw_dir() / f"{subject}{suffix}-PSG.edf")
    return cands


def _download_sleep_edf(subject: str) -> Path | None:
    """Fetch the PSG EDF for a Sleep-EDF Cassette subject by direct HTTPS.

    Returns the local path on success, None on failure. Avoids MNE's lzma path.
    """
    import urllib.request
    import ssl
    base = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
    ctx = ssl.create_default_context()
    for cand in _candidate_edf_paths(subject):
        if cand.exists():
            return cand
        url = base + cand.name
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=120) as r, open(cand, "wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            return cand
        except Exception:
            try:
                cand.unlink(missing_ok=True)
            except Exception:
                pass
            continue
    return None


def load_D3_audit(subject: str = "SC4001",
                  duration_s: float = 600.0) -> D3Audit:
    """Try to load Sleep-EDF; fall back to a synthetic correlated triple."""
    try:
        import mne                                            # noqa: F401
        # Prefer a locally-cached EDF; otherwise direct HTTPS to PhysioNet.
        psg = _download_sleep_edf(subject)
        if psg is None:
            from mne.datasets.sleep_physionet.age import fetch_data
            files = fetch_data(subjects=[0], recording=[1], path=str(raw_dir()))
            psg, _ = files[0]
        raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR")
        picks = {ch.lower(): ch for ch in raw.ch_names}
        eeg_ch = next((c for k, c in picks.items() if "eeg" in k), None)
        eog_ch = next((c for k, c in picks.items() if "eog" in k), None)
        emg_ch = next((c for k, c in picks.items() if "emg" in k), None)
        if eeg_ch is None or eog_ch is None or emg_ch is None:
            raise RuntimeError("Required EEG/EOG/EMG channels missing in SC4001.")
        raw.filter(0.5, 45.0, picks=[eeg_ch, eog_ch, emg_ch], verbose="ERROR")
        data, times = raw[[eeg_ch, eog_ch, emg_ch]]
        n = min(data.shape[1], int(duration_s * raw.info["sfreq"]))
        return D3Audit(eeg=data[0, :n].astype(np.float32),
                       eog=data[1, :n].astype(np.float32),
                       emg=data[2, :n].astype(np.float32),
                       fs=float(raw.info["sfreq"]),
                       is_synthetic=False, annotations=None)
    except Exception as exc:  # pragma: no cover - fall-back path
        print(f"[load_D3_audit] WARNING: real Sleep-EDF unavailable ({exc!r}); using surrogate.")
        return _synthetic_audit(duration_s=duration_s, fs=100.0)


def main() -> None:
    rec = load_D3_audit()
    print(f"[load_D3_audit] fs={rec.fs} Hz, samples={rec.eeg.size}, synthetic={rec.is_synthetic}")


if __name__ == "__main__":
    main()
