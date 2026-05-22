"""Thin wrappers around artifact_limits.data used by every experiment script."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from artifact_limits.data.load_D1 import D1Split, load_D1, make_stratified_mixture


@dataclass
class TestMixture:
    x: np.ndarray
    s: np.ndarray
    a: np.ndarray
    snr_db: np.ndarray
    kind: str               # "EOG" or "EMG"


def load_eegdenoisenet() -> D1Split:
    return load_D1()


def make_test_mixtures(d1: D1Split,
                       rng: np.random.Generator,
                       kinds: tuple[str, ...] = ("EOG", "EMG")) -> list[TestMixture]:
    out = []
    for kind in kinds:
        mix = make_stratified_mixture(d1.s_test, d1.a_test[kind], rng=rng)
        out.append(TestMixture(x=mix["x"], s=mix["s"], a=mix["a"],
                               snr_db=mix["snr_db"], kind=kind))
    return out
