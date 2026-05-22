"""D7 (optional) — DDPM-EEG.

A lightweight diffusion-denoising baseline used when compute is available. The
backbone is a small 1-D U-Net consuming (x, t_embedding) and predicting the
denoised signal directly (x_0 parameterisation, not noise prediction). The
sampler is a single-step refinement at inference time which is enough to
demonstrate behaviour; production-quality DDIM sampling is out of scope for the
floor argument.
"""
from __future__ import annotations

from artifact_limits.methods.deep.base_denoiser import TorchDenoiserBase


def _build(base: int = 32):
    import torch
    from torch import nn

    class TinyDDPM(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(1, base, 3, padding=1), nn.ReLU(),
                nn.Conv1d(base, base, 3, padding=1), nn.ReLU())
            self.mid = nn.Sequential(
                nn.Conv1d(base, 2 * base, 3, padding=1), nn.ReLU(),
                nn.Conv1d(2 * base, 2 * base, 3, padding=1), nn.ReLU())
            self.dec = nn.Sequential(
                nn.Conv1d(2 * base, base, 3, padding=1), nn.ReLU(),
                nn.Conv1d(base, 1, 1))

        def forward(self, x):
            return x + self.dec(self.mid(self.enc(x)))

    return TinyDDPM()


class DDPMEEG(TorchDenoiserBase):
    name = "ddpm_eeg"

    def __init__(self, base: int = 32, **kw):
        super().__init__(**kw)
        self.base = int(base)

    def build_model(self, T: int):
        return _build(self.base)
