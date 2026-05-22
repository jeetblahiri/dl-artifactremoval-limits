"""D1 — SimpleCNN (EEGdenoiseNet original).

Four conv layers, kernel 3, BN + ReLU, residual to input. Configurable
``widths`` lets the same module power Experiment E3 (capacity scaling).
"""
from __future__ import annotations

from typing import Sequence

from artifact_limits.methods.deep.base_denoiser import TorchDenoiserBase


def _build(widths: Sequence[int]):
    import torch
    from torch import nn

    class SimpleCNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_ch = 1
            for w in widths:
                layers += [nn.Conv1d(in_ch, w, kernel_size=3, padding=1),
                           nn.BatchNorm1d(w),
                           nn.ReLU()]
                in_ch = w
            layers.append(nn.Conv1d(in_ch, 1, kernel_size=1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return x + self.net(x)              # residual

    return SimpleCNNModel()


class SimpleCNN(TorchDenoiserBase):
    name = "simple_cnn"

    def __init__(self, widths: Sequence[int] = (16, 32, 64, 32), **kw):
        super().__init__(**kw)
        self.widths = tuple(int(w) for w in widths)

    def build_model(self, T: int):
        return _build(self.widths)
