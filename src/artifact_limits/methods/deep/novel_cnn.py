"""D2 — NovelCNN: deeper residual variant from the original EEGdenoiseNet paper."""
from __future__ import annotations

from artifact_limits.methods.deep.base_denoiser import TorchDenoiserBase


def _build(width: int = 32, n_blocks: int = 3):
    import torch
    from torch import nn

    class ResBlock(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU(),
                nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c))
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.f(x) + x)

    class NovelCNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Conv1d(1, width, 3, padding=1)
            self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(n_blocks)])
            self.out_proj = nn.Conv1d(width, 1, 1)

        def forward(self, x):
            return x + self.out_proj(self.blocks(self.in_proj(x)))

    return NovelCNNModel()


class NovelCNN(TorchDenoiserBase):
    name = "novel_cnn"

    def __init__(self, width: int = 32, n_blocks: int = 3, **kw):
        super().__init__(**kw)
        self.width = int(width)
        self.n_blocks = int(n_blocks)

    def build_model(self, T: int):
        return _build(self.width, self.n_blocks)
