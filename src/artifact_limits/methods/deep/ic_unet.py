"""D6 — IC-U-Net: 5-level U-Net for 1-D signals."""
from __future__ import annotations

from typing import Sequence

from artifact_limits.methods.deep.base_denoiser import TorchDenoiserBase


def _build(channels: Sequence[int] = (16, 32, 64, 128, 256)):
    import torch
    from torch import nn

    class DoubleConv(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv1d(in_c, out_c, 3, padding=1),
                nn.BatchNorm1d(out_c), nn.ReLU(),
                nn.Conv1d(out_c, out_c, 3, padding=1),
                nn.BatchNorm1d(out_c), nn.ReLU())

        def forward(self, x):
            return self.f(x)

    class UNet1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.downs = nn.ModuleList()
            in_c = 1
            for c in channels[:-1]:
                self.downs.append(DoubleConv(in_c, c))
                in_c = c
            self.bottleneck = DoubleConv(channels[-2], channels[-1])
            self.ups = nn.ModuleList()
            self.up_convs = nn.ModuleList()
            for i in range(len(channels) - 1, 0, -1):
                # After cat(up_conv(h), skip) we have 2*channels[i-1] channels,
                # regardless of channels[i]. Don't tie the input dim to channels[i].
                self.up_convs.append(nn.ConvTranspose1d(channels[i], channels[i - 1], 2, stride=2))
                self.ups.append(DoubleConv(2 * channels[i - 1], channels[i - 1]))
            self.out = nn.Conv1d(channels[0], 1, 1)
            self.pool = nn.MaxPool1d(2)

        def forward(self, x):
            T = x.shape[-1]
            skips = []
            h = x
            for d in self.downs:
                h = d(h)
                skips.append(h)
                h = self.pool(h)
            h = self.bottleneck(h)
            for up_conv, up, skip in zip(self.up_convs, self.ups, reversed(skips)):
                h = up_conv(h)
                if h.shape[-1] != skip.shape[-1]:
                    h = nn.functional.pad(h, (0, skip.shape[-1] - h.shape[-1]))
                h = up(torch.cat([h, skip], dim=1))
            y = self.out(h)
            if y.shape[-1] != T:
                y = y[..., :T]
            return x + y

    return UNet1D()


class ICUNet(TorchDenoiserBase):
    name = "ic_unet"

    def __init__(self, channels: Sequence[int] = (16, 32, 64, 128, 256), **kw):
        super().__init__(**kw)
        self.channels = tuple(int(c) for c in channels)

    def build_model(self, T: int):
        return _build(self.channels)
