"""D4 — DeepSeparator (encoder–decoder with attention).

This is a lightweight re-implementation in the spirit of Yu et al. 2022; not the
original released checkpoint. The encoder downsamples with strided convs, a
bottleneck self-attention layer mixes information across time, and the decoder
upsamples back to T.
"""
from __future__ import annotations

from artifact_limits.methods.deep.base_denoiser import TorchDenoiserBase


def _build(base: int = 32, depth: int = 3, heads: int = 4):
    import torch
    from torch import nn

    class DownBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv1d(in_c, out_c, 3, stride=2, padding=1),
                nn.BatchNorm1d(out_c), nn.ReLU())

        def forward(self, x):
            return self.f(x)

    class UpBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.f = nn.Sequential(
                nn.ConvTranspose1d(in_c, out_c, 4, stride=2, padding=1),
                nn.BatchNorm1d(out_c), nn.ReLU())

        def forward(self, x):
            return self.f(x)

    class Attention(nn.Module):
        def __init__(self, c, heads):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim=c, num_heads=heads,
                                              batch_first=True)

        def forward(self, x):                    # (B, C, T)
            z = x.transpose(1, 2)
            z, _ = self.attn(z, z, z, need_weights=False)
            return z.transpose(1, 2) + x

    class DeepSeparatorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Conv1d(1, base, 3, padding=1)
            chans = [base * (2 ** i) for i in range(depth + 1)]
            self.encs = nn.ModuleList([DownBlock(chans[i], chans[i + 1]) for i in range(depth)])
            self.bottleneck = Attention(chans[-1], heads=min(heads, max(1, chans[-1] // 8)))
            self.decs = nn.ModuleList([UpBlock(chans[i + 1], chans[i]) for i in range(depth - 1, -1, -1)])
            self.out = nn.Conv1d(base, 1, 1)

        def forward(self, x):
            h = self.in_proj(x)
            skips = []
            for e in self.encs:
                skips.append(h)
                h = e(h)
            h = self.bottleneck(h)
            for d in self.decs:
                h = d(h)
                if skips:
                    s = skips.pop()
                    if h.shape[-1] != s.shape[-1]:
                        h = h[..., :s.shape[-1]]
                    h = h + s
            return x + self.out(h)

    return DeepSeparatorModel()


class DeepSeparator(TorchDenoiserBase):
    name = "deep_separator"

    def __init__(self, base: int = 32, depth: int = 3, heads: int = 4, **kw):
        super().__init__(**kw)
        self.base, self.depth, self.heads = int(base), int(depth), int(heads)

    def build_model(self, T: int):
        return _build(self.base, self.depth, self.heads)
