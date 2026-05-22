"""D5 — EEGDenoiseFormer: transformer encoder over conv-embedded segments."""
from __future__ import annotations

from artifact_limits.methods.deep.base_denoiser import TorchDenoiserBase


def _build(model_dim: int = 64, n_heads: int = 4, n_layers: int = 4,
           kernel: int = 16, stride: int = 8):
    import torch
    from torch import nn

    class EEGDenoiseFormerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Conv1d(1, model_dim, kernel_size=kernel,
                                   stride=stride, padding=kernel // 2)
            enc_layer = nn.TransformerEncoderLayer(d_model=model_dim,
                                                   nhead=n_heads,
                                                   dim_feedforward=4 * model_dim,
                                                   batch_first=True,
                                                   activation="gelu")
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.unembed = nn.ConvTranspose1d(model_dim, 1, kernel_size=kernel,
                                              stride=stride, padding=kernel // 2)

        def forward(self, x):                   # (B, 1, T)
            T = x.shape[-1]
            z = self.embed(x).transpose(1, 2)   # (B, T', C)
            z = self.encoder(z).transpose(1, 2)  # (B, C, T')
            y = self.unembed(z)
            if y.shape[-1] != T:
                # pad / crop to T
                if y.shape[-1] < T:
                    pad = T - y.shape[-1]
                    y = nn.functional.pad(y, (0, pad))
                else:
                    y = y[..., :T]
            return x + y

    return EEGDenoiseFormerModel()


class EEGDenoiseFormer(TorchDenoiserBase):
    name = "eeg_denoise_former"

    def __init__(self, model_dim: int = 64, n_heads: int = 4, n_layers: int = 4,
                 kernel: int = 16, stride: int = 8, **kw):
        super().__init__(**kw)
        self.model_dim = int(model_dim)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.kernel = int(kernel)
        self.stride = int(stride)

    def build_model(self, T: int):
        return _build(self.model_dim, self.n_heads, self.n_layers, self.kernel, self.stride)
