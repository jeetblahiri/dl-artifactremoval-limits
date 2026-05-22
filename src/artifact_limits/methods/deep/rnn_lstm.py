"""D3 — Bi-directional LSTM with a small conv front-end."""
from __future__ import annotations

from artifact_limits.methods.deep.base_denoiser import TorchDenoiserBase


def _build(front_conv: int = 16, hidden: int = 64, layers: int = 2):
    import torch
    from torch import nn

    class RNNLSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = nn.Conv1d(1, front_conv, 3, padding=1)
            self.lstm = nn.LSTM(input_size=front_conv,
                                hidden_size=hidden,
                                num_layers=layers,
                                bidirectional=True,
                                batch_first=True)
            self.head = nn.Linear(2 * hidden, 1)

        def forward(self, x):                       # x: (B, 1, T)
            z = self.front(x).transpose(1, 2)       # (B, T, C)
            h, _ = self.lstm(z)
            y = self.head(h).transpose(1, 2)        # (B, 1, T)
            return x + y                            # residual

    return RNNLSTMModel()


class RNNLSTM(TorchDenoiserBase):
    name = "rnn_lstm"

    def __init__(self, front_conv: int = 16, hidden: int = 64, layers: int = 2, **kw):
        super().__init__(**kw)
        self.front_conv, self.hidden, self.layers = int(front_conv), int(hidden), int(layers)

    def build_model(self, T: int):
        return _build(self.front_conv, self.hidden, self.layers)
