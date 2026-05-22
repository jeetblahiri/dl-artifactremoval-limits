"""Shared training loop for every deep denoiser.

Each subclass supplies a `build_model(T) -> nn.Module` factory; this base class
handles SNR-stratified mixture generation, training, early stopping, and
inference. We deliberately keep this lightweight (no Lightning trainer
configuration knobs) so the same loop drives all six architectures.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from artifact_limits import SNR_DB_LEVELS
from artifact_limits.data.load_D1 import make_stratified_mixture
from artifact_limits.methods.base import DenoiserBase


class TorchDenoiserBase(DenoiserBase):
    """Generic PyTorch training loop for any (1×T → 1×T) denoiser."""

    name = "torch_base"

    def __init__(self,
                 epochs: int = 30,
                 batch_size: int = 64,
                 lr: float = 1e-3,
                 snr_db_levels: Iterable[float] = SNR_DB_LEVELS,
                 device: str | None = None,
                 seed: int = 42,
                 patience: int = 8):
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.snr_db_levels = tuple(snr_db_levels)
        self.device_arg = device
        self.seed = int(seed)
        self.patience = int(patience)
        self.model = None
        self.T: int | None = None
        self._best_state = None

    # subclasses override
    def build_model(self, T: int):                                  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- the standard fit/transform interface ----

    def _device(self):
        import torch
        if self.device_arg is not None:
            return torch.device(self.device_arg)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(self, s_pool: np.ndarray, a_pool: np.ndarray) -> "TorchDenoiserBase":
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        T = s_pool.shape[1]
        self.T = T

        # Train/val partition inside the pool (random 90/10 inside this fit call).
        idx = rng.permutation(s_pool.shape[0])
        n_tr = int(round(0.9 * s_pool.shape[0]))
        s_tr, s_va = s_pool[idx[:n_tr]], s_pool[idx[n_tr:]]
        a_tr = a_pool[rng.integers(0, a_pool.shape[0], size=s_tr.shape[0])]
        a_va = a_pool[rng.integers(0, a_pool.shape[0], size=s_va.shape[0])]

        mix_tr = make_stratified_mixture(s_tr, a_tr, snr_db_levels=self.snr_db_levels, rng=rng)
        mix_va = make_stratified_mixture(s_va, a_va, snr_db_levels=self.snr_db_levels, rng=rng)

        x_tr = torch.from_numpy(mix_tr["x"]).float().unsqueeze(1)
        y_tr = torch.from_numpy(mix_tr["s"]).float().unsqueeze(1)
        x_va = torch.from_numpy(mix_va["x"]).float().unsqueeze(1)
        y_va = torch.from_numpy(mix_va["s"]).float().unsqueeze(1)

        device = self._device()
        self.model = self.build_model(T).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        ds_tr = TensorDataset(x_tr, y_tr)
        ds_va = TensorDataset(x_va, y_va)
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=True)
        dl_va = DataLoader(ds_va, batch_size=self.batch_size, shuffle=False)

        best_val = float("inf")
        patience = 0
        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                vlosses = []
                for xb, yb in dl_va:
                    xb, yb = xb.to(device), yb.to(device)
                    vlosses.append(float(loss_fn(self.model(xb), yb).item()))
            vloss = float(np.mean(vlosses)) if vlosses else float("inf")
            if vloss < best_val - 1e-6:
                best_val = vloss
                patience = 0
                self._best_state = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.patience:
                    break
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        import torch
        if self.model is None:
            raise RuntimeError(f"{self.name}: must call .fit() before .transform().")
        device = self._device()
        self.model.eval()
        x_t = torch.from_numpy(np.asarray(x)).float().unsqueeze(1).to(device)
        out = []
        with torch.no_grad():
            for i in range(0, x_t.shape[0], self.batch_size * 4):
                out.append(self.model(x_t[i:i + self.batch_size * 4]).cpu().numpy())
        return np.concatenate(out, axis=0).squeeze(1).astype(np.float32)

    def n_parameters(self) -> int:
        if self.model is None:
            return 0
        return int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
