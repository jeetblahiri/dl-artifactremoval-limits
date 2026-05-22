"""Mutual-information estimators.

- Kraskov–Stögbauer–Grassberger (KSG) estimator (algorithm 1).
- Gaussian-copula MI (Ince-style).
- MINE — a thin PyTorch implementation, optional.

All estimators return MI in nats. The expected use case is on scalar or
low-dimensional joints (segment-level features), so we lean on cKDTree.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import norm, rankdata


def _atleast_2d_column(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    return x


def ksg_mi(x: np.ndarray,
           y: np.ndarray,
           k: int = 4,
           base: str = "nat") -> float:
    """KSG algorithm-1 estimator of I(X; Y).

    Inputs are (n,) or (n, d). The implementation:

    1. Concatenate (x, y) → z; build cKDTree on z under Chebyshev distance.
    2. For each i, ε_i = distance to its k-th neighbour in z.
    3. Count n_x(i) = number of x-points strictly within ε_i in the x-marginal;
       similarly n_y(i).
    4. Î = ψ(k) + ψ(N) - <ψ(n_x+1) + ψ(n_y+1)>.
    """
    x = _atleast_2d_column(x).astype(np.float64)
    y = _atleast_2d_column(y).astype(np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have matching first dimension.")
    n = x.shape[0]
    if n <= k + 1:
        return 0.0

    # Add tiny jitter to break ties (KSG assumes continuous distributions).
    # Jitter BEFORE building trees so it actually affects both ε_i and the
    # marginal-ball counts.
    rng = np.random.default_rng(0)
    x = x + rng.normal(scale=1e-10, size=x.shape)
    y = y + rng.normal(scale=1e-10, size=y.shape)
    z = np.concatenate([x, y], axis=1)
    tree_z = cKDTree(z)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    eps = tree_z.query(z, k=k + 1, p=np.inf)[0][:, -1]  # k+1 includes self at 0
    nx = np.array([len(tree_x.query_ball_point(x[i], r=eps[i] - 1e-12, p=np.inf)) - 1
                   for i in range(n)], dtype=np.int64)
    ny = np.array([len(tree_y.query_ball_point(y[i], r=eps[i] - 1e-12, p=np.inf)) - 1
                   for i in range(n)], dtype=np.int64)
    nx = np.clip(nx, 1, None)
    ny = np.clip(ny, 1, None)

    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    mi = float(max(mi, 0.0))
    return mi / np.log(2) if base == "bit" else mi


def gaussian_copula_mi(x: np.ndarray, y: np.ndarray) -> float:
    """MI under the Gaussian-copula assumption.

    Convert each variable to standard-normal via the empirical rank transform,
    then I(X;Y) = -½ log det(R) where R is the joint correlation matrix.
    """
    x = _atleast_2d_column(x).astype(np.float64)
    y = _atleast_2d_column(y).astype(np.float64)
    n = x.shape[0]

    def to_normal(arr: np.ndarray) -> np.ndarray:
        u = (rankdata(arr, axis=0) - 0.5) / n
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return norm.ppf(u)

    xg = to_normal(x)
    yg = to_normal(y)
    joint = np.concatenate([xg, yg], axis=1)
    R = np.corrcoef(joint.T)
    dx, dy = xg.shape[1], yg.shape[1]
    Rxx = R[:dx, :dx]
    Ryy = R[dx:, dx:]
    sign_xx, logdet_xx = np.linalg.slogdet(Rxx)
    sign_yy, logdet_yy = np.linalg.slogdet(Ryy)
    sign_jj, logdet_jj = np.linalg.slogdet(R)
    if min(sign_xx, sign_yy, sign_jj) <= 0:
        return 0.0
    mi = 0.5 * (logdet_xx + logdet_yy - logdet_jj)
    return float(max(mi, 0.0))


def permutation_null(x: np.ndarray, y: np.ndarray,
                     estimator=ksg_mi,
                     n_perm: int = 200,
                     seed: int = 0,
                     **kwargs) -> np.ndarray:
    """Distribution of MI(x, π(y)) over random permutations π of y."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    out = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(y.shape[0])
        out[i] = estimator(x, y[perm], **kwargs)
    return out


# ----- MINE (optional, PyTorch) -----------------------------------------------

def mine_mi(x: np.ndarray,
            y: np.ndarray,
            n_iter: int = 2000,
            hidden: int = 64,
            lr: float = 1e-3,
            seed: int = 0,
            device: Optional[str] = None) -> float:  # pragma: no cover - optional
    """Tiny MINE implementation. Returns the running mean of the lower bound."""
    try:
        import torch
        from torch import nn
    except Exception:
        return 0.0
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    x = _atleast_2d_column(x).astype(np.float32)
    y = _atleast_2d_column(y).astype(np.float32)
    n, dx = x.shape
    dy = y.shape[1]

    class T(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Linear(dx + dy, hidden), nn.ELU(),
                nn.Linear(hidden, hidden), nn.ELU(),
                nn.Linear(hidden, 1))

        def forward(self, x_, y_):
            return self.f(torch.cat([x_, y_], dim=-1))

    net = T().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xT = torch.from_numpy(x).to(device)
    yT = torch.from_numpy(y).to(device)
    rng = np.random.default_rng(seed)
    last = 0.0
    for _ in range(n_iter):
        perm = rng.permutation(n)
        joint = net(xT, yT)
        marg = net(xT, yT[perm])
        mi = joint.mean() - torch.logsumexp(marg.squeeze(-1), 0) + np.log(n)
        loss = -mi
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(mi.detach().cpu())
    return float(max(last, 0.0))
