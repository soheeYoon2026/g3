"""PointNet regressor: STL surface points -> Cd/Cl (the real Phase 0/1 model).

Requires torch (not installed in the Phase 0 spike env — see requirements.txt).
This is a faithful PointNet-style permutation-invariant regressor: a shared
per-point MLP, symmetric max-pool to a global feature, then a regression head.
Unlike the pooled sklearn baseline it keeps per-point structure, which is what
lets a network learn where on the body the drag comes from.

It is deliberately the *simplest* thing that is architecturally on the path to
the production model: DoMINO / GINO are, loosely, PointNet-style local operators
with geometric conditioning and field (not just scalar) outputs. Prove the
scalar regressor first, then graduate to surface-field prediction.

Input : (B, N, 6) — xyz + surface normal per point (see PointCloud.features).
Output: (B, len(TARGETS)) — Cd, Cl.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - torch optional in Phase 0
    raise ImportError(
        "PointNetRegressor needs torch. Install the GPU stack from "
        "g3/requirements.txt, or use PooledMLPRegressor for the torch-free path."
    ) from exc

from ..config import TARGETS


class _SharedMLP(nn.Module):
    """1x1 conv stack applied identically to every point."""

    def __init__(self, sizes):
        super().__init__()
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Conv1d(a, b, 1), nn.BatchNorm1d(b), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, C, N)
        return self.net(x)


class PointNetRegressor(nn.Module):
    def __init__(self, in_features: int = 6, out_dim: int = len(TARGETS),
                 point_sizes=(64, 128, 1024), head_sizes=(512, 256)):
        super().__init__()
        self.encoder = _SharedMLP((in_features, *point_sizes))
        head = []
        prev = point_sizes[-1]
        for h in head_sizes:
            head += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        head += [nn.Linear(prev, out_dim)]
        self.head = nn.Sequential(*head)

    def forward(self, x):  # x: (B, N, F)
        x = x.transpose(1, 2)            # -> (B, F, N)
        feats = self.encoder(x)          # -> (B, 1024, N)
        glob = torch.max(feats, dim=2).values  # symmetric pool -> (B, 1024)
        return self.head(glob)           # -> (B, out_dim)

    # Convenience: predict from a list of PointCloud (matches baseline API).
    @torch.no_grad()
    def predict_clouds(self, clouds, device="cpu"):
        self.eval()
        feats = np.stack([c.features for c in clouds]).astype(np.float32)
        t = torch.from_numpy(feats).to(device)
        return self(t).cpu().numpy()
