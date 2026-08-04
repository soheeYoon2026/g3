"""Geometry-conditioned implicit pressure/velocity field surrogate."""

from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("ImplicitFieldNet requires torch") from exc


class GeometryEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, latent_dim, 1), nn.ReLU(),
        )

    def forward(self, geometry):
        features = self.net(geometry.transpose(1, 2))
        return torch.max(features, dim=2).values


class FourierCoordinates(nn.Module):
    def __init__(self, levels: int = 6):
        super().__init__()
        frequencies = (2.0 ** torch.arange(levels)) * math.pi
        self.register_buffer("frequencies", frequencies)
        self.out_dim = 3 + 3 * 2 * levels

    def forward(self, points):
        phase = points.unsqueeze(-1) * self.frequencies
        return torch.cat([points, torch.sin(phase).flatten(-2), torch.cos(phase).flatten(-2)], dim=-1)


class ImplicitFieldNet(nn.Module):
    """Predict ``[Cp, Ux/Uref, Uy/Uref, Uz/Uref]`` at arbitrary query points."""

    def __init__(
        self,
        condition_dim: int = 8,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        fourier_levels: int = 6,
    ):
        super().__init__()
        self.geometry_encoder = GeometryEncoder(latent_dim=latent_dim)
        self.coordinates = FourierCoordinates(fourier_levels)
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        input_dim = latent_dim + 64 + self.coordinates.out_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def encode(self, geometry, conditions):
        return self.geometry_encoder(geometry), self.condition_encoder(conditions)

    def decode(self, latent, condition_latent, query_points):
        q = self.coordinates(query_points)
        n = query_points.shape[1]
        context = torch.cat([latent, condition_latent], dim=1).unsqueeze(1).expand(-1, n, -1)
        return self.decoder(torch.cat([q, context], dim=2))

    def forward(self, geometry, conditions, query_points):
        latent, condition_latent = self.encode(geometry, conditions)
        return self.decode(latent, condition_latent, query_points)


class DragHead(nn.Module):
    """Auxiliary global Cd regressor sharing the field model's context."""

    def __init__(self, latent_dim: int = 256, condition_latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + condition_latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, latent, condition_latent):
        return self.net(torch.cat([latent, condition_latent], dim=1))


class LiftHead(DragHead):
    """Auxiliary global Cl regressor sharing the field model's context.

    ``LiftHead`` deliberately has the same architecture as ``DragHead`` while
    retaining a separate checkpoint key.  That lets a field checkpoint carry
    independent Cd and signed Cl targets without changing the established v2
    drag-head state format.
    """


class CoefficientExpert(nn.Module):
    """One independently calibrated Cd/Cl expert on a shared field encoder."""

    def __init__(
        self,
        latent_dim: int = 256,
        condition_latent_dim: int = 64,
        predict_drag: bool = True,
        predict_lift: bool = True,
    ):
        super().__init__()
        self.drag = DragHead(latent_dim, condition_latent_dim) if predict_drag else None
        self.lift = LiftHead(latent_dim, condition_latent_dim) if predict_lift else None

    def forward(self, latent, condition_latent):
        return {
            "cd": self.drag(latent, condition_latent) if self.drag is not None else None,
            "cl": self.lift(latent, condition_latent) if self.lift is not None else None,
        }


class CoefficientExpertBank(nn.Module):
    """Named coefficient experts sharing one geometry/condition representation.

    Experts are deliberately selected by an explicit label domain (for
    example ``g2_su2_clean`` or ``g4_lbm``).  A geometry-only gate cannot
    distinguish two solvers that evaluated the same STL with different force
    normalizations, so silently blending those labels would be invalid.
    """

    def __init__(self, expert_configs: dict[str, dict] | list[str]):
        super().__init__()
        if isinstance(expert_configs, list):
            expert_configs = {name: {} for name in expert_configs}
        self.experts = nn.ModuleDict({
            name: CoefficientExpert(**config) for name, config in expert_configs.items()
        })

    def forward(self, name: str, latent, condition_latent):
        if name not in self.experts:
            raise KeyError(f"unknown coefficient expert {name!r}; choose from {list(self.experts)}")
        return self.experts[name](latent, condition_latent)
