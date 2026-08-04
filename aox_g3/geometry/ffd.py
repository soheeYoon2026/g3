"""Free-form deformation (FFD) — the shared shape parametrisation.

AOX's optimisation loop (G1/G2/G4) deforms geometry by moving the control
points of an FFD lattice. G3 must speak the same language: the surrogate has to
accept an FFD displacement vector and return the resulting Cd/Cl so it can drop
into the existing optimisation loop.

This is a trivariate Bezier (Bernstein) volume. The important property we rely
on is **linear precision**: a regular control lattice reproduces the embedded
point exactly, so a zero displacement is the identity map. That means

    deform(delta) = base_points + W @ delta_flat

where ``W`` is a precomputed (N, L*M*N) weight matrix — cheap and, crucially,
linear in ``delta``, so gradients d(point)/d(delta) are just ``W`` and the whole
thing is trivially differentiable for gradient-based shape optimisation.
"""

from __future__ import annotations

import numpy as np
from scipy.special import comb

from ..config import FFD_PAD


def _bernstein(n: int, t: np.ndarray) -> np.ndarray:
    """Bernstein basis B_i^n(t) for i=0..n. ``t`` is (N,), returns (N, n+1)."""
    i = np.arange(n + 1)
    coeff = comb(n, i)  # (n+1,)
    t = t[:, None]
    return coeff * (t**i) * ((1.0 - t) ** (n - i))


class FFD:
    """A regular FFD lattice embedding a fixed set of points.

    Parameters
    ----------
    points : (N, 3) array
        The geometry (e.g. surface point cloud) to embed and deform.
    dims : (L, M, N) tuple
        Number of control points along x, y, z. Bezier degree is dim-1 per axis.
    pad : float
        Lattice padding beyond the point bounding box, as a fraction of size.
    """

    def __init__(self, points: np.ndarray, dims=(4, 3, 3), pad: float = FFD_PAD):
        points = np.asarray(points, dtype=np.float64)
        self.dims = tuple(int(d) for d in dims)
        L, M, N = self.dims

        lo = points.min(0)
        hi = points.max(0)
        size = hi - lo
        self.origin = lo - pad * size
        self.size = size * (1.0 + 2.0 * pad)
        self.size[self.size == 0] = 1.0  # guard flat axes

        # Local coordinates (s, t, u) in [0, 1] for each point.
        stu = (points - self.origin) / self.size
        stu = np.clip(stu, 0.0, 1.0)

        # Per-axis Bernstein bases, then outer product -> (N, L*M*N) weights.
        bs = _bernstein(L - 1, stu[:, 0])  # (N, L)
        bt = _bernstein(M - 1, stu[:, 1])  # (N, M)
        bu = _bernstein(N - 1, stu[:, 2])  # (N, N)
        w = bs[:, :, None, None] * bt[:, None, :, None] * bu[:, None, None, :]
        self.weights = w.reshape(points.shape[0], L * M * N)  # (N, P)

        # Base control-point grid (regular lattice). Shape (P, 3).
        gi, gj, gk = np.meshgrid(
            np.linspace(0, 1, L),
            np.linspace(0, 1, M),
            np.linspace(0, 1, N),
            indexing="ij",
        )
        self.control_points = (
            self.origin
            + np.stack([gi, gj, gk], axis=-1).reshape(-1, 3) * self.size
        )
        self.base_points = points

    @property
    def n_control(self) -> int:
        return self.control_points.shape[0]

    def deform(self, delta: np.ndarray) -> np.ndarray:
        """Apply control-point displacements ``delta`` (P, 3) -> new points.

        ``delta`` may also be passed flat (P*3,). Returns (N, 3). Zero delta
        returns the original points to within floating point.
        """
        delta = np.asarray(delta, dtype=np.float64).reshape(self.n_control, 3)
        return self.base_points + self.weights @ delta

    def jacobian(self) -> np.ndarray:
        """d(points)/d(delta): the constant (N, P) weight matrix.

        Because the map is linear in ``delta`` this Jacobian is exactly
        ``self.weights`` and does not depend on the current displacement — which
        is why chaining it with a differentiable surrogate gives clean
        d(Cd)/d(control-point) gradients for the optimisation loop.
        """
        return self.weights
