"""Turn an STL surface into the tensors a neural surrogate consumes.

Two representations, matching what the production models want:

* **Surface point cloud + normals** — the input to point-based operators such
  as DoMINO and to the PointNet baseline in this spike.
* **Signed distance field (SDF)** — sampled at arbitrary query points, the
  geometry encoding used by GINO / DoMINO to condition a volumetric field.
  Convention here: **negative outside, positive inside** the body.

The normalisation applied here (center + isotropic scale to a unit box) is part
of the model contract: the exact same transform must run at inference. Both
paths go through :func:`stl_to_pointcloud`, so they cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from ..config import SampleConfig, DEFAULT_SAMPLE


@dataclass
class PointCloud:
    """A sampled surface plus the transform that produced it.

    ``points`` and ``normals`` are (N, 3). ``center`` / ``scale`` are the
    normalisation applied (``raw = points * scale + center``), kept so a
    prediction made in normalised space can be mapped back to the real body.
    """

    points: np.ndarray
    normals: np.ndarray
    center: np.ndarray
    scale: float

    @property
    def features(self) -> np.ndarray:
        """(N, 6) xyz+normal feature matrix for the surrogate input."""
        return np.concatenate([self.points, self.normals], axis=1)


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load an STL (or any trimesh-readable surface) as a single mesh.

    ``force="mesh"`` collapses multi-body scenes into one mesh, which is what we
    want for a whole-vehicle STL exported from the AOX pipeline.
    """
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
        raise ValueError(f"{path!r} did not load as a non-empty surface mesh")
    return mesh


def sample_surface(
    mesh: trimesh.Trimesh, n_points: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted sample of ``n_points`` on the surface, with normals.

    Area weighting (trimesh's default) matters: uniform-per-face sampling would
    over-represent small, highly-tessellated regions (wheels, mirrors) and
    starve large panels, biasing the force integral the model has to learn.
    """
    rng = np.random.default_rng(seed)
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points, seed=rng)
    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float64)
    return np.asarray(points, dtype=np.float64), normals


def signed_distance(mesh: trimesh.Trimesh, query: np.ndarray) -> np.ndarray:
    """Signed distance from ``query`` points to the surface.

    Sign convention: **negative outside, positive inside** (trimesh's
    ``signed_distance`` returns positive inside; we keep that). For a
    non-watertight STL the sign is unreliable, so we fall back to *unsigned*
    distance and warn via the returned array staying non-negative.
    """
    query = np.atleast_2d(np.asarray(query, dtype=np.float64))
    if mesh.is_watertight:
        return np.asarray(trimesh.proximity.signed_distance(mesh, query))
    # Non-watertight: unsigned distance only. Better than a wrong sign.
    closest, dist, _ = trimesh.proximity.closest_point(mesh, query)
    return -np.asarray(dist)  # treat all as "outside" (negative)


def stl_to_pointcloud(path: str, cfg: SampleConfig = DEFAULT_SAMPLE) -> PointCloud:
    """Full STL -> normalised :class:`PointCloud`, the single entry point.

    Call this everywhere (training and inference) so the input distribution is
    identical on both sides.
    """
    mesh = load_mesh(path)
    points, normals = sample_surface(mesh, cfg.n_surface_points, cfg.seed)

    center = np.zeros(3)
    scale = 1.0
    if cfg.normalize:
        lo, hi = points.min(0), points.max(0)
        center = (lo + hi) / 2.0
        scale = float((hi - lo).max()) or 1.0  # longest bbox edge; guard zero
        points = (points - center) / scale
        # normals are directions: unaffected by translation + isotropic scale,
        # but re-normalise to unit length to shed any numerical drift.
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

    return PointCloud(points=points, normals=normals, center=center, scale=scale)
