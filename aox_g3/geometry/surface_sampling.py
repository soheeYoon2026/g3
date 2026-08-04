"""Shared, deterministic surface sampling for training and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GEOMETRY_PREPROCESSING_VERSION = 2


@dataclass(frozen=True)
class SurfaceSamples:
    points: np.ndarray
    normals: np.ndarray
    face_indices: np.ndarray
    barycentric: np.ndarray


def sample_triangle_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_points: int,
    seed: int = 0,
) -> SurfaceSamples:
    """Sample triangle interiors in proportion to area using face normals."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("surface mesh must contain vertices and triangles")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("face index is outside the vertex array")

    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(double_area) & (double_area > 1e-14)
    if not valid.any():
        raise ValueError("surface mesh has no finite, non-degenerate triangles")
    valid_faces = np.flatnonzero(valid)
    probabilities = double_area[valid] / double_area[valid].sum()

    rng = np.random.default_rng(seed)
    face_indices = rng.choice(valid_faces, size=n_points, replace=True, p=probabilities)
    r1 = np.sqrt(rng.random(n_points))
    r2 = rng.random(n_points)
    barycentric = np.column_stack([
        1.0 - r1,
        r1 * (1.0 - r2),
        r1 * r2,
    ])
    sampled_triangles = triangles[face_indices]
    points = np.einsum("ni,nij->nj", barycentric, sampled_triangles)
    normals = cross[face_indices] / double_area[face_indices, None]
    return SurfaceSamples(points, normals, face_indices, barycentric)


def interpolate_vertex_values(
    values: np.ndarray,
    faces: np.ndarray,
    samples: SurfaceSamples,
) -> np.ndarray:
    """Barycentrically interpolate scalar or vector vertex data at samples."""
    values = np.asarray(values)
    faces = np.asarray(faces, dtype=np.int64)
    triangle_values = values[faces[samples.face_indices]]
    if triangle_values.ndim == 2:
        return np.einsum("ni,ni->n", samples.barycentric, triangle_values)
    return np.einsum("ni,nij->nj", samples.barycentric, triangle_values)


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Center a point set and scale it by its longest bounding-box edge."""
    points = np.asarray(points, dtype=np.float64)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2.0
    scale = float(np.max(hi - lo)) or 1.0
    return (points - center) / scale, center, scale
