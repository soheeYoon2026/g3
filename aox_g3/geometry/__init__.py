"""Geometry processing: STL -> point cloud / SDF, and FFD deformation."""

from .stl_sampler import (
    load_mesh,
    sample_surface,
    signed_distance,
    stl_to_pointcloud,
    PointCloud,
)
from .ffd import FFD

__all__ = [
    "load_mesh",
    "sample_surface",
    "signed_distance",
    "stl_to_pointcloud",
    "PointCloud",
    "FFD",
]
