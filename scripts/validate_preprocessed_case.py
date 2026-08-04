#!/usr/bin/env python3
"""Validate one preprocessing-v2 NPZ before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aox_g3.geometry.surface_sampling import GEOMETRY_PREPROCESSING_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    args = parser.parse_args()
    required = {
        "geometry_points", "geometry_normals", "surface_points", "surface_cp",
        "geometry_preprocessing_version",
    }
    with np.load(args.npz) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"missing NPZ arrays: {missing}")
        version = int(data["geometry_preprocessing_version"])
        if version != GEOMETRY_PREPROCESSING_VERSION:
            raise ValueError(
                f"preprocessing version {version}; expected "
                f"{GEOMETRY_PREPROCESSING_VERSION}"
            )
        report = {"path": str(args.npz.resolve()), "version": version, "arrays": {}}
        for name in data.files:
            value = np.asarray(data[name])
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
            report["arrays"][name] = list(value.shape)
        points = np.asarray(data["geometry_points"], float)
        normals = np.asarray(data["geometry_normals"], float)
        if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("geometry points/normals must both have shape (N, 3)")
        lengths = np.linalg.norm(normals, axis=1)
        if not np.allclose(lengths, 1.0, atol=1e-5):
            raise ValueError("geometry normals are not unit length")
        report.update({
            "geometry_points": len(points),
            "normalized_bounds": [points.min(0).tolist(), points.max(0).tolist()],
            "normal_length_range": [float(lengths.min()), float(lengths.max())],
            "surface_cp_range": [
                float(np.min(data["surface_cp"])), float(np.max(data["surface_cp"]))
            ],
        })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
