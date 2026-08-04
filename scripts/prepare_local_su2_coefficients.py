#!/usr/bin/env python3
"""Convert local SU2-label STL/CSV pairs into G3 v2 coefficient NPZ cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from aox_g3.geometry.stl_sampler import load_mesh, sample_surface
from aox_g3.geometry.surface_sampling import GEOMETRY_PREPROCESSING_VERSION


def mesh_digest(mesh) -> str:
    digest = hashlib.sha256()
    for value in (np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int64)):
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def read_metadata(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    return {key: float(value) for key, value in row.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--geometry-points", type=int, default=20000)
    parser.add_argument("--viscosity", type=float, default=1.7894e-5)
    parser.add_argument("--temperature", type=float, default=288.15)
    parser.add_argument("--require-watertight", action="store_true")
    args = parser.parse_args()

    cases_dir = args.out / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    cases, skipped = [], []
    for run_dir in sorted(args.root.glob("run_*"), key=lambda p: int(p.name.split("_")[-1])):
        run_id = run_dir.name.split("_")[-1]
        stl = run_dir / f"drivaer_{run_id}.stl"
        csv_path = run_dir / f"geo_ref_{run_id}.csv"
        if not stl.is_file() or not csv_path.is_file():
            skipped.append({"run": run_id, "reason": "missing STL or CSV"})
            continue
        try:
            metadata = read_metadata(csv_path)
            cd, cl = metadata["su2_cd"], metadata["su2_cl"]
            if not np.isfinite([cd, cl]).all():
                raise ValueError("non-finite Cd/Cl")
            if metadata["U"] <= 0 or metadata["rho"] <= 0 or metadata["ref_area"] <= 0:
                raise ValueError("non-positive flow metadata")
            if args.require_watertight and int(metadata.get("watertight", 0)) != 1:
                raise ValueError("not watertight")
            mesh = load_mesh(str(stl))
            lo, hi = np.asarray(mesh.bounds, np.float64)
            center = (lo + hi) / 2.0
            scale = float(np.max(hi - lo))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("invalid STL extent")
            points, normals = sample_surface(mesh, args.geometry_points, seed=0)
            points = ((points - center) / scale).astype(np.float32)
            normals = normals.astype(np.float32)
            digest = mesh_digest(mesh)
            case_id = f"su2_labels_v2_run_{int(run_id):03d}"
            npz_path = cases_dir / f"{case_id}.npz"
            np.savez_compressed(
                npz_path,
                geometry_points=points,
                geometry_normals=normals,
                geometry_preprocessing_version=np.asarray(
                    GEOMETRY_PREPROCESSING_VERSION, dtype=np.int64
                ),
            )
            cases.append({
                "case_id": case_id,
                "group_id": f"mesh_{digest[:16]}",
                "npz": str(npz_path.resolve()),
                "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
                "conditions": {
                    "u_x": metadata["U"], "u_y": 0.0, "u_z": 0.0,
                    "density": metadata["rho"], "viscosity": args.viscosity,
                    "temperature": args.temperature, "ref_length": scale,
                    "ref_area": metadata["ref_area"],
                },
                "coefficients": {"cd": cd, "cl": cl},
                "condition_provenance": {
                    "u_x": "geo_ref_csv", "density": "geo_ref_csv",
                    "ref_area": "geo_ref_csv", "ref_length": "stl_max_extent",
                    "viscosity": "configured_default", "temperature": "configured_default",
                },
                "source": {
                    "stl": str(stl.resolve()), "geo_ref": str(csv_path.resolve()),
                    "mesh_digest": digest,
                },
            })
        except Exception as exc:
            skipped.append({"run": run_id, "reason": str(exc)})

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "format": "g3-local-su2-coefficients-v1",
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "source_root": str(args.root.resolve()),
        "assumptions": {
            "viscosity": args.viscosity,
            "temperature": args.temperature,
            "ref_length": "maximum STL bounding-box extent",
        },
        "cases": cases,
        "skipped": skipped,
    }, indent=2) + "\n")
    print(json.dumps({
        "runs": len(list(args.root.glob("run_*"))),
        "accepted": len(cases),
        "unique_meshes": len({row["group_id"] for row in cases}),
        "skipped": len(skipped),
        "manifest": str(manifest),
    }, indent=2))


if __name__ == "__main__":
    main()
