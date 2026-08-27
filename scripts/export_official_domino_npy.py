"""Convert v3 run dirs into the .npy dicts the official DoMINO pipeline reads.

The stock training path is process_data -> cache_data.py -> train.py, and every
stage after the first expects one pickled dict per case with the STL geometry,
the surface mesh, and the surface fields. Our G2 cases already carry all of it
(drivaer_N.stl + boundary_N.vtp + conditions_N.json), so this writes the dict
directly and skips the AWS-specific processor.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv
import trimesh


def convert(run_dir: Path, run: str, out: Path, air_density_ref: float) -> dict:
    mesh = pv.read(run_dir / f"boundary_{run}.vtp").extract_surface().triangulate()
    conditions = json.loads((run_dir / f"conditions_{run}.json").read_text())
    stl = trimesh.load(run_dir / f"drivaer_{run}.stl", force="mesh", process=False)

    sized = mesh.compute_cell_sizes(length=False, volume=False)
    areas = np.asarray(sized.cell_data["Area"], dtype=np.float32)
    centers = np.asarray(mesh.cell_centers().points, dtype=np.float32)
    normals = np.asarray(mesh.cell_normals, dtype=np.float32)

    speed = float(conditions.get("speed") or 30.0)
    density = float(conditions.get("density") or air_density_ref)

    # The pretrained checkpoint's scaling statistics describe a non-dimensional
    # pressure field. G2 writes pMeanTrim in Pa, so divide by rho*U^2 -- measured
    # on 12 cases as the constant that makes the surface integral reproduce SU2's
    # recorded Cd (4% spread). The shear columns already arrive non-dimensional.
    fields = []
    for name in ("pMeanTrim", "wallShearStressMeanTrim"):
        if name not in mesh.cell_data:
            raise ValueError(f"run_{run}: missing {name} in {list(mesh.cell_data)}")
        value = np.asarray(mesh.cell_data[name], dtype=np.float32)
        if name == "pMeanTrim":
            value = value / np.float32(density * speed ** 2)
        fields.append(value.reshape(len(areas), -1))
    surface_fields = np.concatenate(fields, axis=1)

    stl_faces = np.asarray(stl.faces, dtype=np.int32)
    stl_vertices = np.asarray(stl.vertices, dtype=np.float32)
    tri = stl_vertices[stl_faces]
    stl_centers = tri.mean(axis=1).astype(np.float32)
    stl_areas = np.asarray(stl.area_faces, dtype=np.float32)

    payload = {
        "stl_coordinates": stl_vertices,
        "stl_centers": stl_centers,
        "stl_faces": stl_faces.reshape(-1).astype(np.float32),
        "stl_areas": stl_areas,
        "surface_mesh_centers": centers,
        "surface_normals": normals,
        "surface_areas": areas,
        "surface_fields": surface_fields,
        "filename": f"run_{run}",
        "global_params_values": np.array([[speed], [density]], dtype=np.float32),
        "global_params_reference": np.array([[30.0], [air_density_ref]], dtype=np.float32),
    }
    np.save(out / f"run_{run}.npy", payload, allow_pickle=True)
    return {"run": run, "cells": int(len(areas)), "stl_faces": int(len(stl_areas)),
            "fields": int(surface_fields.shape[1]), "speed": speed, "density": density}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", required=True, help="comma list, or 'all'")
    ap.add_argument("--air-density-ref", type=float, default=1.205)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.runs == "all":
        runs = sorted(p.name.split("_", 1)[1] for p in args.root.glob("run_*") if p.is_dir())
    else:
        runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    ok = 0
    for run in runs:
        try:
            info = convert(args.root / f"run_{run}", run, args.out, args.air_density_ref)
            ok += 1
            print(f"run_{run}: cells {info['cells']:,} stl {info['stl_faces']:,} "
                  f"fields {info['fields']} speed {info['speed']}", flush=True)
        except Exception as exc:
            print(f"run_{run}: FAILED {type(exc).__name__}: {exc}", flush=True)
    print(f"converted {ok}/{len(runs)} -> {args.out}")


if __name__ == "__main__":
    main()
