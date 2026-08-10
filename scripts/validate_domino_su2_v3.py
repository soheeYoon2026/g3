#!/usr/bin/env python
"""Verify flow-aligned DoMINO cases by integrating their stored Cp/Cf fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def integrate_case(run_dir: str | Path) -> dict[str, float]:
    import pyvista as pv

    run_dir = Path(run_dir)
    suffix = run_dir.name.removeprefix("run_")
    conditions = json.loads((run_dir / f"conditions_{suffix}.json").read_text())
    mesh = pv.read(run_dir / f"boundary_{suffix}.vtp").triangulate()
    centers = mesh.cell_centers().points
    areas = mesh.compute_cell_sizes(area=True).cell_data["Area"]
    normals = mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=True,
        auto_orient_normals=False,
        inplace=False,
    ).cell_data["Normals"]
    signed_volume = np.sum(np.sum(centers * normals, axis=1) * areas)
    if signed_volume < 0:
        normals = -normals

    q_kinematic = 0.5 * float(conditions["speed"]) ** 2
    cp = np.asarray(mesh.cell_data["pMeanTrim"]).reshape(-1) / q_kinematic
    cf = np.asarray(mesh.cell_data["wallShearStressMeanTrim"]) / q_kinematic
    ref_area = float(conditions["ref_area"])
    pressure_force = np.sum((-cp[:, None] * normals) * areas[:, None], axis=0) / ref_area
    friction_force = -np.sum(cf * areas[:, None], axis=0) / ref_area
    total = pressure_force + friction_force
    return {
        "cd": float(total[0]),
        "cl": float(total[2]),
        "pressure_cd": float(pressure_force[0]),
        "friction_cd": float(friction_force[0]),
        "su2_cd": float(conditions["su2_cd"]),
        "su2_cl": float(conditions["su2_cl"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    failed = False
    for run in args.runs:
        result = integrate_case(run)
        cd_error = abs(result["cd"] - result["su2_cd"])
        cl_error = abs(result["cl"] - result["su2_cl"])
        result["cd_abs_error"] = cd_error
        result["cl_abs_error"] = cl_error
        print(json.dumps({"run": run, **result}))
        failed |= not np.isfinite(cd_error) or not np.isfinite(cl_error)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
