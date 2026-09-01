"""Which reference area actually produced the stored su2_cd label?

106 of ~660 cases declare a `ref_area` that disagrees with their own bounding-box
frontal area, by 3x to 9x. Reasoning alone cannot say whether the declared value or
the geometry is stale, so this integrates the stored surface field and checks which
divisor reproduces the label.

Whichever wins, `evaluate_domino_v3.py` divides by the declared value, so if the
label used a different one the prediction is wrong by exactly that ratio on those
cases -- which is what was seen on the gate (run_52 went from 65% error to 4% when
rescaled).
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--runs", nargs="+", required=True)
args = ap.parse_args()

print(f"{'run':>6s} {'라벨 Cd':>9s} {'선언면적':>9s} {'실측면적':>9s} "
      f"{'선언으로 적분':>12s} {'실측으로 적분':>12s} {'일치':>10s}")
print("-" * 78)
for run in args.runs:
    run_dir = args.root / f"run_{run}"
    try:
        conditions = json.loads((run_dir / f"conditions_{run}.json").read_text())
        mesh = pv.read(run_dir / f"boundary_{run}.vtp").extract_surface().triangulate()
    except Exception as exc:
        print(f"{run:>6s}  읽기 실패 {type(exc).__name__}")
        continue

    label = conditions.get("su2_cd")
    if label is None or not np.isfinite(float(label)):
        print(f"{run:>6s}  라벨 없음")
        continue
    declared = float(conditions["ref_area"])
    span = np.sort(np.diff(np.asarray(mesh.bounds).reshape(3, 2), axis=1).ravel())[::-1]
    measured = float(span[1] * span[2])

    sized = mesh.compute_cell_sizes(length=False, volume=False)
    areas = np.asarray(sized.cell_data["Area"], dtype=np.float64)
    normals = np.asarray(mesh.cell_normals, dtype=np.float64)
    centres = np.asarray(mesh.cell_centers().points, dtype=np.float64)
    if np.sum(np.sum(centres * normals, axis=1) * areas) < 0:
        normals = -normals

    speed = float(conditions["speed"])
    # pMeanTrim is stored as Cp * 0.5 U^2; divide it back out to get the coefficient
    cp = np.asarray(mesh.cell_data["pMeanTrim"], dtype=np.float64) / (0.5 * speed ** 2)
    cf = np.asarray(mesh.cell_data["wallShearStressMeanTrim"], dtype=np.float64) / (0.5 * speed ** 2)
    force_x = float(np.sum(-cp * normals[:, 0] * areas) - np.sum(cf[:, 0] * areas))

    with_declared = force_x / declared
    with_measured = force_x / measured
    err_d = abs(with_declared - label) / abs(label)
    err_m = abs(with_measured - label) / abs(label)
    winner = "선언" if err_d < err_m else "실측"
    print(f"{run:>6s} {label:9.4f} {declared:9.3f} {measured:9.3f} "
          f"{with_declared:9.4f}({100*err_d:3.0f}%) {with_measured:9.4f}({100*err_m:3.0f}%) "
          f"{winner:>10s}")
