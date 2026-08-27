"""Find the field normalization that makes the evaluator's integral reproduce SU2's Cd.

evaluate_domino_v3.integrate_prediction computes, from the model's field f:
    Cd = 2*rho * ( sum(-f_p * n_x * A) - sum(f_fx * A) ) / ref_area
So the training targets must be the raw Pa field divided by whatever constant N
makes that identity hold against the recorded su2_cd. This measures N per case and
checks which closed form (rho^2 U^2, 2*rho*q, q, ...) it matches.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--roots", type=Path, nargs="+", required=True, help="v3 dataset roots")
ap.add_argument("--per-root", type=int, default=6, help="cases to sample per root")
args = ap.parse_args()

rows = []
for root in args.roots:
    root = Path(root)
    for run in sorted(p for p in root.glob("run_*") if p.is_dir())[:args.per_root]:
        tag = run.name.split("_", 1)[1]
        try:
            conditions = json.loads((run / f"conditions_{tag}.json").read_text())
            mesh = pv.read(run / f"boundary_{tag}.vtp").extract_surface().triangulate()
        except Exception as exc:
            print(f"{root.name}/{run.name}: skip ({type(exc).__name__})")
            continue

        sized = mesh.compute_cell_sizes(length=False, volume=False)
        areas = np.asarray(sized.cell_data["Area"], dtype=np.float64)
        normals = np.asarray(mesh.cell_normals, dtype=np.float64)
        centers = np.asarray(mesh.cell_centers().points, dtype=np.float64)
        if np.sum(np.sum(centers * normals, axis=1) * areas) < 0:
            normals = -normals

        pressure = np.asarray(mesh.cell_data["pMeanTrim"], dtype=np.float64)
        shear = np.asarray(mesh.cell_data["wallShearStressMeanTrim"], dtype=np.float64)
        ref_area = float(conditions["ref_area"])
        rho = float(conditions["density"])
        speed = float(conditions["speed"])
        q = 0.5 * rho * speed ** 2

        # drag-direction integrals of the RAW Pa field, before any normalization
        drag_p = float(np.sum(-pressure * normals[:, 0] * areas))
        drag_f = float(np.sum(-shear[:, 0] * areas))
        cd_true = float(conditions["su2_cd"])

        # evaluator applies: Cd = 2*rho*(drag_p + drag_f)/N/ref_area  -> solve N
        n_solved = 2.0 * rho * (drag_p + drag_f) / (cd_true * ref_area)
        rows.append({
            "case": f"{root.name[7:20]}/{run.name}", "speed": speed, "rho": rho, "q": q,
            "cd_true": cd_true, "N": n_solved,
            "N/q": n_solved / q, "N/(rho^2 U^2)": n_solved / (rho ** 2 * speed ** 2),
            "N/(2*rho*q)": n_solved / (2 * rho * q),
            "friction_share": drag_f / (drag_p + drag_f) if (drag_p + drag_f) else float("nan"),
        })

print(f"{'case':28s} {'speed':>7s} {'cd_true':>8s} {'N':>12s} {'N/q':>8s} "
      f"{'N/(r2U2)':>9s} {'N/(2rq)':>8s} {'fric%':>7s}")
for r in rows:
    print(f"{r['case']:28s} {r['speed']:7.1f} {r['cd_true']:8.4f} {r['N']:12.2f} "
          f"{r['N/q']:8.3f} {r['N/(rho^2 U^2)']:9.3f} {r['N/(2*rho*q)']:8.3f} "
          f"{100*r['friction_share']:6.1f}%")

for key in ("N/q", "N/(rho^2 U^2)", "N/(2*rho*q)"):
    vals = np.array([r[key] for r in rows])
    print(f"{key:16s} mean {vals.mean():8.3f}  spread(max/min) {vals.max()/vals.min():6.2f}")
