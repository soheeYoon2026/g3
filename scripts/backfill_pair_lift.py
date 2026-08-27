"""Fill in the lift coefficient by integrating the stored surface field.

The pair manifest resolved drag only, and the per-design history.csv is the
adjoint history with no CD column, so lift has to come from the field itself.
Integrating drag the same way doubles as a check on the whole fetch: it should
land close to the manifest's value.
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
ap.add_argument("--write", action="store_true", help="update conditions files")
args = ap.parse_args()

rows = []
run_dirs = [p for p in args.root.glob("run_*") if p.is_dir()]
for run_dir in sorted(run_dirs, key=lambda p: int(p.name.split("_")[1])):
    tag = run_dir.name.split("_", 1)[1]
    conditions_path = run_dir / f"conditions_{tag}.json"
    conditions = json.loads(conditions_path.read_text())

    mesh = pv.read(run_dir / f"boundary_{tag}.vtp").extract_surface().triangulate()
    sized = mesh.compute_cell_sizes(length=False, volume=False)
    areas = np.asarray(sized.cell_data["Area"], dtype=np.float64)
    normals = np.asarray(mesh.cell_normals, dtype=np.float64)
    centres = np.asarray(mesh.cell_centers().points, dtype=np.float64)
    if np.sum(np.sum(centres * normals, axis=1) * areas) < 0:
        normals = -normals

    speed = float(conditions["speed"])
    ref_area = float(conditions["ref_area"])
    # stored as pMeanTrim = Cp * 0.5 U^2 (kinematic), so divide it back out
    cp = np.asarray(mesh.cell_data["pMeanTrim"], dtype=np.float64) / (0.5 * speed ** 2)
    cf = np.asarray(mesh.cell_data["wallShearStressMeanTrim"], dtype=np.float64) / (0.5 * speed ** 2)
    force = (np.sum(-cp[:, None] * normals * areas[:, None], axis=0)
             - np.sum(cf * areas[:, None], axis=0)) / ref_area

    integrated_cd, integrated_cl = float(force[0]), float(force[2])
    rows.append((run_dir.name, conditions.get("su2_cd"), integrated_cd, integrated_cl))
    if args.write:
        conditions["su2_cl"] = integrated_cl
        conditions["integrated_cd"] = integrated_cd
        conditions_path.write_text(json.dumps(conditions, indent=2) + "\n")

manifest_cd = np.array([r[1] for r in rows], dtype=float)
integrated = np.array([r[2] for r in rows], dtype=float)
ratio = integrated / manifest_cd
print(f"런 {len(rows)}개")
print(f"매니페스트 Cd  {manifest_cd.min():.4f} ~ {manifest_cd.max():.4f}")
print(f"적분 Cd        {integrated.min():.4f} ~ {integrated.max():.4f}")
print(f"적분/매니페스트 비: 중앙값 {np.median(ratio):.3f}  "
      f"사분위 {np.percentile(ratio,25):.3f}/{np.percentile(ratio,75):.3f}")
print(f"상대차 중앙값 {100*np.median(np.abs(ratio-1)):.1f}%")
cl = np.array([r[3] for r in rows])
print(f"적분 Cl        {cl.min():+.4f} ~ {cl.max():+.4f} (중앙값 {np.median(cl):+.4f})")
if args.write:
    print("conditions 파일 갱신함")
