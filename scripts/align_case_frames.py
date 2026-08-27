"""Translate each case into the frame the pretrained encoder expects.

geo_rep_surface normalizes against a fixed DrivAerML box and samples an SDF over
[-1, 1]; measured on this dataset only 12.6% of vertices land inside it, and most
cases sit entirely outside in z. Off the grid the encoder returns essentially the
same thing for any shape, which is why deformations move its output by 0.2% of
what separates different cars.

Translation only -- no rotation, no scaling -- so forces, areas and therefore the
coefficients are untouched.

The anchor matters: aligning on a bounding-box extremity would absorb part of the
deformation, since a 4 cm move of the tail shifts the box by 4 cm. Anchoring on
the area-weighted centroid (in x, y) moves only by the deformed fraction of the
surface, and on the ground plane (z), which deformations here do not touch.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv
import trimesh

# where a DrivAerML car sits, from the pretrained checkpoint's own statistics
TARGET_CENTROID_XY = (1.562, 0.0)
TARGET_GROUND_Z = -0.32


def surface_anchor(mesh):
    sized = mesh.compute_cell_sizes(length=False, volume=False)
    areas = np.asarray(sized.cell_data["Area"], dtype=np.float64)
    centres = np.asarray(mesh.cell_centers().points, dtype=np.float64)
    centroid = (centres * areas[:, None]).sum(axis=0) / areas.sum()
    ground = float(np.asarray(mesh.points)[:, 2].min())
    return centroid, ground


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--write", action="store_true")
args = ap.parse_args()

run_dirs = sorted((p for p in args.root.glob("run_*") if p.is_dir()),
                  key=lambda p: int(p.name.split("_")[1]))
shifts = []
for run_dir in run_dirs:
    tag = run_dir.name.split("_", 1)[1]
    vtp_path = run_dir / f"boundary_{tag}.vtp"
    mesh = pv.read(vtp_path).extract_surface().triangulate()
    centroid, ground = surface_anchor(mesh)
    shift = np.array([TARGET_CENTROID_XY[0] - centroid[0],
                      TARGET_CENTROID_XY[1] - centroid[1],
                      TARGET_GROUND_Z - ground])
    shifts.append(shift)
    if not args.write:
        continue

    mesh.points = np.asarray(mesh.points, dtype=np.float64) + shift
    mesh.save(vtp_path)

    stl_path = run_dir / f"drivaer_{tag}.stl"
    if stl_path.exists():
        stl = trimesh.load(stl_path, force="mesh", process=False)
        stl.vertices = np.asarray(stl.vertices, dtype=np.float64) + shift
        stl.export(stl_path)

    conditions_path = run_dir / f"conditions_{tag}.json"
    conditions = json.loads(conditions_path.read_text())
    conditions["frame_shift"] = shift.tolist()
    conditions_path.write_text(json.dumps(conditions, indent=2) + "\n")

shifts = np.array(shifts)
print(f"런 {len(shifts)}개")
print(f"이동량 중앙값 {np.round(np.median(shifts, axis=0), 3).tolist()} m")
print(f"이동량 범위 x {shifts[:,0].min():+.2f}~{shifts[:,0].max():+.2f}  "
      f"y {shifts[:,1].min():+.2f}~{shifts[:,1].max():+.2f}  "
      f"z {shifts[:,2].min():+.2f}~{shifts[:,2].max():+.2f}")
print("→ z 이동량이 크게 갈리면 케이스마다 좌표계가 달랐다는 뜻")
if args.write:
    print("boundary/stl/conditions 갱신함")
else:
    print("(--write 없이 실행: 계산만 하고 파일은 건드리지 않음)")
