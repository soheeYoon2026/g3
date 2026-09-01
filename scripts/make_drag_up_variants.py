"""Create deformations that should INCREASE drag, on a chosen base car.

The benchmark is short of exactly this: the 400-pair pool holds three such pairs
because those pairs come from optimisation runs, which by construction reduce
drag. Three deformations, each attacking the wake in a way that is aerodynamically
unambiguous:

  blunt_tail   - pull the rear face back toward vertical, enlarging the base area
  wide_rear    - flare the rear quarters outward, widening the wake
  raise_roof   - lift the rear roofline, delaying the downwash and thickening the wake

Each is a smooth radial-basis displacement so the surface stays closed and the
mesh stays valid; magnitudes are set well above the 4-level LES noise floor.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv
import trimesh


def bump(points, centre, radius, direction, amplitude):
    """Smooth compact displacement: cos^2 falloff, exactly zero past the radius."""
    delta = points - centre
    distance = np.linalg.norm(delta, axis=1)
    weight = np.zeros(len(points))
    inside = distance < radius
    weight[inside] = np.cos(0.5 * np.pi * distance[inside] / radius) ** 2
    return weight[:, None] * (np.asarray(direction, dtype=float) * amplitude)


def variants_for(points, bounds):
    lo, hi = bounds[:, 0], bounds[:, 1]
    length, width, height = hi - lo
    # the car's own frame, so the same recipe works whatever the export convention
    rear_x = lo[0]
    mid_y = 0.5 * (lo[1] + hi[1])
    roof_z = hi[2]
    ground_z = lo[2]

    return {
        # square off the tail: push the rear face backwards over a broad patch
        "blunt_tail": [
            dict(centre=[rear_x, mid_y, ground_z + 0.45 * height],
                 radius=0.30 * length, direction=[-1, 0, 0], amplitude=0.055),
        ],
        # flare both rear quarters outward, widening the wake
        "wide_rear": [
            dict(centre=[rear_x + 0.18 * length, lo[1], ground_z + 0.40 * height],
                 radius=0.28 * length, direction=[0, -1, 0], amplitude=0.045),
            dict(centre=[rear_x + 0.18 * length, hi[1], ground_z + 0.40 * height],
                 radius=0.28 * length, direction=[0, 1, 0], amplitude=0.045),
        ],
        # lift the rear roofline so the flow separates later and the wake thickens
        "raise_roof": [
            dict(centre=[rear_x + 0.30 * length, mid_y, roof_z],
                 radius=0.32 * length, direction=[0, 0, 1], amplitude=0.050),
        ],
    }


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--vtp", type=Path, required=True)
ap.add_argument("--out-dir", type=Path, required=True)
ap.add_argument("--name", required=True)
args = ap.parse_args()

mesh = pv.read(args.vtp).extract_surface().triangulate().clean()
points = np.asarray(mesh.points, dtype=np.float64)
bounds = np.asarray(mesh.bounds).reshape(3, 2)
faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
args.out_dir.mkdir(parents=True, exist_ok=True)

base = trimesh.Trimesh(vertices=points, faces=faces, process=False)
print(f"{args.name} 기준: 정점 {len(points):,} 수밀 {base.is_watertight} "
      f"치수 {np.round(base.extents, 3).tolist()} 체적 {base.volume:.3f}")
base.export(args.out_dir / f"{args.name}_base.stl")

report = [{"variant": "base", "max_disp_mm": 0.0, "watertight": bool(base.is_watertight),
           "volume": round(float(base.volume), 4),
           "frontal_m2": round(float((bounds[1, 1] - bounds[1, 0]) *
                                     (bounds[2, 1] - bounds[2, 0])), 4)}]

for name, bumps in variants_for(points, bounds).items():
    moved = points.copy()
    for spec in bumps:
        moved += bump(points, np.asarray(spec["centre"], dtype=float),
                      spec["radius"], spec["direction"], spec["amplitude"])
    tri = trimesh.Trimesh(vertices=moved, faces=faces, process=False)
    disp = np.linalg.norm(moved - points, axis=1)
    new_bounds = np.array([[moved[:, k].min(), moved[:, k].max()] for k in range(3)])
    frontal = (new_bounds[1, 1] - new_bounds[1, 0]) * (new_bounds[2, 1] - new_bounds[2, 0])
    ok = tri.is_watertight and int((tri.area_faces <= 0).sum()) == 0
    print(f"  {name}: 최대변위 {1000*disp.max():5.1f}mm  이동정점 {int((disp>1e-6).sum()):,}  "
          f"수밀 {tri.is_watertight}  체적 {tri.volume:.3f}  정면적 {frontal:.4f}")
    if ok:
        tri.export(args.out_dir / f"{args.name}_{name}.stl")
    report.append({"variant": name, "max_disp_mm": round(1000 * float(disp.max()), 2),
                   "moved_vertices": int((disp > 1e-6).sum()),
                   "watertight": bool(tri.is_watertight),
                   "volume": round(float(tri.volume), 4),
                   "frontal_m2": round(float(frontal), 4), "exported": bool(ok)})

(args.out_dir / f"{args.name}_report.json").write_text(json.dumps(report, indent=1) + "\n")
print(f"\n내보낸 STL: {len(list(args.out_dir.glob(f'{args.name}_*.stl')))}개 -> {args.out_dir}")
