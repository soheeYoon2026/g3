"""Close the underbody with a flat floor - by decision, not repair - and wrap.

    flat_floor_wrap.py --in var/runs/cas-a/mesh_full.stl --out var/runs/cas-a-assumed

The rule the user set: the floor is flat. Everything else here follows from it.

  1. Find the underbody opening (the largest boundary loop) and read the height
     of its rocker sections - the loop's edge along the sills, away from the
     bumper skirts that dip lower and the wheel arches that climb higher.
  2. Cut a horizontal section through the whole body at that height. Its outer
     outline is the car's footprint at floor level, tyres included.
  3. Cap that outline with a planar polygon. Tyres below the floor stay below it;
     the wheelhouse above it is closed off - the standard smooth-underbody
     simplification.
  4. Put the floor beside the body and alpha-wrap the pair. The wrap takes the
     outside surface, so the overlapping glass, the floating spoke caps and the
     mirror seam all resolve here at no extra cost.

This is an assumption and the outputs say so in their names. A flat floor moves
the absolute Cd; it is fine for a trend and must be declared when comparing with
a reference that modelled the real underbody.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from aox_g3 import fair  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="src", type=Path, required=True,
                help="full-car mesh (already mirrored), e.g. mesh_full.stl")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--floor-z", type=float, help="override the floor height")
ap.add_argument("--side-fraction", type=float, default=0.35,
                help="loop points beyond this fraction of half-width count as rocker")
ap.add_argument("--no-wrap", action="store_true")
ap.add_argument("--alpha-div", type=float,
                help="wrap alpha = diagonal / this (default tier uses 180 ≈ 29 mm); "
                     "smaller divisor = wider alpha, bridges bigger gaps")
args = ap.parse_args()

import trimesh  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

args.out.mkdir(parents=True, exist_ok=True)
summary = {"input": str(args.src), "assumptions": ["flat floor (plane)",
                                                    "mirrored about y=0"]}
python = sys.executable

mesh = trimesh.load(args.src, force="mesh")
lo, hi = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
width = hi[1] - lo[1]
print(f"입력 삼각형 {len(mesh.faces):,}   치수 {np.round(hi - lo, 0)}")

# ------------------------------------------------------------- 1. floor height
def outline_at(z):
    """Largest closed outline of the horizontal section at height z, or None."""
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return None, None
    planar, to_3d = section.to_2D() if hasattr(section, "to_2D") else section.to_planar()
    polys = [p for p in planar.polygons_full if p.area > 0]
    if not polys:
        return None, to_3d
    merged = unary_union(polys)
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)
    return merged, to_3d


box_area = (hi[0] - lo[0]) * width
# The largest boundary loop is not a safe guide to the floor - on the mirrored
# mesh the underbody loop is not a simple cycle and the biggest simple one turned
# out to be the cabin band at roof height. So scan heights instead: cut sections
# from the bottom up and take the lowest height at which the outline closes into
# the full footprint. Below the rockers the section is just tyre ellipses; at
# rocker height the skin is continuous all the way round and the area jumps.
if args.floor_z is not None:
    floor_z = args.floor_z
    scan = []
else:
    scan = []
    for z in np.arange(lo[2] + 20.0, lo[2] + 0.45 * (hi[2] - lo[2]), 20.0):
        poly, _ = outline_at(float(z))
        scan.append((float(z), float(poly.area) if poly is not None else 0.0))
    # The first height at which the outline closes into a footprint - and stays
    # closed for the next two steps - is the rocker line. On CAS-A the outline
    # closes at 70-190 mm (73-77% of the box), opens again at 250-370 where the
    # wheel arches lead into the open wheelhouses, and closes once more at 430
    # on the door skins. A rule that chased the largest outline picked 410 - a
    # floor 720 mm above the tyres. The floor belongs at the first closure, one
    # step up so it sits on the sill rather than on its very edge.
    closed = [a >= 0.5 * box_area for _, a in scan]
    first = next((i for i in range(len(scan) - 2)
                  if closed[i] and closed[i + 1] and closed[i + 2]), None)
    if first is None:
        raise SystemExit("어느 높이에서도 윤곽이 닫히지 않습니다 — --floor-z 로 지정하세요")
    floor_z = scan[first + 1][0]
    print("높이 훑기 (z → 윤곽 면적 / 상자):")
    for z, a in scan[::max(1, len(scan) // 12)]:
        print(f"   {z:7.0f}  {100 * a / box_area:5.1f}%")
summary["floor_z"] = round(floor_z, 1)
summary["floor_scan"] = [[round(z, 1), round(a / 1e6, 4)] for z, a in scan]
print(f"바닥 높이 z = {floor_z:.0f}  (차체 최저 z {lo[2]:.0f}, 바닥이 그보다 {floor_z - lo[2]:.0f} 위)")

# -------------------------------------------------------------- 2. footprint
footprint, to_3d = outline_at(floor_z)
if footprint is None:
    raise SystemExit("바닥 높이에서 단면이 닫힌 윤곽을 만들지 못했습니다")
outline = footprint.exterior
print(f"바닥 윤곽 면적 {footprint.area / 1e6:.3f} m²  (평면 경계상자 {box_area / 1e6:.3f} m²의 "
      f"{100 * footprint.area / box_area:.0f}%)   둘레 {outline.length / 1000:.1f} m")
summary["floor_area_m2"] = round(footprint.area / 1e6, 3)

# Simplify until the ear clipper can take it, without moving the outline by
# more than a few millimetres
tolerance = 2.0
ring = np.asarray(outline.coords)[:-1]
while len(ring) > fair.EAR_CLIP_MAX_VERTICES - 20 and tolerance < 200:
    ring = np.asarray(Polygon(ring).simplify(tolerance).exterior.coords)[:-1]
    tolerance *= 1.5
print(f"윤곽 정점 {len(ring)}개 (단순화 허용오차 {tolerance / 1.5:.1f} mm)")

# ------------------------------------------------------------------ 3. cap
pts3 = trimesh.transform_points(np.column_stack([ring, np.zeros(len(ring))]), to_3d)
tris = fair.ear_clip(pts3, np.array([0.0, 0.0, 1.0]))
if not tris:
    c = pts3.mean(axis=0)
    pts3 = np.vstack([pts3, c])
    n = len(pts3) - 1
    tris = [(k, (k + 1) % n, n) for k in range(n)]
    print("귀 자르기 실패 — 부채꼴 사용")
floor = trimesh.Trimesh(vertices=pts3, faces=np.asarray(tris), process=False)
if floor.face_normals[:, 2].mean() > 0:
    floor.faces = floor.faces[:, ::-1]      # floor faces down, out of the body
floor.export(args.out / "floor.stl")
print(f"바닥 캡: 삼각형 {len(floor.faces)}  면적 {floor.area / 1e6:.3f} m²")

combined = trimesh.util.concatenate([mesh, floor])
combined_path = args.out / "body_with_floor.stl"
combined.export(combined_path)
summary["combined_triangles"] = int(len(combined.faces))

# ------------------------------------------------------------------ 4. wrap
def run(script, arguments):
    proc = subprocess.run([python, "-u", str(HERE / script), *arguments],
                          capture_output=True, text=True, timeout=3600)
    return re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout + proc.stderr)


if not args.no_wrap:
    t0 = time.time()
    wrapped_path = args.out / "wrapped.stl"
    if args.alpha_div:
        # A wider alpha bridges the gap between tyre and arch lip, which the
        # default (diagonal/180, about 29 mm) cannot: the wheelhouses are open
        # into the body, so above the floor the outside still gets in there
        text = run("wrap_once.py", ["--mesh", str(combined_path),
                                    "--out", str(wrapped_path),
                                    "--alpha-div", str(args.alpha_div)])
        summary["alpha_div"] = args.alpha_div
    else:
        text = run("seal_geometry.py", ["--in", str(combined_path),
                                        "--out", str(wrapped_path),
                                        "--report", str(args.out / "wrap.json")])
    (args.out / "wrap.txt").write_text(text, encoding="utf-8")
    print(f"\n랩 {time.time() - t0:.0f}s")
    for line in text.splitlines():
        if any(k in line for k in ("체적비", "결과", "alpha", "vdb", "성공", "실패", "·")):
            print("  " + line.strip()[:150])
    if wrapped_path.exists():
        wrapped = trimesh.load(wrapped_path, force="mesh")
        vol = float(wrapped.volume) if wrapped.is_watertight else float("nan")
        bbox_vol = float(np.prod(wrapped.extents))
        summary.update({
            "wrapped_triangles": int(len(wrapped.faces)),
            "watertight": bool(wrapped.is_watertight),
            "volume_m3": round(vol / 1e9, 4) if wrapped.is_watertight else None,
            "volume_fraction_of_bbox": round(vol / bbox_vol, 3) if wrapped.is_watertight else None,
        })
        print(f"  수밀 {wrapped.is_watertight}   삼각형 {len(wrapped.faces):,}"
              + (f"   체적 {vol / 1e9:.3f} m³ ({100 * vol / bbox_vol:.0f}% of bbox)"
                 if wrapped.is_watertight else ""))
        area_text = run("frontal_area.py", ["--in", str(wrapped_path)])
        m = re.search(r"전면 면적 \(투영 합집합\) ([\d.]+) m²", area_text)
        summary["frontal_area_m2"] = float(m.group(1)) if m else None
        print(f"  전면 면적 {summary['frontal_area_m2']} m²")

(args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\n출력: {args.out}")
