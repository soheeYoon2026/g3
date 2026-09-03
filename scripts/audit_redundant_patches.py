"""Did any patch get laid over a surface that was already there?

A free boundary does not always bound an opening. Styling CAD models panels
oversized and overlapping, trimming them later, so a panel laid on top of another
has a free loop all the way round it with solid surface underneath. CAS-A's rear
side glass is exactly that: the loop is 4 mm above a continuous panel, and looking
through it shows body, not sky.

Capping such a loop is worse than leaving it. It welds a second skin a few
millimetres above the first, which no picture will show and which gives the mesher
two surfaces where the flow sees one. The right treatment is the intersection step,
not the filling step.

So for each hole, build the patch that would be used and ask what is behind it:
sample the patch and measure the distance to every face that is not part of the
hole's own boundary. Close means there was already something there.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aox_g3 import brep, cad, topology  # noqa: E402

from OCP.TopAbs import TopAbs_EDGE  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--step", type=Path, required=True)
ap.add_argument("--seal-below", type=float, default=900.0)
args = ap.parse_args()

import trimesh  # noqa: E402

shape, report = cad.read_step(args.step)
if shape is None:
    raise SystemExit(f"읽기 실패: {report.warnings}")
cad.diagnose(shape, report)
shape, report = cad.sew_progressive(shape, report)

mesh, owner, faces = topology.tessellate_with_owner(shape)
index_of = {face: i for i, face in enumerate(faces)}
supports = brep.face_supports(shape)
holes, _ = brep.free_boundaries(shape)
print(f"삼각형 {len(mesh.faces):,}   구멍 {len(holes)}개\n")

print(f"{'구멍크기':>9} {'패치':>10} {'뒤쪽 표면까지':>13} {'/크기':>8} "
      f"{'덮인비율':>8}   판정")
counts = {"덮어씀": 0, "진짜 구멍": 0, "패치 없음": 0}
for boundary, wire in holes:
    if boundary.size > args.seal_below:
        continue
    patch = brep.fill_boundary(wire, boundary, supports)
    if patch is None:
        print(f"{boundary.size:9.1f} {'실패':>10}")
        counts["패치 없음"] += 1
        continue

    own = set()
    for e in brep._explore(wire, TopAbs_EDGE):
        edge = TopoDS.Edge_s(e)
        if supports.Contains(edge):
            for f in supports.FindFromKey(edge):
                idx = index_of.get(TopoDS.Face_s(f))
                if idx is not None:
                    own.add(idx)
    keep = ~np.isin(owner, list(own)) if own else np.ones(len(owner), bool)
    if not keep.any():
        continue
    other = trimesh.Trimesh(vertices=mesh.vertices,
                            faces=np.asarray(mesh.faces)[keep], process=False)

    # Sample the patch itself: its interior is where a second skin would sit
    patch_mesh = topology.tessellate_by_curvature(patch, 20.0)
    if len(patch_mesh.faces) == 0:
        continue
    points = np.asarray(patch_mesh.triangles_center, dtype=float)
    if len(points) > 400:
        points = points[np.linspace(0, len(points) - 1, 400).astype(int)]
    _, distance, _ = trimesh.proximity.closest_point(other, points)

    med = float(np.median(distance))
    ratio = med / boundary.size if boundary.size else float("inf")
    # The median is a weak discriminator: a genuine hole also has surface a few
    # millimetres away near its rim, so the two populations nearly touch. What
    # separates them is how much of the patch has something underneath it. An
    # overlapping panel is backed everywhere; a real opening only at its edge.
    covered = float((distance < 5.0).mean())
    redundant = covered > 0.5
    counts["덮어씀" if redundant else "진짜 구멍"] += 1
    print(f"{boundary.size:9.1f} {boundary.fill_method:>10} {med:13.2f} "
          f"{ratio:8.4f} {100 * covered:7.0f}%   "
          f"{'덮어씀 — 뒤에 면이 있음' if redundant else '진짜 구멍'}")

print("\n합계:", {k: v for k, v in counts.items() if v})
