"""For each remaining hole: is there geometry on the other side, or nothing?

The two cases need opposite treatments and look identical in a hole list. A window
whose glass is present but never stitched to the body is a sewing problem - the
surface exists, it is just a few millimetres away and topologically unrelated, and
capping it with a patch would put a second skin over the glass. A window with no
glass at all is a modelling gap and does need a patch.

Telling them apart is a distance question: sample the boundary, and measure how far
it is to the nearest face that is not one of the faces the boundary already belongs
to. Small means the neighbour is right there and the fix is to join them; large
means there is nothing there.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aox_g3 import brep, cad, topology  # noqa: E402

from OCP.BRep import BRep_Tool  # noqa: E402
from OCP.BRepAdaptor import BRepAdaptor_Curve  # noqa: E402
from OCP.TopAbs import TopAbs_EDGE  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--step", type=Path, required=True)
ap.add_argument("--max-size", type=float, default=1e9,
                help="only look at holes up to this size")
ap.add_argument("--samples", type=int, default=12, help="points per edge")
args = ap.parse_args()

import trimesh  # noqa: E402

shape, report = cad.read_step(args.step)
if shape is None:
    raise SystemExit(f"읽기 실패: {report.warnings}")
cad.diagnose(shape, report)
shape, report = cad.sew_progressive(shape, report)

mesh, owner, faces = topology.tessellate_with_owner(shape)
supports = brep.face_supports(shape)
index_of = {}
for i, face in enumerate(faces):
    index_of[face] = i
print(f"삼각형 {len(mesh.faces):,}   B-rep 면 {len(faces):,}")

holes, _ = brep.free_boundaries(shape)
print(f"구멍 {len(holes)}개\n")

print(f"{'구멍크기':>9} {'최근접면까지 중앙':>16} {'최소':>9} {'최대':>9} "
      f"{'/크기':>8}   판정")
for boundary, wire in holes:
    if boundary.size > args.max_size:
        continue

    own = set()
    points = []
    for e in brep._explore(wire, TopAbs_EDGE):
        edge = TopoDS.Edge_s(e)
        if supports.Contains(edge):
            for f in supports.FindFromKey(edge):
                idx = index_of.get(TopoDS.Face_s(f))
                if idx is not None:
                    own.add(idx)
        curve = BRepAdaptor_Curve(edge)
        first, last = BRep_Tool.Range_s(edge)
        for t in np.linspace(first, last, args.samples):
            p = curve.Value(float(t))
            points.append((p.X(), p.Y(), p.Z()))
    if not points:
        continue
    points = np.asarray(points)

    # Everything except the faces this boundary is already part of; without that
    # exclusion the nearest surface is always the boundary's own face at zero
    keep = ~np.isin(owner, list(own)) if own else np.ones(len(owner), bool)
    if not keep.any():
        print(f"{boundary.size:9.1f}   비교할 면이 없습니다")
        continue
    other = trimesh.Trimesh(vertices=mesh.vertices,
                            faces=np.asarray(mesh.faces)[keep], process=False)
    _, distance, _ = trimesh.proximity.closest_point(other, points)

    med = float(np.median(distance))
    ratio = med / boundary.size if boundary.size else float("inf")
    # A neighbour within a few percent of the hole's own size is a surface that is
    # there and unstitched; further than that and there is nothing to stitch to
    verdict = "미봉합 — 옆에 면이 있습니다" if ratio < 0.05 else \
        ("애매" if ratio < 0.15 else "결손 — 채워야 합니다")
    print(f"{boundary.size:9.1f} {med:16.2f} {distance.min():9.2f} "
          f"{distance.max():9.2f} {ratio:8.4f}   {verdict}")
