"""Does each hole have a partner boundary facing it, or is it alone?

Distance to the nearest face said almost every remaining hole has geometry right
against it, median 0.00 mm - including the 4.85 m underbody boundary, which would
mean the floor is present and merely unstitched. That reading cannot be trusted on
its own, because this model self-intersects 8,830 times and a face passing straight
through a boundary is also at distance zero without being anything to stitch to.

The signature of an unstitched seam is different and much sharper: both sides are
free. A glass panel that was never joined to the body has its own free boundary
running alongside the body's window opening. A genuine hole has nothing facing it.

So measure boundary-to-boundary: for every sample on a loop, the distance to the
nearest sample on a different loop.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aox_g3 import brep, cad  # noqa: E402

from OCP.BRep import BRep_Tool  # noqa: E402
from OCP.BRepAdaptor import BRepAdaptor_Curve  # noqa: E402
from OCP.TopAbs import TopAbs_EDGE  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--step", type=Path, required=True)
ap.add_argument("--samples", type=int, default=20, help="points per edge")
ap.add_argument("--near", type=float, default=20.0,
                help="a partner this close counts as facing it")
args = ap.parse_args()

from scipy.spatial import cKDTree  # noqa: E402

shape, report = cad.read_step(args.step)
if shape is None:
    raise SystemExit(f"읽기 실패: {report.warnings}")
cad.diagnose(shape, report)
shape, report = cad.sew_progressive(shape, report)
holes, _ = brep.free_boundaries(shape)

points, which = [], []
for index, (boundary, wire) in enumerate(holes):
    for e in brep._explore(wire, TopAbs_EDGE):
        edge = TopoDS.Edge_s(e)
        curve = BRepAdaptor_Curve(edge)
        first, last = BRep_Tool.Range_s(edge)
        for t in np.linspace(first, last, args.samples):
            p = curve.Value(float(t))
            points.append((p.X(), p.Y(), p.Z()))
            which.append(index)
points = np.asarray(points)
which = np.asarray(which)
print(f"구멍 {len(holes)}개, 표본 {len(points):,}개\n")

tree = cKDTree(points)
# Enough neighbours to get past every sample of our own loop
k = min(len(points), 400)
distances, indices = tree.query(points, k=k)

nearest_other = np.full(len(points), np.inf)
for row in range(len(points)):
    mine = which[row]
    others = which[indices[row]] != mine
    if others.any():
        nearest_other[row] = distances[row][others][0]

print(f"{'구멍크기':>9} {'상대까지 중앙':>13} {'가까운 표본 비율':>16} "
      f"{'상대 구멍':>9}   판정")
for index, (boundary, _) in enumerate(holes):
    pick = which == index
    d = nearest_other[pick]
    finite = d[np.isfinite(d)]
    if not len(finite):
        print(f"{boundary.size:9.1f}   상대 없음")
        continue
    med = float(np.median(finite))
    close = float((finite < args.near).mean())
    partner = ""
    near_rows = np.where(pick & (nearest_other < args.near))[0]
    if len(near_rows):
        cand = which[indices[near_rows[0]]]
        cand = cand[cand != index]
        if len(cand):
            partner = f"{holes[int(cand[0])][0].size:.0f}"
    # A loop with most of its length running beside another free loop is one side
    # of a seam; one with none is an opening onto nothing
    verdict = ("미봉합 이음매" if close > 0.8 else
               "일부만 맞닿음" if close > 0.2 else "진짜 구멍")
    print(f"{boundary.size:9.1f} {med:13.2f} {100 * close:15.0f}% "
          f"{partner:>9}   {verdict}")
