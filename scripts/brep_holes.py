"""Ask the B-rep where the holes are, instead of hunting for them in a voxel grid.

The mesh path had to infer that CAS-A leaks: flood fill from outside, notice it
reaches every interior cell, conclude there are openings wider than the 21 mm
pitch, and then have no way to say where. In B-rep the same question is a query.
ShapeAnalysis_FreeBounds returns the free boundaries as wires - closed ones are
holes in an otherwise sound shell, open ones are edges that do not even form a
loop - and each carries its own length and position.

Reading and sewing CAS-A costs about ten seconds, so the sewn shape is cached as
a native .brep next to the input and reused.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from OCP.BRep import BRep_Tool, BRep_Builder
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepTools import BRepTools
from OCP.Bnd import Bnd_Box
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.TopAbs import TopAbs_EDGE, TopAbs_WIRE, TopAbs_FACE, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--step", type=Path, required=True)
ap.add_argument("--cache", type=Path, help="where to keep the sewn .brep")
ap.add_argument("--no-sew", action="store_true")
ap.add_argument("--top", type=int, default=25, help="how many holes to list")
args = ap.parse_args()


def load_or_build():
    """Sewing is the expensive part; do it once and cache the result."""
    cache = args.cache or args.step.with_suffix(".sewn.brep")
    if cache.exists() and not args.no_sew:
        shape = TopoDS_Shape()
        BRepTools.Read_s(shape, str(cache), BRep_Builder())
        print(f"캐시된 꿰맨 형상 사용: {cache.name}")
        return shape

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from aox_g3 import cad

    t0 = time.time()
    shape, report = cad.read_step(args.step)
    if shape is None:
        raise SystemExit(f"STEP 읽기 실패: {report.warnings}")
    cad.diagnose(shape, report)
    print(f"읽기 {time.time() - t0:.1f}s   "
          f"면 {report.faces:,}  셸 {report.shells:,}  자유모서리 {report.free_edges:,}")
    if not args.no_sew:
        t0 = time.time()
        shape, report = cad.sew(shape, report)
        print(f"꿰맴 {time.time() - t0:.1f}s   허용오차 {report.sew_tolerance:.2f}")
        BRepTools.Write_s(shape, str(cache))
        print(f"캐시 저장: {cache.name}")
    return shape


def count(shape, kind):
    n, ex = 0, TopExp_Explorer(shape, kind)
    while ex.More():
        n += 1
        ex.Next()
    return n


def wire_metrics(wire):
    """Length of the loop and the size of the opening it bounds.

    Length alone does not say how big the hole is - a long thin slot and a round
    port can measure the same. The bounding box diagonal of the loop is the
    number that decides whether a 21 mm flood fill walks through it.
    """
    length = 0.0
    points = []
    ex = TopExp_Explorer(wire, TopAbs_EDGE)
    while ex.More():
        edge = TopoDS.Edge_s(ex.Current())
        curve = BRepAdaptor_Curve(edge)
        try:
            length += GCPnts_AbscissaPoint.Length_s(curve)
        except Exception:
            pass
        first, last = BRep_Tool.Range_s(edge)
        for t in np.linspace(first, last, 5):
            p = curve.Value(float(t))
            points.append((p.X(), p.Y(), p.Z()))
        ex.Next()
    if not points:
        return length, 0.0, np.zeros(3)
    pts = np.asarray(points)
    extent = pts.max(axis=0) - pts.min(axis=0)
    return length, float(np.linalg.norm(extent)), pts.mean(axis=0)


shape = load_or_build()

box = Bnd_Box()
BRepBndLib.Add_s(shape, box)
lo, hi = box.CornerMin(), box.CornerMax()
diag = float(np.linalg.norm([hi.X() - lo.X(), hi.Y() - lo.Y(), hi.Z() - lo.Z()]))
print(f"\n형상: 면 {count(shape, TopAbs_FACE):,}  셸 {count(shape, TopAbs_SHELL):,}  "
      f"대각선 {diag:.0f}")

t0 = time.time()
# splitclosed=False, splitopen=False: keep each boundary whole rather than cutting
# it at vertices, because a hole is one loop and should be reported as one
finder = ShapeAnalysis_FreeBounds(shape, False, False)
closed = finder.GetClosedWires()
opened = finder.GetOpenWires()
print(f"자유경계 분석 {time.time() - t0:.1f}s")

for label, compound in (("닫힌 자유경계 (= 구멍)", closed),
                        ("열린 자유경계 (= 고리도 못 이룬 모서리)", opened)):
    wires = []
    ex = TopExp_Explorer(compound, TopAbs_WIRE)
    while ex.More():
        wires.append(TopoDS.Wire_s(ex.Current()))
        ex.Next()
    print(f"\n{label}: {len(wires):,}개")
    if not wires:
        continue

    rows = []
    for w in wires:
        length, size, centre = wire_metrics(w)
        rows.append((size, length, centre, w))
    rows.sort(key=lambda r: -r[0])

    sizes = np.array([r[0] for r in rows])
    # 21 mm is the voxel pitch at which the mesh flood fill leaked, so it is the
    # threshold that decides whether a hole explains the leak
    for cut in (5.0, 21.0, 50.0, 200.0):
        print(f"  {cut:>6.0f} 초과: {int((sizes > cut).sum()):,}개")
    print(f"  총 둘레 {sum(r[1] for r in rows) / 1000:.1f} m")

    print(f"  --- 큰 것부터 {min(args.top, len(rows))}개 ---")
    for size, length, centre, _ in rows[:args.top]:
        print(f"    크기 {size:9.1f}  둘레 {length:9.1f}  "
              f"중심 ({centre[0]:8.0f},{centre[1]:7.0f},{centre[2]:7.0f})")
