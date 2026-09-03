"""Why does the healed shape re-read with 26 free boundaries when it left with 8?

Three suspects, and they can be separated in one process:

  memory        the healed shape as heal() returned it            (baseline)
  STEP, no pcurves   what heal_step.py writes by default
  STEP, pcurves      same file with the parametric curves kept
  BREP               OCC's native format, lossless

If BREP round-trips at 8 and STEP does not, the format or its reader is losing the
seams. If the pcurve variant round-trips and the default does not, the missing
parametric curves are the cause - a cap edge merged with a body edge at 10 mm
tolerance has a 3D curve that does not lie on the planar cap, and without a stored
pcurve the reader has to rebuild that relationship and may refuse.

For every extra boundary the re-read produces, report its size, where it sits, and
how far it is from the nearest patch that was added - if they all hug the caps, it
is the cap seams and nothing older.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aox_g3 import brep, cad  # noqa: E402

from OCP.BRep import BRep_Builder, BRep_Tool  # noqa: E402
from OCP.BRepTools import BRepTools  # noqa: E402
from OCP.TopAbs import TopAbs_EDGE  # noqa: E402
from OCP.TopoDS import TopoDS, TopoDS_Shape  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--step", type=Path, required=True)
ap.add_argument("--seal-below", type=float, default=900.0)
ap.add_argument("--close-near", action="append", default=[])
args = ap.parse_args()

close_near = [tuple(float(v) for v in s.split(",")) for s in args.close_near]

shape, report = cad.read_step(args.step)
cad.diagnose(shape, report)
shape, report = cad.sew_progressive(shape, report)
healed, heal_report = brep.heal(shape, sealing_size=args.seal_below,
                                close_near=close_near)
patch_centres = np.array([b.centre for b in heal_report.filled], dtype=float)
print(f"봉합: 메움 {heal_report.boundaries_filled}  남김 {heal_report.boundaries_left}")


def boundaries(s):
    holes, _ = brep.free_boundaries(s)
    return [(b.size, np.asarray(b.centre, dtype=float), b.length) for b, _ in holes]


def edge_tolerances(s):
    tols = []
    for e in brep._explore(s, TopAbs_EDGE):
        tols.append(BRep_Tool.Tolerance_s(TopoDS.Edge_s(e)))
    tols = np.asarray(tols)
    return tols


base = boundaries(healed)
tols = edge_tolerances(healed)
print(f"\n[메모리]  자유경계 {len(base)}개   모서리 허용오차: 중앙 {np.median(tols):.3f}  "
      f"최대 {tols.max():.2f}  10 mm 초과 {int((tols > 10).sum())}개  "
      f"1 mm 초과 {int((tols > 1).sum())}개")

tmp = Path(tempfile.mkdtemp())
variants = []

for label, pcurves in (("STEP pcurve 없음", False), ("STEP pcurve 있음", True)):
    path = tmp / f"rt_{int(pcurves)}.stp"
    brep.write_step(healed, path, pcurves=pcurves)
    t0 = time.time()
    back, _ = cad.read_step(path)
    variants.append((label, back, time.time() - t0, path.stat().st_size / 1e6))

path = tmp / "rt.brep"
BRepTools.Write_s(healed, str(path))
back = TopoDS_Shape()
t0 = time.time()
BRepTools.Read_s(back, str(path), BRep_Builder())
variants.append(("BREP (무손실)", back, time.time() - t0, path.stat().st_size / 1e6))


def match(new, ref, tol=60.0):
    """Boundaries in `new` that have no counterpart in `ref` by centre and size."""
    extra = []
    for size, centre, length in new:
        ok = any(np.linalg.norm(centre - c) < tol and
                 abs(size - s) < 0.25 * max(size, s, 1.0) for s, c, _ in ref)
        if not ok:
            extra.append((size, centre, length))
    return extra


for label, back, seconds, mb in variants:
    got = boundaries(back)
    t = edge_tolerances(back)
    extra = match(got, base)
    print(f"\n[{label}]  {mb:.1f} MB  되읽기 {seconds:.1f}s   자유경계 {len(got)}개  "
          f"(새로 생긴 것 {len(extra)}개)   모서리 허용오차 최대 {t.max():.2f}  "
          f"10 mm 초과 {int((t > 10).sum())}개")
    if extra and len(patch_centres):
        near = 0
        rows = []
        for size, centre, length in sorted(extra, key=lambda r: -r[0])[:10]:
            d = float(np.linalg.norm(patch_centres - centre, axis=1).min())
            rows.append((size, length, centre, d))
        near = sum(1 for s, c, l in extra
                   if np.linalg.norm(patch_centres - c, axis=1).min() < s + 50)
        print(f"   새 경계 중 패치 바로 옆(패치 중심까지 < 자기크기+50) {near}/{len(extra)}개")
        print(f"   {'크기':>8} {'둘레':>8} {'중심':>26} {'최근접 패치까지':>12}")
        for size, length, centre, d in rows:
            print(f"   {size:8.1f} {length:8.1f} "
                  f"({centre[0]:7.0f},{centre[1]:6.0f},{centre[2]:6.0f}) {d:12.1f}")
