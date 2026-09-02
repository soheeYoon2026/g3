"""Measure every patch against the hole it claims to fill.

BRepOffsetAPI_MakeFilling returns IsDone for boundaries it has not really solved.
Filling 42 of CAS-A's holes took the model from 15.02 m2 to 191.44 m2 - one car's
worth of surface became twelve - and nothing in the API said so. A patch that
large is not a repair, it is a fold of surface flapping through the body, and in a
CFD model it would silently change the answer.

So the fill needs an acceptance test, and the threshold for it has to come from
measurement rather than taste. A hole whose boundary spans d can be closed by
something of order d^2; anything far past that is the solver having run away. This
prints the ratio per hole so the cut can be placed where the two populations
separate.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aox_g3 import brep, cad  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--step", type=Path, required=True)
ap.add_argument("--max-size", type=float, default=1000.0,
                help="skip holes bigger than this (they take minutes each)")
args = ap.parse_args()

from OCP.BRepGProp import BRepGProp  # noqa: E402
from OCP.GProp import GProp_GProps  # noqa: E402

shape, report = cad.read_step(args.step)
cad.diagnose(shape, report)
shape, report = cad.sew_progressive(shape, report)
holes, _ = brep.free_boundaries(shape)
print(f"구멍 {len(holes)}개, 그 중 {args.max_size:.0f} 이하 "
      f"{sum(1 for h, _ in holes if h.size <= args.max_size)}개를 검사합니다\n")

print(f"{'구멍크기':>9} {'둘레':>9} {'모서리':>6} {'평면':>4} "
      f"{'패치면적 m2':>12} {'크기^2 m2':>10} {'비율':>8} {'방법':>10} {'초':>6}")
rows = []
for boundary, wire in holes:
    if boundary.size > args.max_size:
        continue
    t0 = time.time()
    face = brep.fill_boundary(wire, boundary)
    seconds = time.time() - t0
    if face is None:
        print(f"{boundary.size:9.1f} {boundary.length:9.1f} {boundary.edges:6d} "
              f"{'예' if boundary.planar else '':>4} {'실패':>12} "
              f"{'':>10} {'':>8} {'':>10} {seconds:6.1f}")
        continue
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    area = props.Mass()
    # The natural scale for closing a hole of extent d is d^2; the ratio says how
    # far past that the solver went
    scale = boundary.size ** 2
    ratio = area / scale if scale else float("inf")
    rows.append((boundary.size, ratio, boundary.fill_method))
    print(f"{boundary.size:9.1f} {boundary.length:9.1f} {boundary.edges:6d} "
          f"{'예' if boundary.planar else '':>4} {area / 1e6:12.4f} "
          f"{scale / 1e6:10.4f} {ratio:8.2f} {boundary.fill_method:>10} "
          f"{seconds:6.1f}")

if rows:
    ratios = sorted(r for _, r, _ in rows)
    print(f"\n비율 분포: 최소 {ratios[0]:.2f}  중앙 {ratios[len(ratios) // 2]:.2f}  "
          f"최대 {ratios[-1]:.2f}")
    for cut in (1.0, 2.0, 5.0, 10.0):
        n = sum(1 for r in ratios if r > cut)
        print(f"  {cut:>5.1f} 초과 {n:3d}개 / {len(ratios)}개")
