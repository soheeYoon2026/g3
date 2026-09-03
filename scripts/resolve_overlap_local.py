"""Resolve one overlapping panel, locally, before trying it on a whole car.

The glass is present and lying on the body rather than trimmed into it, so what it
needs is the intersection step: cut both surfaces along the curve where they cross,
then drop the piece that is buried. General Fuse does the cutting.

Doing that to 2,847 faces at once is not something to attempt blind - the
self-intersection check alone takes 472 s on this model. So take the faces around
one opening, fuse those, and see whether the crossing becomes a real edge and how
long it costs. If it works here it can be scaled up or applied per region; if it
does not, nothing has been wasted.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aox_g3 import brep, cad, topology  # noqa: E402

from OCP.Bnd import Bnd_Box  # noqa: E402
from OCP.BRep import BRep_Builder  # noqa: E402
from OCP.BRepBndLib import BRepBndLib  # noqa: E402
from OCP.BRepGProp import BRepGProp  # noqa: E402
from OCP.GProp import GProp_GProps  # noqa: E402
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE  # noqa: E402
from OCP.TopoDS import TopoDS, TopoDS_Compound  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--step", type=Path, required=True)
ap.add_argument("--hole", type=int, default=2,
                help="which hole to work around, 1 = largest")
ap.add_argument("--margin", type=float, default=0.6,
                help="how far past the hole to take faces, as a fraction of its size")
args = ap.parse_args()


def area(shape):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    return float(props.Mass())


shape, report = cad.read_step(args.step)
if shape is None:
    raise SystemExit(f"읽기 실패: {report.warnings}")
cad.diagnose(shape, report)
shape, report = cad.sew_progressive(shape, report)
holes, _ = brep.free_boundaries(shape)
boundary, wire = holes[min(args.hole, len(holes)) - 1]
print(f"대상 구멍: 크기 {boundary.size:.0f}  중심 "
      f"{np.round(np.asarray(boundary.centre), 0)}")

lo = np.asarray(boundary.bbox[0], dtype=float) - args.margin * boundary.size
hi = np.asarray(boundary.bbox[1], dtype=float) + args.margin * boundary.size

builder = BRep_Builder()
region = TopoDS_Compound()
builder.MakeCompound(region)
picked = 0
for f in topology._explore(shape, TopAbs_FACE):
    box = Bnd_Box()
    BRepBndLib.Add_s(f, box)
    if box.IsVoid():
        continue
    flo = np.array([box.CornerMin().X(), box.CornerMin().Y(), box.CornerMin().Z()])
    fhi = np.array([box.CornerMax().X(), box.CornerMax().Y(), box.CornerMax().Z()])
    if np.all(fhi >= lo) and np.all(flo <= hi):
        builder.Add(region, f)
        picked += 1

print(f"영역 면 {picked}개  표면적 {area(region) / 1e6:.4f} m2")
if picked < 2:
    raise SystemExit("면이 부족합니다")

before = brep.free_boundaries(region)[0]
edges_before = len(topology._explore(region, TopAbs_EDGE))
print(f"  자유경계 {len(before)}개  모서리 {edges_before:,}")

t0 = time.time()
found = topology.check_intersections(region)
print(f"\n자기교차 검사 {found.seconds:.1f}s → {found.self_intersections:,}건, "
      f"관련 형상 {found.faulty_faces}개")

t0 = time.time()
split, warnings = topology.split_at_intersections(region)
seconds = time.time() - t0
for w in warnings:
    print(f"  경고: {w}")
after_faces = len(topology._explore(split, TopAbs_FACE))
after_edges = len(topology._explore(split, TopAbs_EDGE))
print(f"분할 {seconds:.1f}s → 면 {picked} → {after_faces}  "
      f"모서리 {edges_before:,} → {after_edges:,}  "
      f"표면적 {area(split) / 1e6:.4f} m2")

recheck = topology.check_intersections(split)
print(f"분할 후 재검사 {recheck.seconds:.1f}s → 자기교차 "
      f"{recheck.self_intersections:,}건")
after = brep.free_boundaries(split)[0]
print(f"자유경계 {len(before)} → {len(after)}개")
for b in sorted(after, key=lambda x: -x.size)[:6]:
    print(f"    크기 {b.size:8.1f}  둘레 {b.length:9.1f}")
