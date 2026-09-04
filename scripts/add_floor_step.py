"""Give the CAD engineer the same assumptions as the CFD mesh, as real surfaces.

    add_floor_step.py --in var/runs/cas-a/healed.stp --floor var/runs/cas-a-assumed/floor.stl \\
                      --out var/runs/cas-a-assumed/healed_full_floor.stp

The wrapped STL is the solver's surface; handing it back as STEP would mean one
planar face per triangle - a file no one can edit. This writes the B-rep instead:
the healed half body, its mirror about y=0, and the flat floor as one planar face
bounded by the footprint outline the wrap stage found. Real surfaces, real
edges, the two declared assumptions visible as geometry. It is not sewn into a
single solid - the overlapping styling panels still prevent that - but it is what
a CAD system can open, measure and change.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aox_g3 import brep, cad  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="src", type=Path, required=True, help="healed half-body STEP")
ap.add_argument("--floor", type=Path, required=True, help="floor.stl from flat_floor_wrap.py")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--no-mirror", action="store_true")
args = ap.parse_args()

import trimesh  # noqa: E402
from OCP.BRep import BRep_Builder  # noqa: E402
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,  # noqa: E402
                                BRepBuilderAPI_Transform)
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf  # noqa: E402
from OCP.TopoDS import TopoDS_Compound  # noqa: E402

shape, report = cad.read_step(args.src)
if shape is None:
    raise SystemExit(f"읽기 실패: {report.warnings}")
cad.diagnose(shape, report)
print(f"본체: 면 {report.faces:,}  bbox y {report.bbox[1]:.0f} ~ {report.bbox[4]:.0f}")

builder = BRep_Builder()
compound = TopoDS_Compound()
builder.MakeCompound(compound)
builder.Add(compound, shape)

if not args.no_mirror:
    # Mirror across the y=0 plane; the transform keeps the surfaces exact
    mirror = gp_Trsf()
    mirror.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)))
    mirrored = BRepBuilderAPI_Transform(shape, mirror, True).Shape()
    builder.Add(compound, mirrored)
    print("미러 추가 (y=0)")

# The floor outline: the boundary ring of the floor cap the wrap stage built,
# in order, as a closed planar polygon
floor_mesh = trimesh.load(args.floor, force="mesh")
from aox_g3 import fair  # noqa: E402
loops, _ = fair.boundary_loops(floor_mesh)
if not loops:
    raise SystemExit("floor.stl 에 경계 고리가 없습니다")
ring = floor_mesh.vertices[max(loops, key=len)]
z = float(np.median(ring[:, 2]))
polygon = BRepBuilderAPI_MakePolygon()
for p in ring:
    polygon.Add(gp_Pnt(float(p[0]), float(p[1]), z))
polygon.Close()
face = BRepBuilderAPI_MakeFace(polygon.Wire(), True).Face()
builder.Add(compound, face)
print(f"평바닥 면 추가: z = {z:.0f}, 꼭짓점 {len(ring)}개")

ok = brep.write_step(compound, args.out, units="MM" if report.units_hint == "mm" else "M")
print(f"{'저장' if ok else '실패'}: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)" if ok else "저장 실패")
