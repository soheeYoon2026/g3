"""Measure what steps 7, 8 and 9 of the heal order would actually have to fix.

Building the three stages against an imagined defect list is how you end up with
code that handles cases the file does not have and misses the ones it does. So
measure first:

  7  intersections   faces that cut through each other, which a mesher will either
                     refuse or silently resolve the wrong way
  8  hidden faces    geometry the external flow never reaches, and baffles, which
     and baffles     are zero-thickness sheets with flow on both sides
  9  orientation     faces whose normal points into the body instead of out of it

The interesting one is 8, because "hidden" and "baffle" need a definition that can
be measured rather than eyeballed. Flood the empty space from outside the bounding
box, then probe each triangle just off its front and back. Outside on one side only
means it is skin; outside on both means a baffle; outside on neither means it is
buried inside and the flow never sees it.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aox_g3 import cad  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--step", type=Path, required=True)
ap.add_argument("--pitch-div", type=float, default=300.0,
                help="voxel pitch as a fraction of the diagonal")
ap.add_argument("--skip-intersections", action="store_true",
                help="the self-intersection check is the slow one")
ap.add_argument("--fuzzy", type=float, default=0.0,
                help="tolerance for the intersection check")
args = ap.parse_args()

import trimesh  # noqa: E402
from scipy import ndimage  # noqa: E402

from OCP.BOPAlgo import BOPAlgo_CheckerSI  # noqa: E402
from OCP.BRep import BRep_Tool  # noqa: E402
from OCP.BRepMesh import BRepMesh_IncrementalMesh  # noqa: E402
from OCP.ShapeAnalysis import ShapeAnalysis_Shell  # noqa: E402
from OCP.TopAbs import (TopAbs_FACE, TopAbs_SHELL, TopAbs_REVERSED,  # noqa: E402
                        TopAbs_FORWARD)
from OCP.TopExp import TopExp_Explorer  # noqa: E402
from OCP.TopLoc import TopLoc_Location  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402
from OCP.TopTools import TopTools_ListOfShape  # noqa: E402

shape, report = cad.read_step(args.step)
if shape is None:
    raise SystemExit(f"읽기 실패: {report.warnings}")
cad.diagnose(shape, report)
shape, report = cad.sew_progressive(shape, report)
after = cad.CadReport()
cad.diagnose(shape, after)
diag = float(np.linalg.norm(np.array(report.bbox[3:]) - np.array(report.bbox[:3])))
print(f"꿰맴 후: 면 {after.faces:,}  셸 {after.shells}  "
      f"자유모서리 {after.free_edges:,}  대각선 {diag:.0f}\n")

# ---------------------------------------------------------------- 9  orientation
print("== 9. 법선 방향 ==")
forward = reversed_ = 0
explorer = TopExp_Explorer(shape, TopAbs_FACE)
while explorer.More():
    if explorer.Current().Orientation() == TopAbs_REVERSED:
        reversed_ += 1
    elif explorer.Current().Orientation() == TopAbs_FORWARD:
        forward += 1
    explorer.Next()
print(f"  FORWARD {forward:,}  REVERSED {reversed_:,}")

bad_shells = 0
explorer = TopExp_Explorer(shape, TopAbs_SHELL)
while explorer.More():
    analyser = ShapeAnalysis_Shell()
    analyser.LoadShells(explorer.Current())
    # True asks it to check orientation consistency, not just connectivity
    analyser.CheckOrientedShells(explorer.Current(), True, True)
    if analyser.HasBadEdges():
        bad_shells += 1
    explorer.Next()
print(f"  방향이 어긋난 셸 {bad_shells} / {after.shells}")

# ------------------------------------------------------------ 8  hidden, baffles
print("\n== 8. 숨은 면과 baffle ==")
t0 = time.time()
deflection = diag * 0.001
BRepMesh_IncrementalMesh(shape, deflection, False, 0.5, True)

vertices, faces, owner = [], [], []
explorer = TopExp_Explorer(shape, TopAbs_FACE)
face_shapes = []
while explorer.More():
    face = TopoDS.Face_s(explorer.Current())
    index = len(face_shapes)
    face_shapes.append(face)
    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation_s(face, location)
    explorer.Next()
    if triangulation is None:
        continue
    transform = location.Transformation()
    offset = len(vertices)
    for i in range(1, triangulation.NbNodes() + 1):
        p = triangulation.Node(i).Transformed(transform)
        vertices.append((p.X(), p.Y(), p.Z()))
    for i in range(1, triangulation.NbTriangles() + 1):
        a, b, c = triangulation.Triangle(i).Get()
        faces.append((offset + a - 1, offset + b - 1, offset + c - 1))
        owner.append(index)

mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float),
                       faces=np.asarray(faces, dtype=np.int64), process=False)
owner = np.asarray(owner)
print(f"  삼각형화 {time.time() - t0:.1f}s   삼각형 {len(mesh.faces):,}  "
      f"B-rep 면 {len(face_shapes):,}")

t0 = time.time()
lo, hi = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
pitch = diag / args.pitch_div
pad = 4 * pitch
origin = lo - pad
grid = tuple(int(np.ceil((hi[i] + pad - origin[i]) / pitch)) + 1 for i in range(3))

wall = np.zeros(grid, dtype=bool)
tris = np.asarray(mesh.triangles, dtype=float)
a = np.clip(np.floor((tris.min(axis=1) - origin) / pitch).astype(np.int64),
            0, np.array(grid) - 1)
b = np.clip(np.floor((tris.max(axis=1) - origin) / pitch).astype(np.int64),
            0, np.array(grid) - 1)
for (i0, j0, k0), (i1, j1, k1) in zip(a, b):
    wall[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = True

labels, _ = ndimage.label(~wall)
outside = labels == labels[0, 0, 0]
print(f"  복셀 {grid} 피치 {pitch:.1f}  홍수채움 {time.time() - t0:.1f}s   "
      f"벽 {wall.sum():,}  바깥 {outside.sum():,}")


def probe(points):
    idx = np.floor((points - origin) / pitch).astype(np.int64)
    np.clip(idx, 0, np.array(grid) - 1, out=idx)
    return outside[idx[:, 0], idx[:, 1], idx[:, 2]]


centres = np.asarray(mesh.triangles_center, dtype=float)
normals = np.asarray(mesh.face_normals, dtype=float)
# Two voxels clears the wall the triangle itself painted
step = 2.0 * pitch
front = probe(centres + normals * step)
back = probe(centres - normals * step)

skin = front ^ back
baffle = front & back
buried = ~front & ~back
print(f"  삼각형: 외피 {skin.sum():,}  baffle {baffle.sum():,}  "
      f"숨음 {buried.sum():,}")

n_faces = len(face_shapes)
areas = np.asarray(mesh.area_faces)
per_face = np.zeros((n_faces, 3))
for kind, column in ((skin, 0), (baffle, 1), (buried, 2)):
    np.add.at(per_face[:, column], owner[kind], areas[kind])
total = per_face.sum(axis=1)
alive = total > 0
verdict = np.full(n_faces, -1)
verdict[alive] = per_face[alive].argmax(axis=1)
names = ["외피", "baffle", "숨음"]
print("  B-rep 면 분류 (면적 다수결):")
for k, name in enumerate(names):
    n = int((verdict == k).sum())
    area = per_face[verdict == k, k].sum() / 1e6
    print(f"    {name:>7} {n:5,}개   {area:8.2f} m2")
print(f"    {'삼각형 없음':>7} {int((verdict == -1).sum()):5,}개")
print(f"  전체 표면적 {mesh.area / 1e6:.2f} m2")

# ------------------------------------------------------------- 7  intersections
if not args.skip_intersections:
    print("\n== 7. 자기교차 ==")
    t0 = time.time()
    checker = BOPAlgo_CheckerSI()
    arguments = TopTools_ListOfShape()
    arguments.Append(shape)
    checker.SetArguments(arguments)
    checker.SetRunParallel(True)
    if args.fuzzy:
        checker.SetFuzzyValue(args.fuzzy)
    checker.Perform()
    seconds = time.time() - t0
    if checker.HasErrors():
        print(f"  검사기가 오류를 냈습니다 ({seconds:.1f}s)")
        checker.DumpErrors()
    else:
        result = checker.DS()
        n = result.NbShapes() if hasattr(result, "NbShapes") else -1
        interfered = checker.DS().Interferences() \
            if hasattr(checker.DS(), "Interferences") else None
        print(f"  검사 {seconds:.1f}s   DS 형상 {n:,}")
        for kind in ("FaceFace", "EdgeFace", "EdgeEdge", "VertexFace"):
            getter = getattr(result, kind, None)
            if getter is None:
                continue
            try:
                print(f"    {kind:<11} {getter().Size():,}")
            except Exception:
                pass
