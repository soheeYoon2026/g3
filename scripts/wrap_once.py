"""Wrap once at a given alpha and write the result, so it can be inspected."""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import trimesh

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--mesh", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--alpha-div", type=float, default=90.0)
args = ap.parse_args()

from CGAL import CGAL_Alpha_wrap_3 as AW
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3

mesh = trimesh.load(args.mesh, force="mesh")
diag = float(np.linalg.norm(np.asarray(mesh.extents, dtype=float)))
alpha = diag / args.alpha_div
offset = alpha / 30.0

points = AW.Point_3_Vector()
points.reserve(len(mesh.vertices))
for v in np.asarray(mesh.vertices, dtype=float):
    points.append(Point_3(float(v[0]), float(v[1]), float(v[2])))
polygons = AW.Polygon_Vector()
polygons.reserve(len(mesh.faces))
for f in np.asarray(mesh.faces):
    iv = AW.Int_Vector()
    iv.reserve(3)
    for k in f:
        iv.append(int(k))
    polygons.append(iv)

out = Polyhedron_3()
AW.alpha_wrap_3(points, polygons, alpha, offset, out)
handle, tmp = tempfile.mkstemp(suffix=".off")
os.close(handle)
out.write_to_file(tmp)
wrapped = trimesh.load(tmp, force="mesh")
os.unlink(tmp)
wrapped.export(args.out)
print(f"alpha {alpha:.1f} (대각선/{args.alpha_div:.0f})  offset {offset:.2f}")
print(f"삼각형 {len(wrapped.faces):,}  -> {args.out}")
