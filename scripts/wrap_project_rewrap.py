"""Sharpen a coarse wrap: remesh, pull it back onto the original, wrap again finely.

    wrap_project_rewrap.py --wrap coarse.stl --reference original.stl --out sharp.stl

The August GT-R recipe. A coarse alpha (29 mm) is the only wrap that closes a
mesh whose panel seams are wider than a fine alpha, but it smears every detail.
The coarse wrap is closed, so: remesh it to --edge, move every vertex to the
closest point of the original when that is within --max-move (the seams and
the assumed floor stay where the wrap put them), then wrap the result at
--fine-alpha. The fine wrap now has no seam to leak through, and it removes the
self-intersections the projection creates. Measured 2026-08-21 on GT-R: alpha 29 →
remesh 10 + project 13 → alpha 6.5 gave the recommended B_sharp_clean.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--wrap", type=Path, required=True, help="the coarse, closed wrap")
ap.add_argument("--reference", type=Path, required=True, help="the original mesh")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--edge", type=float, default=10.0, help="mm; remesh edge length before projecting; 0 = no remesh")
ap.add_argument("--max-move", type=float, default=13.0, help="mm; a vertex farther than this from the original stays")
ap.add_argument("--fine-alpha", type=float, default=6.5, help="mm; the second wrap")
ap.add_argument("--report", type=Path)
args = ap.parse_args()

t0 = time.time()
ref = trimesh.load(args.reference, force="mesh")
ref.merge_vertices()
coarse = trimesh.load(args.wrap, force="mesh")
coarse.merge_vertices()
print(f"거친 랩 {args.wrap.name}: 삼각형 {len(coarse.faces):,}  수밀 {coarse.is_watertight}  체적 {abs(coarse.volume)/1e9:.3f} m³")

# 1. remesh (CGAL isotropic, whole mesh, read back through vertex ids); --edge 0 skips it
#    (a marching-cubes level set is already at its voxel size, and its merged vertices
#    can make the CGAL polyhedron builder reject it)
if args.edge > 0:
    from CGAL import CGAL_Polygon_mesh_processing as P
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    h, tmp = tempfile.mkstemp(suffix=".off")
    os.close(h)
    coarse.export(tmp)
    poly = Polyhedron_3(tmp)
    facets = list(poly.facets())
    P.isotropic_remeshing(facets, float(args.edge), poly, 3)
    os.unlink(tmp)
    verts, faces = [], []
    for i, v in enumerate(poly.vertices()):
        pt = v.point()
        verts.append((pt.x(), pt.y(), pt.z()))
        v.set_id(i)
    for f in poly.facets():
        hh = f.halfedge()
        tri = []
        for _ in range(3):
            tri.append(hh.vertex().id())
            hh = hh.next()
        faces.append(tri)
    mesh = trimesh.Trimesh(np.array(verts), np.array(faces), process=False)
    mesh.merge_vertices()
    print(f"재메쉬 {args.edge:.0f} mm: 삼각형 {len(mesh.faces):,}  {time.time()-t0:.0f}s")
else:
    mesh = coarse
    print(f"재메쉬 생략 (--edge 0): 삼각형 {len(mesh.faces):,}")

# 2. project onto the original
t1 = time.time()
cp, d, _ = trimesh.proximity.closest_point(ref, mesh.vertices)
move = d <= args.max_move
V = mesh.vertices.copy()
V[move] = cp[move]
bad = ~np.isfinite(V).all(axis=1)
if bad.any():
    V[bad] = mesh.vertices[bad]
    print(f"   투영 결과가 유한하지 않은 정점 {int(bad.sum())}개는 제자리")
mesh = trimesh.Trimesh(V, mesh.faces, process=False)
print(f"투영: 정점 {int(move.sum()):,}/{len(V):,} 이동 (≤ {args.max_move:.0f} mm), 거리 p50 {np.median(d):.1f} p90 {np.percentile(d,90):.1f} mm  {time.time()-t1:.0f}s")
projected = args.out.with_name(args.out.stem + "_projected.stl")
mesh.export(projected)

# 3. fine wrap, in a separate process: once CGAL_Polygon_mesh_processing has been
# imported (for the remesh), the SWIG Polyhedron type no longer matches what
# alpha_wrap_3 expects in this process ("wrong number or type of arguments")
t1 = time.time()
import subprocess
diag = float(np.linalg.norm(mesh.extents))
proc = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "wrap_once.py"), "--mesh", str(projected), "--out", str(args.out),
                       "--alpha-div", f"{diag / float(args.fine_alpha):.3f}"], capture_output=True, text=True)
if proc.returncode != 0 or not args.out.exists():
    print("가는 랩 실패:", (proc.stdout + proc.stderr).strip().splitlines()[-1] if (proc.stdout + proc.stderr).strip() else "")
    sys.exit(1)
fine = trimesh.load(args.out, force="mesh")
fine.merge_vertices()
if fine.volume < 0:
    fine.invert()
fill = abs(fine.volume) / float(np.prod(fine.extents))
pts, _ = trimesh.sample.sample_surface(fine, 20000, seed=1)
_, dev, _ = trimesh.proximity.closest_point(ref, pts)
print(f"가는 랩 {args.fine_alpha} mm: 삼각형 {len(fine.faces):,}  수밀 {fine.is_watertight}  몸체 {fine.body_count}  체적 {abs(fine.volume)/1e9:.3f} m³ (채움률 {fill:.2f})  "
      f"원본과의 거리 p50 {np.median(dev):.2f} p90 {np.percentile(dev,90):.1f} mm  {time.time()-t1:.0f}s")
fine.export(args.out)
print(f"저장 {args.out}  총 {time.time()-t0:.0f}s")
if args.report:
    import json
    args.report.write_text(json.dumps({"coarse_faces": int(len(coarse.faces)), "remesh_faces": int(len(mesh.faces)), "moved": int(move.sum()),
                                       "fine_faces": int(len(fine.faces)), "watertight": bool(fine.is_watertight), "bodies": int(fine.body_count),
                                       "fill": round(fill, 3), "dev_p50_mm": round(float(np.median(dev)), 2), "dev_p90_mm": round(float(np.percentile(dev, 90)), 2),
                                       "seconds": round(time.time() - t0)}, indent=1))
sys.stdout.flush()
os._exit(0)
