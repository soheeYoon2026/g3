"""Smooth the seams a wrap leaves where it bridged gaps, slots and plate edges.

The wrap is faithful to the input everywhere except where it added material
(centroid farther than --web-distance from the reference). Those "webbed" faces,
grown by --rings, are the only ones touched:

  --remesh T      CGAL isotropic remeshing of the webbed faces to edge length T
                  (the sawtooth is made of needle triangles; remeshing evens
                  them out before any smoothing)
  --smooth fair   the pipeline's bi-Laplacian fairing (aox_g3.fair.fair): free
                  vertices minimise thin-plate energy with the faithful zone
                  pinned; only the normal component of the move is applied
  --smooth taubin masked Taubin smoothing (lambda/mu umbrella passes) on the same
                  free set, --iterations passes

Every run reports how far the faithful zone moved, which must stay near zero.
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy import sparse

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="src", type=Path, required=True, help="wrapped mesh")
ap.add_argument("--reference", type=Path, required=True, help="the mesh that was wrapped")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--web-distance", type=float, default=1.0)
ap.add_argument("--rings", type=int, default=2, help="grow the webbed zone by this many vertex rings")
ap.add_argument("--remesh", type=float, default=0.0, help="target edge length for CGAL isotropic remeshing of the webbed faces (0 = skip)")
ap.add_argument("--remesh-iterations", type=int, default=3)
ap.add_argument("--smooth", choices=["none", "fair", "taubin"], default="fair")
ap.add_argument("--iterations", type=int, default=20, help="taubin passes")
ap.add_argument("--max-shift", type=float, help="cap on the normal move per vertex (fair)")
args = ap.parse_args()

t0 = time.time()
ref = trimesh.load(args.reference, force="mesh")
ref.merge_vertices()
mesh = trimesh.load(args.src, force="mesh")
mesh.merge_vertices()
print(f"입력 {args.src.name}: 삼각형 {len(mesh.faces):,}  정점 {len(mesh.vertices):,}  수밀 {mesh.is_watertight}")


def webbed_faces(m):
    _, d, _ = trimesh.proximity.closest_point(ref, m.triangles_center)
    return d > args.web_distance


def grow(m, face_mask, rings):
    vmask = np.zeros(len(m.vertices), bool)
    vmask[m.faces[face_mask]] = True
    adj = m.vertex_neighbors
    for _ in range(rings):
        new = vmask.copy()
        for i in np.flatnonzero(vmask):
            new[adj[i]] = True
        vmask = new
    return vmask


web = webbed_faces(mesh)
print(f"웹 구간: 면 {web.sum():,} ({web.mean()*100:.1f}%)  면적 {mesh.area_faces[web].sum()/mesh.area*100:.1f}%")

if args.remesh > 0:
    from CGAL import CGAL_Polygon_mesh_processing as P
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    h, tmp = tempfile.mkstemp(suffix=".off")
    os.close(h)
    mesh.export(tmp)
    poly = Polyhedron_3(tmp)
    facets = list(poly.facets())
    assert len(facets) == len(mesh.faces)
    # facet order must match trimesh face order; check on a sample by centroid
    for k in np.linspace(0, len(facets) - 1, 50).astype(int):
        hh = facets[k].halfedge()
        pts = []
        for _ in range(3):
            v = hh.vertex().point(); pts.append([v.x(), v.y(), v.z()]); hh = hh.next()
        assert np.allclose(np.mean(pts, axis=0), mesh.triangles_center[k], atol=1e-6), "facet order differs"
    sel_v = grow(mesh, web, args.rings)
    sel_f = sel_v[mesh.faces].any(axis=1)
    flist = [facets[i] for i in np.flatnonzero(sel_f)]
    P.isotropic_remeshing(flist, float(args.remesh), poly, int(args.remesh_iterations))
    poly.write_to_file(tmp)
    mesh = trimesh.load(tmp, force="mesh")
    os.unlink(tmp)
    mesh.merge_vertices()
    print(f"재메쉬(웹 {sel_f.sum():,}면 → 목표 변 {args.remesh} mm): 삼각형 {len(mesh.faces):,}  수밀 {mesh.is_watertight}  {time.time()-t0:.0f}s")
    web = webbed_faces(mesh)

free = grow(mesh, web, args.rings)
print(f"자유 정점 {free.sum():,} / {len(mesh.vertices):,}  (웹 + {args.rings}링)")
before = mesh.vertices.copy()

if args.smooth == "fair" and free.any():
    from aox_g3.fair import fair
    fixed_idx = np.flatnonzero(~free)
    free_idx = np.flatnonzero(free)
    perm = np.concatenate([fixed_idx, free_idx])
    inv = np.empty(len(perm), int); inv[perm] = np.arange(len(perm))
    pts = mesh.vertices[perm]
    tris = inv[mesh.faces]
    new = fair(pts, tris, fixed_count=len(fixed_idx), ring_points=[], ring_faces=[],
               max_normal_shift=args.max_shift)
    verts = np.empty_like(pts); verts[perm] = new
    mesh = trimesh.Trimesh(verts, mesh.faces, process=False)
elif args.smooth == "taubin" and free.any():
    n = len(mesh.vertices)
    e = mesh.edges_unique
    A = sparse.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel(); deg[deg == 0] = 1
    L = sparse.diags(1 / deg) @ A - sparse.identity(n)
    v = mesh.vertices.copy()
    fmask = free[:, None]
    for _ in range(args.iterations):
        v = v + 0.5 * (L @ v) * fmask
        v = v - 0.53 * (L @ v) * fmask
    mesh = trimesh.Trimesh(v, mesh.faces, process=False)

moved = np.linalg.norm(mesh.vertices - before, axis=1)
if free.any():
    print(f"이동: 자유 정점 p50 {np.percentile(moved[free],50):.2f} p90 {np.percentile(moved[free],90):.2f} max {moved[free].max():.2f} mm; 고정 정점 max {moved[~free].max() if (~free).any() else 0:.3f} mm")
mesh.export(args.out)
print(f"저장 {args.out}  삼각형 {len(mesh.faces):,}  수밀 {mesh.is_watertight}  체적 {abs(mesh.volume)/1e9:.4f} m³  {time.time()-t0:.0f}s")
if args.remesh > 0:
    # the SWIG CGAL binding prints one "memory leak ... no destructor found" line per
    # facet handle at interpreter exit (tens of MB for a car); skip the teardown
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
