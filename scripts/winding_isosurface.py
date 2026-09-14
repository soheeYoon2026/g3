"""Closed surface from an open, dirty mesh: the 0.5 level set of the generalized winding number.

    winding_isosurface.py --in body_with_floor.stl --out iso.stl --voxel 5 [--orient none|trimesh]

No alpha. The winding number (Jacobson 2013; fast version Barill 2018, libigl) is
1 inside a closed oriented surface and 0 outside, and between the two where the
input has gaps: a gap between two panels is bridged where the field crosses 0.5,
which is roughly midway, without any size parameter. The price is that the input
must be oriented consistently: a flipped component subtracts 1 from the field
inside it. --orient trimesh flips closed components with negative volume; open
shells are left as they are. --orient igl makes every patch consistent
(bfs_orient) and then flips whole patches outward (orient_outward, an
area-weighted centroid heuristic that is right for convex-ish outer skins and
undecided for inner panels).

The field is sampled on a grid of --voxel and contoured with marching cubes, so the
result is smooth at the voxel scale and thin sheets thinner than a voxel are not
resolved. Only the largest connected piece of the level set is kept (the outer
surface; voids from inverted inner panels are dropped with it).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import igl
import numpy as np
import trimesh
from skimage import measure

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="inp", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--voxel", type=float, default=5.0, help="mm; grid spacing")
ap.add_argument("--iso", type=float, default=0.5)
ap.add_argument("--orient", choices=["none", "trimesh", "igl"], default="none",
                help="igl: bfs_orient makes each patch consistent, orient_outward flips whole patches so the area-weighted normals point away from the patch centroid")
ap.add_argument("--pad", type=int, default=3, help="voxels of empty margin around the bounding box")
ap.add_argument("--web-distance", type=float, default=1.5, help="mm; level-set faces farther than this from the input count as added material")
ap.add_argument("--report", type=Path)
ap.add_argument("--field", type=Path, help="save the sampled field as .npy (float32) for inspection")
args = ap.parse_args()

t0 = time.time()
lap = lambda: "%.0fs" % (time.time() - t0)
m = trimesh.load(args.inp, force="mesh")
m.merge_vertices()
print(f"input {args.inp.name}: 삼각형 {len(m.faces):,}  정점 {len(m.vertices):,}  방향 일관 {m.is_winding_consistent}  수밀 {m.is_watertight}")

# components and how many of them are closed / inverted
labels = trimesh.graph.connected_component_labels(m.face_adjacency, node_count=len(m.faces))
n_comp = int(labels.max()) + 1
boundary_edge_idx = trimesh.grouping.group_rows(m.edges_sorted, require_count=1)
boundary_faces = boundary_edge_idx // 3
open_comp = np.zeros(n_comp, bool)
open_comp[np.unique(labels[boundary_faces])] = True
comp_faces = np.bincount(labels, minlength=n_comp)
print(f"구성요소 {n_comp}  닫힌 것 {int((~open_comp).sum())}  열린 것 {int(open_comp.sum())}  (열린 것의 삼각형 {int(comp_faces[open_comp].sum()):,})")

flipped = 0
if args.orient == "trimesh":
    # signed volume per closed component; flip the negative ones
    tri = m.triangles
    vol6 = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
    comp_vol = np.bincount(labels, weights=vol6, minlength=n_comp) / 6.0
    inv = (~open_comp) & (comp_vol < 0)
    if inv.any():
        sel = inv[labels]
        faces = m.faces.copy()
        faces[sel] = faces[sel][:, ::-1]
        m = trimesh.Trimesh(m.vertices, faces, process=False)
        flipped = int(inv.sum())
    print(f"방향: 닫힌 구성요소 중 체적이 음수인 {flipped}개 뒤집음  {lap()}")

if args.orient == "igl":
    F0 = np.ascontiguousarray(m.faces, dtype=np.int64)
    FF, C = igl.bfs_orient(F0)
    n_changed = int((FF != F0).any(axis=1).sum())
    FF2, I = igl.orient_outward(np.ascontiguousarray(m.vertices, dtype=np.float64), np.ascontiguousarray(FF, dtype=np.int64), np.ascontiguousarray(np.asarray(C).reshape(-1, 1), dtype=np.int64))
    flipped = int(np.asarray(I).astype(bool).sum())
    m = trimesh.Trimesh(m.vertices, FF2, process=False)
    print(f"방향(igl): 패치 안 일관화로 뒤집은 면 {n_changed:,}, 바깥 향하도록 통째로 뒤집은 패치 {flipped}/{int(np.asarray(C).max())+1}  {lap()}")

V = np.ascontiguousarray(m.vertices, dtype=np.float64)
F = np.ascontiguousarray(m.faces, dtype=np.int64)

lo, hi = m.bounds
h = args.voxel
origin = lo - args.pad * h
n = np.ceil((hi - lo) / h).astype(int) + 2 * args.pad + 1
print(f"격자 {n[0]}×{n[1]}×{n[2]} = {int(np.prod(n)):,} 점, 복셀 {h} mm")
xs = origin[0] + h * np.arange(n[0])
ys = origin[1] + h * np.arange(n[1])
zs = origin[2] + h * np.arange(n[2])
field = np.empty(tuple(n), dtype=np.float32)
# build the BVH once; calling igl.fast_winding_number(V, F, Q) per slab rebuilds it every time (5 s each)
try:
    bvh = igl.FastWindingNumberBVH()
    bvh.init(V, F)
    evaluate = lambda Q: bvh.winding_number(Q)
    print(f"BVH 한 번 구축  {lap()}")
except Exception as e:  # older binding without the class
    print(f"BVH 클래스 없음({type(e).__name__}); 큰 덩어리로 한 번에 평가")
    evaluate = lambda Q: igl.fast_winding_number(V, F, Q)
X, Y = np.meshgrid(xs, ys, indexing="ij")
per_layer = n[0] * n[1]
layers = max(1, int(16_000_000 // per_layer))
for k0 in range(0, n[2], layers):
    k1 = min(n[2], k0 + layers)
    Q = np.empty(((k1 - k0) * per_layer, 3), dtype=np.float64)
    Q[:, 0] = np.tile(X.ravel(), k1 - k0)
    Q[:, 1] = np.tile(Y.ravel(), k1 - k0)
    Q[:, 2] = np.repeat(zs[k0:k1], per_layer)
    W = np.asarray(evaluate(Q)).reshape(k1 - k0, n[0], n[1])
    field[:, :, k0:k1] = np.transpose(W, (1, 2, 0)).astype(np.float32)
    print(f"  z 층 {k1}/{n[2]}  {lap()}")
print(f"필드 완료 {lap()}  값 범위 {field.min():.2f}..{field.max():.2f}  "
      f"음수(< -0.1) {float((field < -0.1).mean()):.3f}  1 초과(> 1.1) {float((field > 1.1).mean()):.3f}  "
      f"채움률(> {args.iso}) {float((field > args.iso).mean()):.3f}")
if args.field:
    np.save(args.field, field)

verts, faces, _, _ = measure.marching_cubes(field, level=args.iso, spacing=(h, h, h))
verts = verts + origin
iso = trimesh.Trimesh(verts, faces, process=True)
print(f"등위면: 삼각형 {len(iso.faces):,}  {lap()}")
parts = iso.split(only_watertight=False)
parts = sorted(parts, key=lambda p: len(p.faces), reverse=True)
print(f"  조각 {len(parts)}개, 가장 큰 것 {len(parts[0].faces):,} 면, 다음 {[len(p.faces) for p in parts[1:4]]}")
outer = parts[0]
outer.fix_normals()
bbox_vol = float(np.prod(hi - lo))
vol = float(abs(outer.volume))
print(f"바깥 조각: 수밀 {outer.is_watertight}  체적 {vol/1e9:.3f} m³  채움률 {vol/bbox_vol:.3f}")

# distance to the input: sampled points and face centroids
P = outer.sample(200_000)
d2, _, _ = igl.point_mesh_squared_distance(np.ascontiguousarray(P, dtype=np.float64), V, F)
d = np.sqrt(d2)
C = outer.triangles_center
dc2, _, _ = igl.point_mesh_squared_distance(np.ascontiguousarray(C, dtype=np.float64), V, F)
dc = np.sqrt(dc2)
area = outer.area_faces
added = float(area[dc > args.web_distance].sum() / area.sum())
print(f"원본과의 거리 p50 {np.percentile(d, 50):.2f}  p90 {np.percentile(d, 90):.2f}  p99 {np.percentile(d, 99):.1f} mm;  "
      f"덧댄 면적(> {args.web_distance} mm) {added*100:.1f} %  {lap()}")

outer.export(args.out)
rep = {
    "input": str(args.inp), "faces_in": int(len(m.faces)), "components": n_comp,
    "closed_components": int((~open_comp).sum()), "flipped": flipped,
    "voxel_mm": h, "iso": args.iso, "grid": [int(v) for v in n],
    "field_min": float(field.min()), "field_max": float(field.max()),
    "share_negative": float((field < -0.1).mean()), "share_over_one": float((field > 1.1).mean()),
    "faces_out": int(len(outer.faces)), "pieces": len(parts), "watertight": bool(outer.is_watertight),
    "volume_m3": vol / 1e9, "fill": vol / bbox_vol,
    "dev_p50_mm": float(np.percentile(d, 50)), "dev_p90_mm": float(np.percentile(d, 90)),
    "dev_p99_mm": float(np.percentile(d, 99)), "added_area_share": added,
    "seconds": round(time.time() - t0),
}
if args.report:
    args.report.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
print(f"저장 {args.out}  총 {lap()}")
sys.stdout.flush()
os._exit(0)
