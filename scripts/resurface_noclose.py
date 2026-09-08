"""Re-surface a mesh without closing any opening (MeshLib voxel offset at zero).

    offsetMesh(mesh, 0, voxel)  -> a closed level-set surface of the input at voxel
                                   resolution: every gap wider than about two voxels
                                   stays open, walls thinner than a voxel come out
                                   solid (car5's 1.6 mm tube walls), needle triangles
                                   disappear
    decimateMesh(maxError)      -> back to a sensible triangle count
    drop dust bodies            -> voxel specks under --min-body-faces faces

Measured on car5 (2026-09-08): 13 s, 734k triangles, watertight, the 4 delivered
bodies, added area 0.0 %, deviation p90 0.06 mm, plates 4.9 mm intact. Use it
when the openings must all stay, or as the base before a local closing.
"""

import argparse
import collections
import time
from pathlib import Path

import numpy as np
import trimesh
from meshlib import mrmeshpy as MR
from meshlib import mrmeshnumpy as MN

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="src", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--voxel", type=float, default=2.0, help="mm; gaps under ~2 voxels close, walls under 1 voxel fill")
ap.add_argument("--max-error", type=float, default=0.5, help="decimation error bound, mm")
ap.add_argument("--max-edge", type=float, default=15.0, help="decimation edge cap, mm")
ap.add_argument("--min-body-faces", type=int, default=100, help="drop bodies smaller than this (voxel dust)")
args = ap.parse_args()

t0 = time.time()
src = trimesh.load(args.src, force="mesh")
src.merge_vertices()
print(f"입력 {args.src.name}: 삼각형 {len(src.faces):,}  수밀 {src.is_watertight}  몸체 {src.body_count}  체적 {abs(src.volume)/1e9:.4f} m³")

mesh = MR.loadMesh(str(args.src))
op = MR.OffsetParameters()
op.voxelSize = float(args.voxel)
t1 = time.time()
surf = MR.offsetMesh(mesh, 0.0, op)
print(f"offsetMesh(0, 복셀 {args.voxel} mm): 삼각형 {surf.topology.numValidFaces():,}  {time.time()-t1:.0f}s")

t1 = time.time()
ds = MR.DecimateSettings()
ds.maxError = float(args.max_error)
ds.maxEdgeLen = float(args.max_edge)
ds.packMesh = True
MR.decimateMesh(surf, ds)
print(f"decimateMesh(오차 {args.max_error} mm, 변 ≤ {args.max_edge:.0f}): 삼각형 {surf.topology.numValidFaces():,}  {time.time()-t1:.0f}s")

out = trimesh.Trimesh(np.asarray(MN.getNumpyVerts(surf)), np.asarray(MN.getNumpyFaces(surf.topology)), process=False)
out.merge_vertices()
parts = out.split(only_watertight=False)
kept = [p for p in parts if len(p.faces) >= args.min_body_faces]
if len(kept) < len(parts):
    print(f"먼지 몸체 {len(parts) - len(kept)}개 제거 (면 < {args.min_body_faces})")
out = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]
out.merge_vertices()
if out.volume < 0:
    out.invert()

cnt = collections.Counter(map(tuple, out.edges_sorted))
boundary = sum(1 for v in cnt.values() if v == 1)
nonmanifold = sum(1 for v in cnt.values() if v > 2)
_, d, _ = trimesh.proximity.closest_point(src, out.triangles_center)
added = out.area_faces[d > 1.5].sum()
out.export(args.out)
print(f"결과: 삼각형 {len(out.faces):,}  수밀 {out.is_watertight} (경계 {boundary}, 비다양체 {nonmanifold})  몸체 {out.body_count}  "
      f"체적 {abs(out.volume)/1e9:.4f} m³ (입력 대비 {abs(out.volume)/abs(src.volume)*100-100:+.1f} %)  "
      f"덧댄 면적 {added/100:.0f} cm² ({added/out.area*100:.2f} %)  {time.time()-t0:.0f}s")
print(f"저장 {args.out}")
