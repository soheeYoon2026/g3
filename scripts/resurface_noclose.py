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
ap.add_argument("--offset", type=float, help="offset distance, mm. Default 0 for a closed input (faithful "
                     "resurfacing) and one voxel for an open one: at 0 an open sheet is a zero-thickness "
                     "isosurface and vanishes (CAS-A kept 9.7 of 34 m²); at +voxel it becomes a thin shell")
ap.add_argument("--max-error", type=float, default=0.5, help="decimation error bound, mm")
ap.add_argument("--max-edge", type=float, default=15.0, help="decimation edge cap, mm")
ap.add_argument("--min-body-faces", type=int, default=100, help="drop bodies smaller than this (voxel dust)")
ap.add_argument("--sign-mode", choices=["auto", "default", "HoleWindingRule", "WindingRule", "ProjectionNormal", "Unsigned"],
                default="default", help="inside/outside test for the voxel offset. Keep the MeshLib default: "
                     "on CAS-A HoleWindingRule looked clean (0 non-manifold edges) only because it had "
                     "dropped the whole body skin (open sheets) and kept the wheels; the default keeps "
                     "open sheets as thin shells (323 non-manifold edges where sheets touch)")
args = ap.parse_args()

t0 = time.time()
src = trimesh.load(args.src, force="mesh")
src.merge_vertices()
print(f"입력 {args.src.name}: 삼각형 {len(src.faces):,}  수밀 {src.is_watertight}  몸체 {src.body_count}  체적 {abs(src.volume)/1e9:.4f} m³")

# "open" means a real share of boundary edges, not the few left by degenerate slivers
# in a tessellated STEP (Cv10: 0.01 % of edges, still a closed model)
_, counts_in = np.unique(src.edges_sorted, axis=0, return_counts=True)
open_share = float((counts_in == 1).sum()) / max(1, len(counts_in))
open_input = open_share > 1e-3
print(f"입력 경계 모서리 비율 {open_share*100:.3f} %  → {'열림' if open_input else '닫힘'}")
mesh = MR.loadMesh(str(args.src))
op = MR.OffsetParameters()
op.voxelSize = float(args.voxel)
mode = args.sign_mode
if mode == "auto":
    mode = "default"
if mode != "default":
    op.signDetectionMode = getattr(MR.SignDetectionMode, mode)
print(f"부호 판정 {mode}  (입력 {'열림' if open_input else '닫힘'})")
if open_input:
    print("입력이 열려 있음: 오프셋 +복셀로 열린 판을 얇은 껍질 고체로 만든다(닫힌 몸체는 그만큼 부풂). 닫힌 차가 필요하면 평바닥 랩(C)")
offset = args.offset if args.offset is not None else (float(args.voxel) if open_input else 0.0)
t1 = time.time()
surf = MR.offsetMesh(mesh, float(offset), op)
print(f"offsetMesh({offset:g}, 복셀 {args.voxel} mm): 삼각형 {surf.topology.numValidFaces():,}  {time.time()-t1:.0f}s")

t1 = time.time()
ds = MR.DecimateSettings()
ds.maxError = float(args.max_error)
ds.maxEdgeLen = float(args.max_edge)
ds.packMesh = True
MR.decimateMesh(surf, ds)
print(f"decimateMesh(오차 {args.max_error} mm, 변 ≤ {args.max_edge:.0f}): 삼각형 {surf.topology.numValidFaces():,}  {time.time()-t1:.0f}s")

# dust bodies: drop them inside MeshLib (a trimesh split of millions of faces takes minutes)
t1 = time.time()
comps = MR.getAllComponents(surf)
dust = MR.FaceBitSet()
n_dust = 0
for comp in comps:
    if comp.count() < args.min_body_faces:
        dust |= comp
        n_dust += 1
if n_dust:
    surf.deleteFaces(dust)
    surf.pack()
    print(f"먼지 몸체 {n_dust}개 제거 (면 < {args.min_body_faces}), 남은 몸체 {len(comps) - n_dust}  {time.time()-t1:.0f}s")
out = trimesh.Trimesh(np.asarray(MN.getNumpyVerts(surf)), np.asarray(MN.getNumpyFaces(surf.topology)), process=False)
out.merge_vertices()
if out.volume < 0:
    out.invert()

# edge manifoldness with numpy, not a Python Counter over tens of millions of tuples
_, counts = np.unique(out.edges_sorted, axis=0, return_counts=True)
boundary = int((counts == 1).sum())
nonmanifold = int((counts > 2).sum())
# added area on a sample of faces: the exact closest-point pass over every face was
# 7 minutes on the GT-R (5.4M faces) for a number that only needs two digits
rng = np.random.default_rng(0)
idx = rng.choice(len(out.faces), size=min(200_000, len(out.faces)), replace=False)
_, d, _ = trimesh.proximity.closest_point(src, out.triangles_center[idx])
added = float(out.area_faces[idx][d > 1.5 + offset].sum() / out.area_faces[idx].sum() * out.area)
out.export(args.out)
print(f"결과: 삼각형 {len(out.faces):,}  수밀 {boundary == 0 and nonmanifold == 0} (경계 {boundary}, 비다양체 {nonmanifold})  몸체 {len(comps) - n_dust}  "
      f"체적 {abs(out.volume)/1e9:.4f} m³ (입력 대비 {abs(out.volume)/abs(src.volume)*100-100:+.1f} %)  "
      f"덧댄 면적 ≈ {added/100:.0f} cm² ({added/out.area*100:.2f} %, 표본 {len(idx):,}면)  {time.time()-t0:.0f}s")
print(f"저장 {args.out}")
