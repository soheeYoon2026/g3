"""Try MeshLib on the welded car5 mesh: what does each operation do to the openings?

Steps, each timed and checked (triangles, watertight, bodies):
  1 degenerate faces -> fixMeshDegeneracies
  2 tunnels shorter than --tunnel-max (the hollow tube bores) -> eliminateTunnels
  3 remesh to --edge
  4 offsetMesh(0) at --voxel and doubleOffsetMesh(+r,-r): a level-set resurfacing and closing,
    to compare with the alpha wrap
Writes the wrap-free result of 1-3 as --out.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
from meshlib import mrmeshpy as MR
from meshlib import mrmeshnumpy as MN

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--in", dest="src", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--tunnel-max", type=float, default=200.0, help="mm; tunnel loops shorter than this are bores to fill")
ap.add_argument("--edge", type=float, default=8.0)
ap.add_argument("--voxel", type=float, default=2.0)
ap.add_argument("--close-radius", type=float, default=8.0)
ap.add_argument("--skip-offset", action="store_true")
args = ap.parse_args()


def to_trimesh(m):
    return trimesh.Trimesh(np.asarray(MN.getNumpyVerts(m)), np.asarray(MN.getNumpyFaces(m.topology)), process=False)


def report(label, m, t0):
    t = to_trimesh(m)
    t.merge_vertices()
    print(f"{label}: 삼각형 {len(t.faces):,}  수밀 {t.is_watertight}  몸체 {t.body_count}  체적 {abs(t.volume)/1e9:.4f} m³  {time.time()-t0:.1f}s")
    return t


t0 = time.time()
mesh = MR.loadMesh(str(args.src))
report("입력", mesh, t0)

# 1 degeneracies
t0 = time.time()
deg = MR.findDegenerateFaces(mesh, 1000.0)
n_before = deg.count()
p = MR.FixMeshDegeneraciesParams()
p.maxDeviation = 0.05
p.tinyEdgeLength = 0.05
p.criticalTriAspectRatio = 1000.0
MR.fixMeshDegeneracies(mesh, p)
n_after = MR.findDegenerateFaces(mesh, 1000.0).count()
print(f"1 퇴화 삼각형(종횡비>1000) {n_before} → {n_after}")
report("  fixMeshDegeneracies 뒤", mesh, t0)

# 2 tunnels
t0 = time.time()
loops = MR.detectBasisTunnels(mesh)
lengths = []
for loop in loops:
    L = 0.0
    for e in loop:
        L += mesh.edgeLength(MR.UndirectedEdgeId(e.undirected()))
    lengths.append(L)
lengths = np.array(lengths)
print(f"2 기본 터널 고리 {len(loops)}개: 길이 mm 정렬 {np.sort(lengths).round(0).astype(int).tolist()[:40]}")
s = MR.DetectTunnelSettings()
s.maxTunnelLength = float(args.tunnel_max)
MR.eliminateTunnels(mesh, MR.FaceBitSet(), s)
loops2 = MR.detectBasisTunnels(mesh)
print(f"  eliminateTunnels(maxTunnelLength={args.tunnel_max:.0f}) 뒤 터널 고리 {len(loops2)}개")
report("  터널 제거 뒤", mesh, t0)

# 3 remesh
t0 = time.time()
rs = MR.RemeshSettings()
rs.targetEdgeLen = float(args.edge)
rs.maxEdgeSplits = 10_000_000
rs.finalRelaxIters = 2
rs.useCurvature = True
MR.remesh(mesh, rs)
out = report(f"3 remesh(변 {args.edge:.0f} mm) 뒤", mesh, t0)
out.export(args.out)
print(f"저장 {args.out}")

# 4 voxel offsets on the input
if not args.skip_offset:
    src = MR.loadMesh(str(args.src))
    op = MR.OffsetParameters()
    op.voxelSize = float(args.voxel)
    t0 = time.time()
    o0 = MR.offsetMesh(src, 0.0, op)
    t = report(f"4a offsetMesh(0, 복셀 {args.voxel} mm)", o0, t0)
    t.export(args.out.with_name(args.out.stem + "_offset0.stl"))
    t0 = time.time()
    r = float(args.close_radius)
    o2 = MR.doubleOffsetMesh(src, r, -r, op)
    t = report(f"4b doubleOffsetMesh(+{r:.0f}, -{r:.0f}, 복셀 {args.voxel} mm)", o2, t0)
    t.export(args.out.with_name(args.out.stem + f"_close{r:.0f}.stl"))
