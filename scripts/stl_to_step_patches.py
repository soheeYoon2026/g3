"""Put what the mesh tier changed back into the STEP.

    stl_to_step_patches.py --step healed.stp --stl wrapped.stl --out patched.stp

Compares the B-rep (tessellated per face) with the modified mesh (a wrap or a
resurfacing) and writes a STEP that contains:

  * every original face the mesh still has (samples of the face within
    --face-tol of the mesh), and
  * the material the mesh added (mesh triangles farther than --tol from the
    B-rep, grouped into patches, decimated, written as faceted shells).

Faces the mesh no longer has - hidden faces inside the wrap, panels under an
overlay - are listed, and dropped with --remove-missing. A half-model STEP gets
the mesh clipped to its side of y=0 so the patches stay a half too. The patches
are planar triangles, not NURBS: real surfaces stay real, and only the closed
openings are faceted.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aox_g3 import brep, cad, topology  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--step", type=Path, required=True, help="healed STEP (half or full)")
ap.add_argument("--stl", type=Path, required=True, help="the modified mesh (wrap, resurfacing)")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--report", type=Path)
ap.add_argument("--tol", type=float, default=3.0, help="mm; mesh triangles farther than this from the B-rep are new material")
ap.add_argument("--face-tol", type=float, default=5.0, help="mm; a B-rep face whose samples are mostly farther than this from the mesh is missing")
ap.add_argument("--min-patch-area", type=float, default=20.0, help="cm²; smaller patches are ignored")
ap.add_argument("--decimate", type=float, default=1.0, help="mm; patch decimation error (0 = keep the wrap triangles)")
ap.add_argument("--remove-missing", action="store_true", help="drop the B-rep faces the mesh no longer has")
ap.add_argument("--plane-tol", type=float, default=1.5, help="mm; a patch region this close to one plane is written as one planar face with its loops (the floor: 40,806 triangles -> 1 face)")
ap.add_argument("--full", action="store_true", help="the STEP is a full model; do not clip the mesh at y=0")
args = ap.parse_args()

import networkx as nx  # noqa: E402
import trimesh  # noqa: E402
from OCP.BRep import BRep_Builder  # noqa: E402
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Sewing  # noqa: E402
from OCP.gp import gp_Pnt  # noqa: E402
from OCP.TopoDS import TopoDS_Compound  # noqa: E402

t0 = time.time()
shape, cad_report = cad.read_step(args.step)
if shape is None:
    raise SystemExit(f"STEP 읽기 실패: {cad_report.warnings}")
cad.diagnose(shape, cad_report)
step_mesh, owner, faces = topology.tessellate_with_owner(shape)
print(f"STEP {args.step.name}: 면 {len(faces):,}  삼각형 {len(step_mesh.faces):,}  bbox y {cad_report.bbox[1]:.0f}~{cad_report.bbox[4]:.0f}  {time.time()-t0:.0f}s")

mesh = trimesh.load(args.stl, force="mesh")
mesh.merge_vertices()
half = not args.full and (cad_report.bbox[1] >= -1.0 or cad_report.bbox[4] <= 1.0 or abs(cad_report.bbox[1] + cad_report.bbox[4]) > 0.3 * (cad_report.bbox[4] - cad_report.bbox[1]))
if half:
    side_negative = cad_report.bbox[1] + cad_report.bbox[4] < 0
    normal = [0, -1, 0] if side_negative else [0, 1, 0]
    mesh = mesh.slice_plane([0, 0, 0], normal, cap=False)
    print(f"반쪽 STEP: 메쉬를 y {'≤' if side_negative else '≥'} 0 으로 자름 → 삼각형 {len(mesh.faces):,}")
else:
    print(f"메쉬 {args.stl.name}: 삼각형 {len(mesh.faces):,}")

# 1. mesh material that the B-rep does not have
t1 = time.time()
_, d_mesh, _ = trimesh.proximity.closest_point(step_mesh, mesh.triangles_center)
new = d_mesh > args.tol
adj = mesh.face_adjacency
g = nx.Graph()
g.add_nodes_from(np.flatnonzero(new))
g.add_edges_from(adj[new[adj].all(axis=1)])
patches = []
for comp in nx.connected_components(g):
    idx = np.fromiter(comp, int)
    area = float(mesh.area_faces[idx].sum())
    if area < args.min_patch_area * 100.0:
        continue
    patches.append(idx)
patches.sort(key=lambda i: -mesh.area_faces[i].sum())
print(f"메쉬가 덧댄 재료: 삼각형 {int(new.sum()):,} → 패치 {len(patches)}개 (면적 ≥ {args.min_patch_area:.0f} cm²)  {time.time()-t1:.0f}s")

# 2. B-rep faces the mesh no longer has
t1 = time.time()
_, d_step, _ = trimesh.proximity.closest_point(mesh, step_mesh.vertices)
tri_owner = owner
vert_owner = np.full(len(step_mesh.vertices), -1)
vert_owner[step_mesh.faces.ravel()] = np.repeat(tri_owner, 3)
face_area = np.zeros(len(faces))
np.add.at(face_area, tri_owner, step_mesh.area_faces)
missing = []
for k in range(len(faces)):
    sel = vert_owner == k
    if not sel.any():
        continue
    if np.median(d_step[sel]) > args.face_tol:
        c = step_mesh.vertices[sel].mean(axis=0)
        missing.append({"face": k, "area_cm2": round(face_area[k] / 100, 1), "center": c.round(0).tolist(),
                        "median_distance_mm": round(float(np.median(d_step[sel])), 1)})
print(f"메쉬에 없는 STEP 면: {len(missing)}개 (면적 {sum(m['area_cm2'] for m in missing)/1e4:.2f} m²)  {time.time()-t1:.0f}s")


def decimate(sub):
    if args.decimate <= 0:
        return sub
    try:
        from meshlib import mrmeshpy as MR, mrmeshnumpy as MN
        m = MN.meshFromFacesVerts(np.asarray(sub.faces, np.int32), np.asarray(sub.vertices, np.float32))
        ds = MR.DecimateSettings()
        ds.maxError = float(args.decimate)
        ds.maxEdgeLen = 200.0
        ds.touchNearBdEdges = False
        ds.packMesh = True
        MR.decimateMesh(m, ds)
        return trimesh.Trimesh(np.asarray(MN.getNumpyVerts(m)), np.asarray(MN.getNumpyFaces(m.topology)), process=False)
    except Exception as exc:
        print(f"   데시메이션 생략 ({type(exc).__name__}: {exc})")
        return sub


from aox_g3 import fair  # noqa: E402
from OCP.ShapeFix import ShapeFix_Face  # noqa: E402


def planar_region(sub):
    """Split a patch into (planar part, rest): faces on the patch's dominant plane."""
    n = sub.face_normals
    a = sub.area_faces
    # dominant normal: area-weighted, from the largest normal cluster (5 deg bins)
    best, best_area = None, 0.0
    for cand in n[np.argsort(-a)[:200]]:
        near = (n @ cand) > np.cos(np.radians(2.0))
        if a[near].sum() > best_area:
            best, best_area = cand, a[near].sum()
    if best is None:
        return None
    near = (n @ best) > np.cos(np.radians(2.0))
    offsets = sub.triangles_center[near] @ best
    off = float(np.median(offsets))
    on_plane = near.copy()
    on_plane[near] = np.abs(offsets - off) < args.plane_tol
    if a[on_plane].sum() < 0.3 * a.sum() or on_plane.sum() < 20:
        return None
    return best, off, on_plane


def planar_face(sub, normal, off, on_plane):
    """One face from the boundary loops of the on-plane faces; None if it fails."""
    part = sub.submesh([np.flatnonzero(on_plane)], append=True)
    loops, _ = fair.boundary_loops(part)
    if not loops:
        return None
    rings = []
    for loop in loops:
        pts = part.vertices[loop]
        pts = pts - np.outer(pts @ normal - off, normal)      # onto the plane exactly
        if len(pts) < 3:
            continue
        rings.append(pts)
    if not rings:
        return None
    # outer = largest projected area; the others are holes
    def ring_area(pts):
        u = np.cross(normal, [1, 0, 0]) if abs(normal[0]) < 0.9 else np.cross(normal, [0, 1, 0])
        u /= np.linalg.norm(u); v = np.cross(normal, u)
        x, y = pts @ u, pts @ v
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    rings.sort(key=ring_area, reverse=True)
    wires = []
    for pts in rings:
        poly = BRepBuilderAPI_MakePolygon()
        for q in pts:
            poly.Add(gp_Pnt(float(q[0]), float(q[1]), float(q[2])))
        poly.Close()
        if not poly.IsDone():
            return None
        wires.append(poly.Wire())
    maker = BRepBuilderAPI_MakeFace(wires[0], True)
    if not maker.IsDone():
        return None
    for w in wires[1:]:
        maker.Add(w)
    if not maker.IsDone():
        return None
    fixer = ShapeFix_Face(maker.Face())
    fixer.Perform()
    return fixer.Face(), part.area, len(rings)


# 3. build the STEP: kept faces + planar faces + faceted patch shells
builder = BRep_Builder()
compound = TopoDS_Compound()
builder.MakeCompound(compound)
drop = {m["face"] for m in missing} if args.remove_missing else set()
kept = 0
for k, f in enumerate(faces):
    if k in drop:
        continue
    builder.Add(compound, f)
    kept += 1

patch_rows = []
t1 = time.time()
for n, idx in enumerate(patches):
    sub = mesh.submesh([idx], append=True)
    before = len(sub.faces)
    planar_note = ""
    region = planar_region(sub)
    if region is not None:
        built = planar_face(sub, *region)
        if built is not None:
            face, plane_area, n_rings = built
            builder.Add(compound, face)
            rest = np.flatnonzero(~region[2])
            planar_note = f"평면 1면({plane_area/1e6:.2f} m², 고리 {n_rings}) + "
            if len(rest) == 0:
                c = mesh.triangles_center[idx].mean(axis=0)
                ext = mesh.triangles_center[idx].max(axis=0) - mesh.triangles_center[idx].min(axis=0)
                patch_rows.append({"patch": n, "area_cm2": round(float(mesh.area_faces[idx].sum()) / 100, 1), "center": c.round(0).tolist(),
                                   "extent": ext.round(0).tolist(), "triangles": int(before), "faces_written": 1, "planar": True})
                print(f"   패치 {n}: {mesh.area_faces[idx].sum()/100:8.0f} cm²  {planar_note}나머지 없음")
                continue
            sub = sub.submesh([rest], append=True)
    sub = decimate(sub)
    sewing = BRepBuilderAPI_Sewing(0.01)
    for a, b, c in sub.vertices[sub.faces]:
        poly = BRepBuilderAPI_MakePolygon(gp_Pnt(*map(float, a)), gp_Pnt(*map(float, b)), gp_Pnt(*map(float, c)), True)
        if poly.IsDone():
            face = BRepBuilderAPI_MakeFace(poly.Wire(), True)
            if face.IsDone():
                sewing.Add(face.Face())
    sewing.Perform()
    shell = sewing.SewedShape()
    builder.Add(compound, shell)
    c = mesh.triangles_center[idx].mean(axis=0)
    ext = mesh.triangles_center[idx].max(axis=0) - mesh.triangles_center[idx].min(axis=0)
    row = {"patch": n, "area_cm2": round(float(mesh.area_faces[idx].sum()) / 100, 1), "center": c.round(0).tolist(),
           "extent": ext.round(0).tolist(), "triangles": int(before), "faces_written": int(len(sub.faces)) + (1 if planar_note else 0),
           "planar": bool(planar_note)}
    patch_rows.append(row)
    print(f"   패치 {n}: {row['area_cm2']:8.0f} cm²  중심 {row['center']}  크기 {row['extent']}  삼각형 {before:,} → {planar_note}면 {len(sub.faces):,}")
print(f"패치 면 만들기·봉합 {time.time()-t1:.0f}s")

ok = brep.write_step(compound, args.out, units="MM" if cad_report.units_hint == "mm" else "M")
size = args.out.stat().st_size / 1e6 if ok else 0
print(f"{'저장' if ok else '실패'}: {args.out}  ({size:.1f} MB)  STEP 면 {kept:,} 유지 + 빠짐 {len(drop)} + 패치 면 {sum(r['faces_written'] for r in patch_rows):,}  총 {time.time()-t0:.0f}s")

# 4. read back
back, back_report = cad.read_step(args.out)
if back is not None:
    cad.diagnose(back, back_report)
    print(f"되읽기: 면 {back_report.faces:,}  bbox {[round(v) for v in back_report.bbox]}")

if args.report:
    args.report.write_text(json.dumps({"step": str(args.step), "stl": str(args.stl), "half": bool(half), "tol": args.tol,
                                       "face_tol": args.face_tol, "patches": patch_rows, "missing_faces": missing,
                                       "removed": bool(args.remove_missing), "faces_kept": kept,
                                       "out": str(args.out), "size_mb": round(size, 1)}, indent=1, ensure_ascii=False))
