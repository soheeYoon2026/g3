"""Close the openings the B-rep heal left, with caps shaped by the wrap.

    close_with_wrap.py --step healed.stp --wrap wrapped.stl --out v19.stp

For every free boundary of the STEP (ShapeAnalysis, so the real holes and not
the tessellation seams), sample its edges on their curves (--deflection), fill
the loop with a constrained Delaunay triangulation refined to --edge, move the
interior vertices onto the wrap along the loop's plane normal (rays both ways,
closest point as fallback), write the cap as planar triangles and sew everything
at --sew so the cap chords are cut into the original edges. The cap boundary is
the hole's own edges; the cap interior is where the wrap put the surface.
Loops lying entirely in the symmetry plane are skipped (the mirror closes them).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aox_g3 import brep, cad, fair  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--step", type=Path, required=True)
ap.add_argument("--wrap", type=Path, required=True, help="the wrap (or any closed mesh) that shows where the surface should be")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--report", type=Path)
ap.add_argument("--deflection", type=float, default=0.3, help="mm; edge sampling chord error")
ap.add_argument("--edge", type=float, default=0.0, help="mm; cap triangle edge (0 = perimeter/150, at least 15)")
ap.add_argument("--sew", type=float, default=0.5, help="mm; sewing tolerance")
ap.add_argument("--min-loop", type=float, default=30.0, help="mm; smaller loops are ignored")
ap.add_argument("--max-loops", type=int, default=0, help="0 = all")
ap.add_argument("--reach", type=float, default=600.0, help="mm; farthest the wrap may pull an interior vertex")
ap.add_argument("--plane-tol", type=float, default=3.0, help="mm; the wrap counterpart counts as a plane when 90 % of projected points are within this")
ap.add_argument("--floor-z", type=float, help="mm; the flat-floor assumption of the wrap stage. A rim lying below it whose "
                     "footprint is over 1 m² is extruded up to this plane and closed with one planar face")
ap.add_argument("--max-fold", type=float, default=0.10, help="skip a cap whose share of dihedral angles over 120 deg exceeds this")
ap.add_argument("--list-only", action="store_true")
ap.add_argument("--no-sew", action="store_true", help="add the caps as loose faces (for inspection)")
ap.add_argument("--caps-stl", type=Path, help="also write every cap as one STL (inspection)")
args = ap.parse_args()

import trimesh  # noqa: E402
import triangle as tr  # noqa: E402
from OCP.BRepAdaptor import BRepAdaptor_Curve  # noqa: E402
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Sewing  # noqa: E402
from OCP.BRep import BRep_Builder  # noqa: E402
from OCP.BRepTools import BRepTools_WireExplorer  # noqa: E402
from OCP.GCPnts import GCPnts_QuasiUniformDeflection  # noqa: E402
from OCP.gp import gp_Pnt  # noqa: E402
from OCP.TopAbs import TopAbs_REVERSED  # noqa: E402
from OCP.TopoDS import TopoDS_Compound  # noqa: E402

t0 = time.time()
shape, cad_report = cad.read_step(args.step)
if shape is None:
    raise SystemExit(f"STEP 읽기 실패: {cad_report.warnings}")
cad.diagnose(shape, cad_report)
holes, dangling = brep.free_boundaries(shape)
print(f"STEP {args.step.name}: 면 {cad_report.faces:,}  자유경계 고리 {len(holes)} (열린 사슬 {len(dangling)})  {time.time()-t0:.0f}s")


def sample_wire(wire):
    """Ordered points along the wire's edges, on the curves, chord error <= deflection."""
    pts = []
    exp = BRepTools_WireExplorer(wire)
    while exp.More():
        edge = exp.Current()
        curve = BRepAdaptor_Curve(edge)
        sampler = GCPnts_QuasiUniformDeflection(curve, args.deflection)
        seg = []
        if sampler.IsDone():
            for i in range(1, sampler.NbPoints() + 1):
                p = sampler.Value(i)
                seg.append((p.X(), p.Y(), p.Z()))
        else:
            for u in (curve.FirstParameter(), curve.LastParameter()):
                p = curve.Value(u)
                seg.append((p.X(), p.Y(), p.Z()))
        if edge.Orientation() == TopAbs_REVERSED:
            seg.reverse()
        if pts and np.linalg.norm(np.subtract(seg[0], pts[-1])) < 1e-6:
            seg = seg[1:]
        pts.extend(seg)
        exp.Next()
    P = np.asarray(pts, float)
    if len(P) > 1 and np.linalg.norm(P[0] - P[-1]) < 1e-6:
        P = P[:-1]
    # drop consecutive duplicates
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]
    return P[keep]


def split_at_symmetry(P, tol=1.0):
    """A half model's free boundary runs along the symmetry cut; cut it there.

    Returns closed loops: each run of off-plane points, with the plane points at
    its two ends kept, closed by the straight segment between those ends (which
    lies in the plane, so the mirror closes it later).
    """
    on = np.abs(P[:, 1]) < tol
    if not on.any() or on.all():
        return [P], False
    start = int(np.flatnonzero(on)[0])
    Q = np.roll(P, -start, axis=0)
    on = np.roll(on, -start)
    pieces, run = [], []
    n = len(Q)
    for i in range(n):
        if on[i]:
            if run:
                pieces.append([Q[(run[0] - 1) % n]] + [Q[j] for j in run] + [Q[i]])
                run = []
        else:
            run.append(i)
    if run:
        pieces.append([Q[(run[0] - 1) % n]] + [Q[j] for j in run] + [Q[0]])
    out = []
    for piece in pieces:
        R = np.asarray(piece, float)
        keep = np.r_[True, np.linalg.norm(np.diff(R, axis=0), axis=1) > 1e-6]
        R = R[keep]
        if len(R) >= 3:
            out.append(R)
    return out, True


loops = []
for boundary, wire in holes:
    P = sample_wire(wire)
    if len(P) < 3:
        continue
    sym = bool(np.abs(P[:, 1]).max() < 1.0)
    pieces, was_split = split_at_symmetry(P)
    for R in pieces:
        per = float(np.linalg.norm(np.roll(R, -1, axis=0) - R, axis=1).sum())
        if per < args.min_loop:
            continue
        loops.append({"points": R, "perimeter": per, "size": boundary.size, "centre": R.mean(0),
                      "extent": R.max(0) - R.min(0), "symmetry_only": sym, "edges": boundary.edges,
                      "split": was_split})
loops.sort(key=lambda L: -L["perimeter"])
if args.max_loops:
    loops = loops[:args.max_loops]
for k, L in enumerate(loops):
    print(f"  고리 {k}: 둘레 {L['perimeter']:7.0f} mm  모서리 {L['edges']:4d}  표본 {len(L['points']):5d}  중심 {L['centre'].round(0).tolist()}  크기 {L['extent'].round(0).tolist()}"
          + ("  [대칭면만, 건너뜀]" if L["symmetry_only"] else "") + ("  [대칭면에서 분할]" if L.get("split") else ""))
if args.list_only:
    sys.exit(0)

wrap = trimesh.load(args.wrap, force="mesh")
wrap.merge_vertices()
print(f"랩 {args.wrap.name}: 삼각형 {len(wrap.faces):,}")


def planar_cdt(P, normal, target_edge):
    """CDT of the loop projected along `normal`; None if the projection is not a simple polygon."""
    from shapely.geometry import Polygon
    from shapely.validation import explain_validity
    n = len(P)
    c = P.mean(0)
    e1 = np.cross(normal, [1, 0, 0]) if abs(normal[0]) < 0.9 else np.cross(normal, [0, 1, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    uv = np.column_stack([(P - c) @ e1, (P - c) @ e2])
    poly = Polygon(uv)
    if not poly.is_valid or poly.area < 1e-6:
        return None, explain_validity(poly)
    area = 0.433 * target_edge ** 2
    seg = np.column_stack([np.arange(n), (np.arange(n) + 1) % n])
    try:
        t = tr.triangulate({"vertices": uv, "segments": seg}, f"pq25a{area:.1f}Yj")
    except Exception as exc:
        return None, f"triangle: {exc}"
    if t is None or "triangles" not in t or len(t["triangles"]) == 0:
        return None, "triangle: empty"
    V2 = t["vertices"]
    tris = np.asarray(t["triangles"], int)
    V3 = c + np.outer(V2[:, 0], e1) + np.outer(V2[:, 1], e2)
    V3[:n] = P
    return (V3, tris), "ok"


def simplify_loop(P, max_points):
    """Douglas-Peucker on a closed 3D loop until at most max_points remain (indices)."""
    n = len(P)
    if n <= max_points:
        return np.arange(n)
    # start from the two farthest-apart points
    i0 = 0
    i1 = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
    keep = {i0, i1}

    def seg_error(a, b):
        idx = list(range(a + 1, b)) if b > a else list(range(a + 1, n)) + list(range(0, b))
        if not idx:
            return -1.0, -1
        pa, pb = P[a], P[b]
        d = pb - pa
        L = np.linalg.norm(d)
        pts = P[idx]
        if L < 1e-9:
            dist = np.linalg.norm(pts - pa, axis=1)
        else:
            t = np.clip((pts - pa) @ d / (L * L), 0, 1)
            dist = np.linalg.norm(pts - (pa + np.outer(t, d)), axis=1)
        j = int(np.argmax(dist))
        return float(dist[j]), idx[j]

    while len(keep) < max_points:
        order = sorted(keep)
        best = (-1.0, -1)
        for a, b in zip(order, order[1:] + order[:1]):
            err, j = seg_error(a, b)
            if err > best[0]:
                best = (err, j)
        if best[1] < 0 or best[0] < 1e-6:
            break
        keep.add(best[1])
    return np.array(sorted(keep))


def reinsert_boundary(tris, keep, n):
    """Put the boundary points dropped by simplify_loop back: fan each chord."""
    out = list(tris)
    keep = list(keep)
    for a, b in zip(keep, keep[1:] + keep[:1]):
        between = list(range(a + 1, b)) if b > a else list(range(a + 1, n)) + list(range(0, b))
        if not between:
            continue
        # the triangle that uses chord (a, b) (in either direction)
        hit = None
        for k, t in enumerate(out):
            if a in t and b in t:
                hit = k
                break
        if hit is None:
            continue
        t = out.pop(hit)
        x = [v for v in t if v not in (a, b)][0]
        chain = [a] + between + [b]
        # keep the winding of the original triangle
        forward = (t.index(a) + 1) % 3 == t.index(b)
        for u, v in zip(chain, chain[1:]):
            out.append((u, v, x) if forward else (v, u, x))
    return out


plane_faces = []   # (loop index, TopoDS face) planar floors built by build_plane_extrude


def build_plane_extrude(P, c, normal):
    """Skirt from the rim to the plane (c, normal) plus one planar face on the projected rim."""
    from shapely.geometry import Polygon
    n = len(P)
    h = (P - c) @ normal
    Pp = P - np.outer(h, normal)
    e1 = np.cross(normal, [1, 0, 0]) if abs(normal[0]) < 0.9 else np.cross(normal, [0, 1, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    uv = np.column_stack([(Pp - c) @ e1, (Pp - c) @ e2])
    if not Polygon(uv).is_valid:
        from shapely.validation import explain_validity
        print(f"     [평면 압출] 투영 다각형 무효: {explain_validity(Polygon(uv))[:80]}")
        return None
    V = np.vstack([P, Pp])
    tris = []
    for i in range(n):
        j = (i + 1) % n
        if abs(h[i]) < 0.5 and abs(h[j]) < 0.5:
            continue
        tris.append((i, j, n + j))
        tris.append((i, n + j, n + i))
    poly = BRepBuilderAPI_MakePolygon()
    for q in Pp:
        poly.Add(gp_Pnt(float(q[0]), float(q[1]), float(q[2])))
    poly.Close()
    if not poly.IsDone():
        return None
    face = BRepBuilderAPI_MakeFace(poly.Wire(), True)
    if not face.IsDone():
        return None
    plane_faces.append(face.Face())
    if not tris:
        # nothing to extrude: the plane face alone closes the loop
        return trimesh.Trimesh(V[:n], np.zeros((0, 3), int), process=False)
    return trimesh.Trimesh(V, np.asarray(tris, int), process=False)


def floor_extrude(P, floor_z):
    """The wrap's flat-floor assumption as B-rep: rim -> vertical skirt -> planar floor.

    Only for a rim that lies under the floor plane with a footprint over 1 m²; the
    footprint polygon is repaired (make_valid, largest piece) because a sill rim
    projected to plan view has a few sub-millimetre crossings.
    """
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.validation import make_valid
    # the rim may climb above the floor plane near the wheel arches and at the
    # symmetry-plane tips; the skirt then runs down to the floor there
    if np.median(P[:, 2]) > floor_z - 10.0:
        return None, None, None
    poly = make_valid(Polygon(P[:, :2]))
    if isinstance(poly, MultiPolygon) or poly.geom_type == "GeometryCollection":
        polys = [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"]
        if not polys:
            return None, None, None
        poly = max(polys, key=lambda g: g.area)
    if poly.geom_type != "Polygon" or poly.area < 1e6:
        return None, None, None
    n = len(P)
    top = np.column_stack([P[:, 0], P[:, 1], np.full(n, floor_z)])
    V = np.vstack([P, top])
    tris = []
    for i in range(n):
        j = (i + 1) % n
        if abs(P[i, 2] - floor_z) < 0.5 and abs(P[j, 2] - floor_z) < 0.5:
            continue
        if abs(P[i, 1]) < 1.0 and abs(P[j, 1]) < 1.0:
            # the closing chord along the symmetry plane: a skirt on it would be a fin
            # under the car's centreline; the mirror closes that plane
            continue
        tris.append((i, j, n + j))
        tris.append((i, n + j, n + i))
    ring = np.asarray(poly.exterior.coords)[:-1]
    mp = BRepBuilderAPI_MakePolygon()
    for q in ring:
        mp.Add(gp_Pnt(float(q[0]), float(q[1]), float(floor_z)))
    mp.Close()
    if not mp.IsDone():
        return None, None, None
    face = BRepBuilderAPI_MakeFace(mp.Wire(), True)
    if not face.IsDone():
        return None, None, None
    plane_faces.append(face.Face())
    moved = np.zeros(len(V))
    moved[:n] = np.abs(P[:, 2] - floor_z)
    return trimesh.Trimesh(V, np.asarray(tris, int), process=False), f"floor-extrude(z={floor_z:.0f}, {poly.area/1e6:.2f} m²)", moved


def cap_loop(P, target_edge, reach):
    """Triangulate the loop (several projections, then 3D), refine, lift interior vertices onto the wrap."""
    n = len(P)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    normal = vt[2]
    method, why = None, []
    V3 = tris = None
    for label, nrm in (("cdt-pca", normal), ("cdt-z", np.array([0, 0, 1.0])), ("cdt-x", np.array([1.0, 0, 0])), ("cdt-y", np.array([0, 1.0, 0]))):
        res, reason = planar_cdt(P, nrm, target_edge)
        if res is not None:
            V3, tris = res
            method = label
            break
        why.append(f"{label}: {reason[:40]}")
    if V3 is None:
        # Liepa's 3D minimum-weight triangulation is O(n^3), so run it on a
        # simplified copy of the loop (<= 80 points) and put the dropped boundary
        # points back by fanning the triangles that sit on the simplified chords
        keep = simplify_loop(P, 80)
        outer = np.tile(normal, (len(keep), 1))
        tris0, _ = fair.min_weight_triangulation(P[keep], outer)
        if tris0:
            tris0 = [tuple(int(keep[i]) for i in t) for t in tris0]
            tris0 = reinsert_boundary(tris0, keep, n)
            pts, tris1 = fair.refine(list(P), tris0, n, target_edge)
            V3, tris = np.asarray(pts, float), np.asarray(tris1, int)
            method = f"mwt({len(keep)})"
    if V3 is None:
        tris0 = fair.ear_clip(P, normal)
        if tris0:
            pts, tris1 = fair.refine(list(P), tris0, n, target_edge)
            V3, tris = np.asarray(pts, float), np.asarray(tris1, int)
            method = "ear"
    if V3 is None or tris is None or len(tris) == 0:
        return None, "; ".join(why) or "no triangulation", None
    interior = np.arange(n, len(V3))
    moved = np.zeros(len(V3))
    flat = V3.copy()
    if len(interior):
        # closest point on the wrap (rays along one normal hit the wrong sheet on a
        # loop that wanders in z: the underbody rim sits 120 mm under the floor and
        # ray hits on the wheel-house closures made 20 m2 of zig-zag out of a 4 m2 cap),
        # then alternate umbrella smoothing of the interior with re-projection
        from scipy import sparse
        rows_, cols_ = [], []
        for a, b, c_ in tris:
            for u_, v_ in ((a, b), (b, c_), (c_, a)):
                rows_ += [u_, v_]
                cols_ += [v_, u_]
        A = sparse.coo_matrix((np.ones(len(rows_)), (rows_, cols_)), shape=(len(V3), len(V3))).tocsr()
        A.data[:] = 1.0
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        free = np.zeros(len(V3), bool)
        free[interior] = True
        for it in range(4):
            cp, d, _ = trimesh.proximity.closest_point(wrap, V3[interior])
            ok = d <= reach
            V3[interior[ok]] = cp[ok]
            if it == 3:
                moved[interior] = np.linalg.norm(V3[interior] - flat[interior], axis=1)
                break
            for _ in range(3):
                avg = (A @ V3) / deg[:, None]
                V3[free] = 0.5 * V3[free] + 0.5 * avg[free]
    cap = trimesh.Trimesh(V3, tris, process=False)
    return cap, method, moved

builder = BRep_Builder()
caps = TopoDS_Compound()
builder.MakeCompound(caps)
rows = []
all_caps = []
n_faces = 0
for k, L in enumerate(loops):
    if L["symmetry_only"]:
        continue
    t1 = time.time()
    target_edge = args.edge if args.edge > 0 else max(15.0, L["perimeter"] / 150.0)
    reach = min(args.reach, L["perimeter"] / 4.0)
    cap = None
    if args.floor_z is not None:
        cap, method, moved = floor_extrude(L["points"], args.floor_z)
    if cap is None:
        cap, method, moved = cap_loop(L["points"], target_edge, reach)
    # If the wrap put a plane where this loop's interior is (the flat floor), build the
    # cap as the rim extruded to that plane plus one planar face: a triangulation
    # dragged onto the wrap folds where the rim hangs 120 mm below the floor
    plane_cap = None
    if cap is not None and len(cap.vertices) > len(L["points"]) + 20:
        Q = cap.vertices[len(L["points"]):]
        # dominant plane with inliers: points the projection dropped onto the rocker
        # skins or wheel-house closures must not hide the floor
        # start from the loop's own normal and the densest height along it (a PCA of
        # all projected points tilts toward the rocker skins and finds nothing)
        P0 = L["points"]
        _, _, vl = np.linalg.svd(P0 - P0.mean(0), full_matrices=False)
        n0 = vl[2]
        hgt = Q @ n0
        bins = np.arange(hgt.min(), hgt.max() + args.plane_tol, args.plane_tol)
        hist, edges_ = np.histogram(hgt, bins=bins)
        peak = edges_[int(np.argmax(hist))] + args.plane_tol / 2
        inl = np.abs(hgt - peak) < args.plane_tol
        nq, cq = n0, Q[inl].mean(0) if inl.any() else Q.mean(0)
        for _ in range(3):
            if inl.sum() < 10:
                break
            cq = Q[inl].mean(0)
            _, sv, vq = np.linalg.svd(Q[inl] - cq, full_matrices=False)
            nq = vq[2]
            resid = np.abs((Q - cq) @ nq)
            inl = resid < args.plane_tol
        share = float(inl.mean())
        rim_dist = (L["points"] - cq) @ nq
        if L["perimeter"] > 5000:
            print(f"     [평면 검사] 내부점 {len(Q)}  평면 안 {share*100:.0f} %  법선 {nq.round(2).tolist()}  평면점 {cq.round(0).tolist()}  테두리 거리 중앙 {np.median(np.abs(rim_dist)):.0f} mm")
        if share >= 0.6 and np.median(np.abs(rim_dist)) > 4 * args.plane_tol:
            plane_cap = build_plane_extrude(L["points"], cq, nq)
            if plane_cap is not None:
                cap, method, moved = plane_cap, f"plane-extrude({share*100:.0f}%)", np.abs(rim_dist)
    if cap is None:
        print(f"  캡 {k}: 둘레 {L['perimeter']:.0f}  삼각화 실패({method}) — 건너뜀")
        rows.append({"loop": k, "perimeter_mm": round(L["perimeter"]), "failed": True})
        continue
    disc = L["perimeter"] ** 2 / (4 * np.pi)
    ratio = (cap.area / disc if disc > 0 else 0) if len(cap.faces) else 0.0
    ang = np.degrees(cap.face_adjacency_angles) if len(cap.faces) and len(cap.face_adjacency) else np.zeros(0)
    folded = float(np.mean(ang > 120.0)) if len(ang) else 0.0
    if folded > args.max_fold and not method.startswith("floor-extrude"):
        print(f"  캡 {k}: 둘레 {L['perimeter']:.0f}  접힘 {folded*100:.1f} % > {args.max_fold*100:.0f} % — 뜻이 필요한 자리, 건너뜀")
        rows.append({"loop": k, "perimeter_mm": round(L["perimeter"]), "centre": L["centre"].round(0).tolist(),
                     "skipped": "folded", "folded_share": round(folded, 3), "method": method})
        continue
    all_caps.append(cap)
    for a, b, c in cap.vertices[cap.faces]:
        poly = BRepBuilderAPI_MakePolygon(gp_Pnt(*map(float, a)), gp_Pnt(*map(float, b)), gp_Pnt(*map(float, c)), True)
        if poly.IsDone():
            f = BRepBuilderAPI_MakeFace(poly.Wire(), True)
            if f.IsDone():
                builder.Add(caps, f.Face())
                n_faces += 1
    row = {"loop": k, "perimeter_mm": round(L["perimeter"]), "centre": L["centre"].round(0).tolist(),
           "extent": L["extent"].round(0).tolist(), "method": method, "target_edge_mm": round(target_edge, 1),
           "triangles": int(len(cap.faces)), "area_m2": round(float(cap.area) / 1e6, 4), "area_over_disc": round(float(ratio), 2),
           "lift_p50_mm": round(float(np.median(moved[moved > 0])) if (moved > 0).any() else 0.0, 1),
           "lift_max_mm": round(float(moved.max()), 1), "folded_share": round(folded, 3)}
    rows.append(row)
    print(f"  캡 {k}: 둘레 {row['perimeter_mm']:6d}  {method}  변 {target_edge:.0f} mm  삼각형 {row['triangles']:,}  면적 {row['area_m2']:.3f} m² (원판 대비 {ratio:.2f})  "
          f"랩으로 이동 p50 {row['lift_p50_mm']} / 최대 {row['lift_max_mm']} mm  접힘(>120°) {folded*100:.1f} %  {time.time()-t1:.0f}s")

for f in plane_faces:
    builder.Add(caps, f)
    n_faces += 1
print(f"캡 면 {n_faces:,}개 (평면 바닥 {len(plane_faces)}개 포함)")
if args.caps_stl and all_caps:
    trimesh.util.concatenate(all_caps).export(args.caps_stl)
    print(f"캡 STL: {args.caps_stl}")
if args.no_sew:
    out = TopoDS_Compound()
    builder.MakeCompound(out)
    builder.Add(out, shape)
    builder.Add(out, caps)
    result = out
else:
    t1 = time.time()
    sewer = BRepBuilderAPI_Sewing(float(args.sew), True, True, True, False)
    sewer.Add(shape)
    sewer.Add(caps)
    sewer.Perform()
    result = sewer.SewedShape()
    print(f"봉합(허용오차 {args.sew} mm): {time.time()-t1:.0f}s")

after_holes, after_dangling = brep.free_boundaries(result)
rep2 = cad.CadReport() if hasattr(cad, "CadReport") else None
cad.diagnose(result, rep2) if rep2 is not None else None
faces_after = rep2.faces if rep2 is not None else -1
big_after = [b for b, w in after_holes if b.size >= args.min_loop]
print(f"결과: 면 {faces_after:,}  자유경계 고리 {len(after_holes)} (크기 ≥ {args.min_loop:.0f} mm: {len(big_after)}; 이전 {len(loops)})  "
      f"열린 사슬 {len(after_dangling)}  bbox {[round(v) for v in rep2.bbox] if rep2 is not None else '?'}")
for b in big_after[:10]:
    print(f"    남은 고리: 크기 {b.size:.0f} 둘레 {b.length:.0f} 중심 {[round(v) for v in b.centre]}")
ok = brep.write_step(result, args.out, units="MM" if cad_report.units_hint == "mm" else "M")
print(f"{'저장' if ok else '실패'}: {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)  총 {time.time()-t0:.0f}s")
if args.report:
    args.report.write_text(json.dumps({"step": str(args.step), "wrap": str(args.wrap), "caps": rows, "cap_faces": n_faces,
                                       "faces_after": faces_after, "free_loops_before": len(loops),
                                       "free_loops_after": len(after_holes), "big_loops_after": len(big_after),
                                       "remaining": [{"size": round(b.size), "length": round(b.length), "centre": [round(v) for v in b.centre]} for b in big_after]},
                                      indent=1, ensure_ascii=False))
