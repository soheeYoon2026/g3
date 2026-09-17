"""Read-only measurements the model may call, with their schemas.

    geometry_tools.py --list                    # the function schemas, as JSON
    geometry_tools.py --call measure_mesh --args '{"path": "x.stl"}'

Every function here only measures: it opens files and returns numbers. None of them writes
geometry or starts a pipeline, so a wrong call costs time, not a result. Writing steps stay
with the rules in plan_geometry.py. The schemas are strict (no extra properties) so the
model's arguments can be checked before anything runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

MAX_BYTES = 400 * 1024 * 1024


def _load(path, merge=True):
    import trimesh
    p = Path(path).expanduser()   # 문서 예시가 ~ 경로를 쓴다
    if not p.exists():
        raise FileNotFoundError(f"파일이 없습니다: {p}")
    if p.stat().st_size > MAX_BYTES:
        raise ValueError(f"파일이 {p.stat().st_size/1e6:.0f} MB 로 너무 큽니다")
    m = trimesh.load(p, force="mesh")
    if merge:
        m.merge_vertices()
    return m


def measure_mesh(path: str) -> dict:
    """삼각형 수, 치수, 수밀 여부, 경계 모서리 비율, 몸체 수, 체적, 표면적."""
    import trimesh
    m = _load(path)
    _, counts = np.unique(m.edges_sorted, axis=0, return_counts=True)
    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    bodies = len(trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(m.faces))))
    lo, hi = m.bounds
    return {"file": str(path), "triangles": int(len(m.faces)), "vertices": int(len(m.vertices)),
            "extents_mm": [round(float(v), 1) for v in m.extents],
            "bbox_min_mm": [round(float(v), 1) for v in lo], "bbox_max_mm": [round(float(v), 1) for v in hi],
            "boundary_edges": boundary, "boundary_edge_share": round(boundary / max(1, len(counts)), 5),
            "nonmanifold_edges": nonmanifold, "bodies": bodies,
            "watertight": bool(boundary == 0 and nonmanifold == 0),
            "volume_m3": round(abs(float(m.volume)) / 1e9, 4),
            "area_m2": round(float(m.area) / 1e6, 3),
            "fill_of_bbox": round(abs(float(m.volume)) / max(1e-9, float(np.prod(hi - lo))), 3)}


def list_openings(path: str, top: int = 12) -> dict:
    """열린 고리(구멍)를 큰 것부터: 크기, 중심, 둘레."""
    from aox_g3 import fair
    m = _load(path)
    loops, _ = fair.boundary_loops(m)
    rows = []
    for lp in sorted(loops, key=lambda l: -np.ptp(m.vertices[l], axis=0).max())[: int(top)]:
        P = m.vertices[lp]
        rows.append({"size_mm": round(float(np.ptp(P, axis=0).max()), 1),
                     "centre_mm": [round(float(v), 1) for v in P.mean(0)],
                     "perimeter_mm": round(float(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum()), 1),
                     "points": int(len(lp))})
    return {"file": str(path), "open_loops": len(loops), "largest": rows}


def _same_frame(a, b) -> dict:
    """두 메쉬가 같은 좌표계·같은 단위인가. 아니면 거리 측정이 그럴듯한 쓰레기가 된다.

    오늘(2026-09-15~16) 이 계열로 두 번 당했다: 통제기가 자기 출력을 다시 환산해 4.6 m 차가 117 m 가 됐고,
    CAS-A 뷰어 메쉬가 캡 이전 단계라 표식이 엉뚱한 자리를 가리켰다. 값이 안 튀고 조용히 틀린다.

    **통과했다고 맞는 것은 아니다.** 경고가 비어 있으면 "확인됐다" 가 아니라 "이 세 갈래(단위·좌표계·단계)로는
    안 걸린다" 로 읽어야 한다. 같은 자리·같은 단위·같은 단계이면서 내용이 틀린 경우는 여기서 안 잡힌다.
    """
    ea, eb = a.extents, b.extents
    ratio = float(max(ea.max() / max(eb.max(), 1e-9), eb.max() / max(ea.max(), 1e-9)))
    lo = np.maximum(a.bounds[0], b.bounds[0])
    hi = np.minimum(a.bounds[1], b.bounds[1])
    overlap = float(np.prod(np.clip(hi - lo, 0, None)) / max(np.prod(ea), np.prod(eb), 1e-9))
    note = ""
    if ratio > 1.2:
        note = f"크기가 {ratio:.2f} 배 다릅니다 — 단위나 축이 다를 수 있습니다"
    elif overlap < 0.5:
        note = f"경계상자가 {overlap*100:.0f} %만 겹칩니다 — 다른 좌표계이거나 다른 단계의 형상일 수 있습니다"
    return {"extent_ratio": round(ratio, 3), "bbox_overlap": round(overlap, 3), "frame_warning": note}


def compare_to_reference(candidate: str, reference: str, web_distance_mm: float = 1.5, samples: int = 200000) -> dict:
    """닫은 형상이 원본에서 얼마나 떨어졌는지와, 틈을 건너뛰어 덧댄 면적·패치 목록."""
    import igl
    import trimesh
    ref = _load(reference)
    cand = _load(candidate)
    frame = _same_frame(cand, ref)
    RV = np.ascontiguousarray(ref.vertices, np.float64)
    RF = np.ascontiguousarray(ref.faces, np.int64)
    P = cand.sample(int(samples))
    d = np.sqrt(igl.point_mesh_squared_distance(np.ascontiguousarray(P, np.float64), RV, RF)[0])
    C = np.ascontiguousarray(cand.triangles_center, np.float64)
    dc = np.sqrt(igl.point_mesh_squared_distance(C, RV, RF)[0])
    A = cand.area_faces
    gap = dc > float(web_distance_mm)
    patches = []
    if gap.any():
        adj = cand.face_adjacency
        both = gap[adj[:, 0]] & gap[adj[:, 1]]
        labels = trimesh.graph.connected_component_labels(adj[both], node_count=len(cand.faces))
        labels[~gap] = -1
        ids, inv = np.unique(labels[gap], return_inverse=True)
        areas = np.bincount(inv, weights=A[gap])
        for k in np.argsort(areas)[::-1][:8]:
            sel = np.nonzero(labels == ids[k])[0]
            if areas[k] < 1e4:
                break
            patches.append({"area_cm2": round(float(areas[k]) / 100, 0),
                            "centre_mm": [round(float(v), 0) for v in (C[sel] * A[sel, None]).sum(0) / A[sel].sum()],
                            "extent_mm": [round(float(v), 0) for v in (C[sel].max(0) - C[sel].min(0))],
                            "bridged_gap_mm": round(2.0 * float(dc[sel].max()), 0)})
    # a median far from zero means these are not the same surface: a different stage of the
    # same model, or a different model. Bounding boxes can still look fine (CAS-A, 2026-09-16:
    # 82 % overlap, same size, median 115 mm apart because the viewer mesh predated the caps).
    diag = float(np.linalg.norm(ref.extents))
    p50 = float(np.percentile(d, 50))
    if p50 > 0.005 * diag and not frame["frame_warning"]:
        frame["frame_warning"] = (f"중앙 이탈 {p50:.1f} mm 가 대각선의 {p50/diag*100:.1f} % 입니다 — "
                                  "같은 형상의 다른 단계이거나 다른 형상일 수 있습니다")
    return {"candidate": str(candidate), "reference": str(reference), **frame,
            "distance_p50_mm": round(float(np.percentile(d, 50)), 2),
            "distance_p90_mm": round(float(np.percentile(d, 90)), 2),
            "distance_p99_mm": round(float(np.percentile(d, 99)), 1),
            "added_area_share": round(float(A[gap].sum() / A.sum()), 4),
            "added_patches": patches}


def section_profile(reference: str, candidate: str, axis: str, value_mm: float, web_distance_mm: float = 1.5) -> dict:
    """한 단면에서 원본에 붙은 길이와 틈을 건너뛴 길이."""
    import igl
    axis_i = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    ref = _load(reference)
    cand = _load(candidate)
    RV = np.ascontiguousarray(ref.vertices, np.float64)
    RF = np.ascontiguousarray(ref.faces, np.int64)

    def cut(mesh):
        normal = np.zeros(3); normal[axis_i] = 1.0
        origin = np.zeros(3); origin[axis_i] = float(value_mm)
        path = mesh.section(plane_origin=origin, plane_normal=normal)
        if path is None:
            return np.zeros((0, 2, 3))
        V = np.asarray(path.vertices, float)
        segs = [np.stack([V[np.asarray(e.points, int)[:-1]], V[np.asarray(e.points, int)[1:]]], axis=1) for e in path.entities]
        return np.concatenate(segs) if segs else np.zeros((0, 2, 3))

    cs = cut(cand)
    if not len(cs):
        return {"axis": axis, "value_mm": value_mm, "candidate_length_mm": 0.0, "bridged_length_mm": 0.0,
                "note": "이 평면에서 후보 단면이 비어 있습니다"}
    mid = cs.mean(axis=1)
    d = np.sqrt(igl.point_mesh_squared_distance(np.ascontiguousarray(mid, np.float64), RV, RF)[0])
    length = np.linalg.norm(np.diff(cs, axis=1), axis=2).ravel()
    bridged = d > float(web_distance_mm)
    return {"axis": axis, "value_mm": float(value_mm),
            "candidate_length_mm": round(float(length.sum()), 0),
            "bridged_length_mm": round(float(length[bridged].sum()), 0),
            "bridged_share": round(float(length[bridged].sum() / max(1e-9, length.sum())), 3),
            "reference_length_mm": round(float(np.linalg.norm(np.diff(cut(ref), axis=1), axis=2).sum()), 0)}


def closed_openings(reference: str, candidate: str, keep_above_mm: float,
                    web_distance_mm: float = 1.5, tolerance_mm: float = 1.5) -> dict:
    """지키기로 한 크기보다 큰 구멍이 닫혔는지. 판정은 덧댄 재료가 건너뛴 틈으로 한다.

    찢어진 경계(열린 고리)만 보면 헛돈다: car5 원본은 열린 고리가 0개이고 윙 슬롯은 판과 판 사이를
    지나는 통로라, 그것을 다 막은 10 mm 랩도 "보존됨"으로 나왔다(2026-09-14 확인). 랩이 무언가를
    닫으면 원본에서 떨어진 면(덧댄 재료)이 생기고, 그 패치가 건너뛴 틈이 곧 닫힌 구멍의 크기다.
    """
    added = compare_to_reference(candidate, reference, web_distance_mm=web_distance_mm)
    # the bridged gap is estimated as twice the farthest point of the patch, which is coarse to
    # about a millimetre, so a patch within tolerance of the threshold is reported but not failed
    tol = float(tolerance_mm)
    closed = [p for p in added["added_patches"] if p["bridged_gap_mm"] >= float(keep_above_mm) + tol]
    borderline = [p for p in added["added_patches"]
                  if float(keep_above_mm) - tol <= p["bridged_gap_mm"] < float(keep_above_mm) + tol]
    # torn boundaries, kept as a second signal for meshes that do have them
    before = list_openings(reference, top=200)
    after = list_openings(candidate, top=200)
    wanted = [r for r in before["largest"] if r["size_mm"] >= keep_above_mm]
    kept = [r for r in after["largest"] if r["size_mm"] >= keep_above_mm]
    lost = []
    for w in wanted:
        c = np.array(w["centre_mm"])
        if not [k for k in kept if np.linalg.norm(np.array(k["centre_mm"]) - c) < max(200.0, w["size_mm"])]:
            lost.append(w)
    return {"keep_above_mm": float(keep_above_mm), "tolerance_mm": tol,
            "closed_by_added_material": closed[:10], "borderline": borderline[:10],
            "added_area_share": added["added_area_share"],
            "open_loops_in_reference": len(wanted), "still_open_in_candidate": len(kept),
            "lost_boundary_loops": lost[:10],
            "ok": not closed and not lost,
            "how": "덧댄 패치가 건너뛴 틈이 keep_above_mm 이상이면 닫힌 것으로 본다"}


def _silhouette_m2(mesh, faces_mask=None, pixel_mm: float = 2.0) -> float:
    """yz 평면에 래스터해 겹침 없는 투영 면적. 법선 합은 안 쓴다(속면이 이중 계산된다)."""
    T = mesh.triangles[:, :, 1:] if faces_mask is None else mesh.triangles[faces_mask][:, :, 1:]
    if not len(T):
        return 0.0
    lo = T.reshape(-1, 2).min(axis=0)
    hi = T.reshape(-1, 2).max(axis=0)
    h = float(pixel_mm)
    nx, ny = int((hi[0] - lo[0]) / h) + 2, int((hi[1] - lo[1]) / h) + 2
    grid = np.zeros((nx, ny), bool)
    for tri in T:   # 삼각형 경계상자 채우기 = 보수적 상한
        a = ((tri[:, 0].min() - lo[0]) / h, (tri[:, 0].max() - lo[0]) / h)
        b = ((tri[:, 1].min() - lo[1]) / h, (tri[:, 1].max() - lo[1]) / h)
        grid[int(a[0]):int(a[1]) + 1, int(b[0]):int(b[1]) + 1] = True
    return float(grid.sum() * h * h / 1e6)


def region_profile(candidate: str, reference: str = "", bands: int = 8, axis: str = "x",
                   web_distance_mm: float = 1.5, pixel_mm: float = 2.0) -> dict:
    """흐름 방향 구간별 표면적과, 원본이 있으면 구간별로 덧댄 면적. 기저면 비도 함께.

    부위 마스크는 조용히 틀린다. 그래서 자체 검사를 붙였다: **기저면 투영이 정면 투영보다 크면
    마스크가 틀린 것**이다(옆 세션 2026-09-17: "뒤 1/4 안에서 n_x>0.7" 마스크가 경사면·언더바디를
    잡아 비가 3.59 로 나왔다. 1 을 넘을 수 없는 양이다).
    """
    import igl
    ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    cand = _load(candidate)
    C = np.asarray(cand.triangles_center, float)
    A = cand.area_faces
    N = np.asarray(cand.face_normals, float)
    lo, hi = cand.bounds
    span = hi[ax] - lo[ax]
    edges = np.linspace(lo[ax], hi[ax], int(bands) + 1)
    added = None
    if reference:
        ref = _load(reference)
        d = np.sqrt(igl.point_mesh_squared_distance(np.ascontiguousarray(C, np.float64),
                                                    np.ascontiguousarray(ref.vertices, np.float64),
                                                    np.ascontiguousarray(ref.faces, np.int64))[0])
        added = d > float(web_distance_mm)
    rows = []
    for i in range(int(bands)):
        sel = (C[:, ax] >= edges[i]) & (C[:, ax] < edges[i + 1] if i < bands - 1 else C[:, ax] <= edges[i + 1])
        row = {"band": i + 1, "from_mm": round(float(edges[i]), 0), "to_mm": round(float(edges[i + 1]), 0),
               "area_cm2": round(float(A[sel].sum()) / 100, 0)}
        if added is not None:
            row["added_cm2"] = round(float(A[sel & added].sum()) / 100, 1)
            row["added_share"] = round(float(A[sel & added].sum() / max(A[sel].sum(), 1e-9)), 4)
        rows.append(row)
    frontal = _silhouette_m2(cand, None, pixel_mm)
    rear = (C[:, ax] >= lo[ax] + 0.75 * span) & (N[:, ax] > 0.5)
    base = _silhouette_m2(cand, rear, pixel_mm)
    out = {"file": str(candidate), "bands": rows,
           "frontal_projection_m2": round(frontal, 4), "base_projection_m2": round(base, 4),
           "base_over_frontal": round(base / max(frontal, 1e-9), 3)}
    if out["base_over_frontal"] > 1.0:
        out["mask_error"] = ("기저면 투영이 정면 투영보다 큽니다 — 부위 마스크가 경사면이나 언더바디를 "
                             "잡고 있습니다. 이 값을 쓰지 마세요")
    if added is not None:
        tot = float(A[added].sum())
        thirds = []
        for name, m in (("front_quarter", C[:, ax] < lo[ax] + 0.25 * span),
                        ("middle_half", (C[:, ax] >= lo[ax] + 0.25 * span) & (C[:, ax] < lo[ax] + 0.75 * span)),
                        ("rear_quarter", C[:, ax] >= lo[ax] + 0.75 * span)):
            thirds.append({"where": name, "added_cm2": round(float(A[m & added].sum()) / 100, 0),
                           "share_of_added": round(float(A[m & added].sum() / max(tot, 1e-9)), 3)})
        out["added_by_region"] = thirds
        out["added_area_share_total"] = round(float(tot / A.sum()), 4)
    return out


TOOLS = {
    "measure_mesh": (measure_mesh, {
        "path": {"type": "string", "description": "STL/OBJ 경로"}}, ["path"]),
    "list_openings": (list_openings, {
        "path": {"type": "string"}, "top": {"type": "integer", "description": "큰 것부터 몇 개 (기본 12)"}}, ["path"]),
    "compare_to_reference": (compare_to_reference, {
        "candidate": {"type": "string", "description": "닫은 형상"},
        "reference": {"type": "string", "description": "원본"},
        "web_distance_mm": {"type": "number", "description": "이보다 멀면 덧댄 면 (기본 1.5)"},
        "samples": {"type": "integer"}}, ["candidate", "reference"]),
    "section_profile": (section_profile, {
        "reference": {"type": "string"}, "candidate": {"type": "string"},
        "axis": {"type": "string", "enum": ["x", "y", "z"]},
        "value_mm": {"type": "number"}, "web_distance_mm": {"type": "number"}},
        ["reference", "candidate", "axis", "value_mm"]),
    "region_profile": (region_profile, {
        "candidate": {"type": "string"}, "reference": {"type": "string", "description": "있으면 구간별 덧댄 면적도 낸다"},
        "bands": {"type": "integer"}, "axis": {"type": "string", "enum": ["x", "y", "z"]},
        "web_distance_mm": {"type": "number"}, "pixel_mm": {"type": "number"}}, ["candidate"]),
    "closed_openings": (closed_openings, {
        "reference": {"type": "string"}, "candidate": {"type": "string"},
        "keep_above_mm": {"type": "number", "description": "이 크기 이상은 열린 채로 남아야 한다"},
        "web_distance_mm": {"type": "number"}, "tolerance_mm": {"type": "number"}},
        ["reference", "candidate", "keep_above_mm"]),
}


def schemas():
    out = []
    for name, (fn, props, required) in TOOLS.items():
        out.append({"type": "function", "function": {
            "name": name, "description": (fn.__doc__ or "").strip().splitlines()[0],
            "parameters": {"type": "object", "properties": props, "required": required,
                           "additionalProperties": False}}})
    return out


def call(name: str, arguments: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"그런 기능이 없습니다: {name}", "available": list(TOOLS)}
    fn, props, required = TOOLS[name]
    missing = [r for r in required if r not in arguments]
    if missing:
        return {"error": f"빠진 인자: {missing}"}
    extra = [k for k in arguments if k not in props]
    if extra:
        return {"error": f"모르는 인자: {extra}"}
    t0 = time.time()
    try:
        result = fn(**arguments)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "tool": name, "arguments": arguments}
    result["_seconds"] = round(time.time() - t0, 1)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--call")
    ap.add_argument("--args", default="{}")
    a = ap.parse_args()
    if a.list:
        print(json.dumps(schemas(), ensure_ascii=False, indent=1))
    elif a.call:
        print(json.dumps(call(a.call, json.loads(a.args)), ensure_ascii=False, indent=1))
    else:
        ap.print_help()
