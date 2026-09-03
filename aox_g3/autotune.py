"""Propose the pipeline's parameters from measurements of the model itself.

Every parameter the pipeline exposes was, on CAS-A, chosen by hand after a
measurement: the sewing ladder from a tolerance sweep, the sealing size from
looking at the hole sizes, the closed-rim points from finding the wheels, the
mirror from noticing the model stops at y=0. This module does those measurements
and writes the proposal down with its reasoning, so a new model gets the same
treatment without a person repeating the work - and so the person can still
overrule any of it.

What it does not do is decide intent. Whether the underbody should be closed or
a grille kept open is not in the geometry; the proposal marks those as questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


@dataclass
class Proposal:
    units: str = ""
    diagonal: float = 0.0
    half_model: bool = False
    symmetry_plane_y: Optional[float] = None
    sew_stages: list = field(default_factory=list)
    seal_below: float = 0.0
    close_near: list = field(default_factory=list)     # [x, y, z, r]
    stitch_tolerance: float = 0.0
    curvature_angle: float = 15.0
    rationale: list = field(default_factory=list)
    sweeps: dict = field(default_factory=dict)
    questions: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


# --------------------------------------------------------------------- units

def detect_units(diagonal: float):
    """A car's bounding-box diagonal is 4-7 m; the number says the unit."""
    if 3000 <= diagonal <= 8000:
        return "mm", f"대각선 {diagonal:.0f} — 차량 크기의 mm"
    if 120 <= diagonal <= 320:
        return "inch", f"대각선 {diagonal:.0f} — 차량 크기의 inch"
    if 3 <= diagonal <= 8:
        return "m", f"대각선 {diagonal:.2f} — 차량 크기의 m"
    return "unknown", f"대각선 {diagonal:.3g} — 차량 크기 범위 밖, 단위 판정 불가"


# ---------------------------------------------------------------- half model

def detect_half_model(points, axis: int = 1, one_sided: float = 0.9):
    """Is nearly everything on one side of a plane? Then it is a half model.

    CAS-A runs from y = -983 to +220: 92% of its vertices are on the negative
    side and the positive spill is centreline parts modelled whole. The plane is
    taken at 0 when 0 lies inside the span, otherwise at the near edge.
    """
    coords = np.asarray(points, dtype=float)[:, axis]
    negative = float((coords < 0).mean())
    positive = 1.0 - negative
    lo, hi = float(coords.min()), float(coords.max())
    span = hi - lo
    if negative >= one_sided or positive >= one_sided:
        side = "음" if negative >= one_sided else "양"
        plane = 0.0 if lo <= 0.0 <= hi else (hi if negative >= one_sided else lo)
        return True, plane, (f"정점의 {100 * max(negative, positive):.0f}%가 y {side}의 "
                             f"한쪽 — 절반 모델, 대칭면 y={plane:.0f}")
    return False, None, (f"y 양쪽에 {100 * negative:.0f}/{100 * positive:.0f}%로 "
                         f"분포 — 전체 모델")


# --------------------------------------------------------------- sewing sweep

def sweep_sewing(shape, bbox, diagonal, divisors=(20000, 10000, 5000, 2000, 1000, 500, 250)):
    """Sew cumulatively, fine to coarse, and watch free edges and invalid faces.

    Cumulative, not one pass per tolerance: sewing at 10.5 mm from scratch stamps
    that tolerance on every edge (96 invalid faces on CAS-A), while reaching
    10.5 mm through 1 and 5 mm first touches only what is still open (46). The
    ladder the pipeline runs is cumulative, so the sweep has to be too - the
    first version swept single passes and concluded that nothing past 1 mm was
    worth it.

    The ladder keeps every stage up to the last one before invalid faces climb
    past twice the minimum seen, provided the stage still removes free edges. A
    model with no free edges at the finest tolerance gets just that tolerance.
    """
    from aox_g3 import cad

    def measure(s):
        after = cad.CadReport()
        cad.diagnose(s, after)
        return after

    # Phase 1, single passes from scratch: where does sewing do the least
    # damage? Starting too fine is not free - at 0.26 mm CAS-A sews with 63
    # invalid faces, at 1.05 mm with 33 - so the ladder starts where invalid
    # faces are lowest, not at the finest tolerance available.
    single = []
    for d in divisors:
        tolerance = diagonal / d
        try:
            sewn, _ = cad.sew(shape, cad.CadReport(bbox=bbox), tolerance)
        except Exception as exc:
            single.append({"tolerance": round(tolerance, 4), "error": type(exc).__name__})
            continue
        after = measure(sewn)
        single.append({"tolerance": round(tolerance, 4), "shells": after.shells,
                       "free_edges": after.free_edges,
                       "invalid_faces": after.invalid_faces})
    good = [r for r in single if "error" not in r]
    rows = {"single": single, "cumulative": []}
    if not good:
        return [round(diagonal / 5000, 4), round(diagonal / 1000, 4),
                round(diagonal / 500, 4)], rows, "쓸기 실패 — 기본 사다리 사용"
    if good[0]["free_edges"] == 0:
        return [good[0]["tolerance"]], rows, \
            f"가장 고운 {good[0]['tolerance']:.3g}에서 이미 자유모서리 0 — 한 단계면 충분"

    min_invalid = min(r["invalid_faces"] for r in good)
    fine_ok = [r for r in good if r["invalid_faces"] <= max(min_invalid * 1.1, min_invalid + 2)]
    fine = max(fine_ok, key=lambda r: r["tolerance"])

    # Phase 2, cumulative from the fine start: how far can the tolerance climb
    # before it costs more than it closes? Reaching 10.5 mm through 1 and 5 mm
    # touches only what is still open (46 invalid on CAS-A); the same 10.5 mm
    # from scratch stamps every edge (96).
    current, _ = cad.sew(shape, cad.CadReport(bbox=bbox), fine["tolerance"])
    ceiling = max(2 * fine["invalid_faces"], fine["invalid_faces"] + 10)
    kept = [fine]
    for r in good:
        if r["tolerance"] <= fine["tolerance"]:
            continue
        try:
            trial, _ = cad.sew(current, cad.CadReport(bbox=bbox), r["tolerance"])
        except Exception:
            break
        after = measure(trial)
        row = {"tolerance": r["tolerance"], "shells": after.shells,
               "free_edges": after.free_edges, "invalid_faces": after.invalid_faces}
        rows["cumulative"].append(row)
        if row["invalid_faces"] > ceiling:
            break
        if row["free_edges"] >= kept[-1]["free_edges"]:
            continue    # a stage that closes nothing is not worth its slop
        kept.append(row)
        current = trial
    stages = [r["tolerance"] for r in kept]
    if len(stages) > 3:
        stages = [stages[0], stages[len(stages) // 2], stages[-1]]
    why = (f"단일 통과에서 무효면 최소({fine['invalid_faces']})인 {fine['tolerance']:.3g}에서 "
           f"시작, 누적으로 {kept[-1]['tolerance']:.3g}까지 "
           f"(무효면 {kept[-1]['invalid_faces']}, 자유모서리 {fine['free_edges']:,} → "
           f"{kept[-1]['free_edges']:,}); 다음 단계는 무효면이 {ceiling}개를 넘거나 닫는 것이 없음")
    return stages, rows, why


# --------------------------------------------------------------- sealing size

def propose_sealing_size(sizes, diagonal):
    """Put the sealing size in the widest gap of the hole-size distribution.

    Small holes are panel gaps and spoke slots; the large ones are missing
    parts. Between them there is usually a gap in size - on CAS-A the sizes run
    4851, 2537, then 871, 799, 785 ... - and the sealing size belongs in it. The
    proposal is the geometric mean across the widest log-gap among holes bigger
    than a fiftieth of the body; everything below it is closed automatically,
    everything above it becomes a question.
    """
    sizes = sorted((float(s) for s in sizes if s > diagonal / 50.0), reverse=True)
    if len(sizes) < 2:
        return diagonal / 100.0, "구멍이 2개 미만 — 대각선/100"
    logs = np.log(np.asarray(sizes))
    gaps = logs[:-1] - logs[1:]
    k = int(np.argmax(gaps))
    proposal = float(np.sqrt(sizes[k] * sizes[k + 1]))
    proposal = min(proposal, diagonal / 3.0)
    above = k + 1
    below = len(sizes) - above
    why = (f"크기 분포에서 가장 큰 간격은 {sizes[k]:.0f} ↔ {sizes[k + 1]:.0f} "
           f"(비 {np.exp(gaps[k]):.1f}배) — 그 사이 {proposal:.0f}: "
           f"이하 {below}개 자동 봉합, 초과 {above}개는 결정 항목")
    return proposal, why


# -------------------------------------------------------------------- wheels

def _loop_points(wire, per_edge=6):
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    from aox_g3.brep import _explore

    pts = []
    for e in _explore(wire, TopAbs_EDGE):
        edge = TopoDS.Edge_s(e)
        curve = BRepAdaptor_Curve(edge)
        first, last = BRep_Tool.Range_s(edge)
        for t in np.linspace(first, last, per_edge, endpoint=False):
            p = curve.Value(float(t))
            pts.append((p.X(), p.Y(), p.Z()))
    return np.asarray(pts, dtype=float)


def circularity(points):
    """4*pi*A/P^2 of the loop projected onto its own plane; 1 is a circle."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return 0.0
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    u, v = vt[0], vt[1]
    flat = np.stack([(pts - centre) @ u, (pts - centre) @ v], axis=1)
    # order by angle so the shoelace sees a polygon, not a scribble
    order = np.argsort(np.arctan2(flat[:, 1], flat[:, 0]))
    flat = flat[order]
    x, y = flat[:, 0], flat[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perimeter = float(np.linalg.norm(np.diff(np.vstack([flat, flat[:1]]), axis=0), axis=1).sum())
    return float(4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0


def detect_wheels(holes, bbox):
    """Circular loops, low, near either end of the body: wheel rims.

    A rim ring is round (circularity above 0.6), a tenth to a quarter of the
    body's length across, sits in the lower half, and is within a third of the
    length from the front or the back. Rings that pass are clustered by
    position; each cluster becomes one close-near point with a radius that
    covers its concentric rings and spoke gaps.
    """
    lo = np.asarray(bbox[:3], dtype=float)
    hi = np.asarray(bbox[3:], dtype=float)
    length = hi[0] - lo[0]
    height = hi[2] - lo[2]
    candidates = []
    for boundary, wire in holes:
        s = boundary.size
        if not (0.10 * length <= s <= 0.25 * length):
            continue
        c = np.asarray(boundary.centre, dtype=float)
        if c[2] > lo[2] + 0.5 * height:
            continue
        from_front = (c[0] - lo[0]) / length
        if not (from_front <= 0.35 or from_front >= 0.65):
            continue
        try:
            circ = circularity(_loop_points(wire))
        except Exception:
            continue
        if circ < 0.6:
            continue
        candidates.append((c, s, circ))

    clusters = []
    for c, s, circ in candidates:
        for cluster in clusters:
            if np.linalg.norm(cluster["centre"] - c) < 0.4 * max(s, cluster["size"]):
                cluster["members"].append((c, s, circ))
                cluster["centre"] = np.mean([m[0] for m in cluster["members"]], axis=0)
                cluster["size"] = max(cluster["size"], s)
                break
        else:
            clusters.append({"centre": c, "size": s, "members": [(c, s, circ)]})
    points, why = [], []
    for cluster in clusters:
        c = cluster["centre"]
        r = 0.6 * cluster["size"]
        points.append([round(float(c[0]), 1), round(float(c[1]), 1),
                       round(float(c[2]), 1), round(float(r), 1)])
        why.append(f"휠 후보: 원형 고리 {len(cluster['members'])}개, 최대 크기 "
                   f"{cluster['size']:.0f}, 중심 ({c[0]:.0f},{c[1]:.0f},{c[2]:.0f}), "
                   f"반경 {r:.0f}")
    return points, why


# --------------------------------------------------------------------- driver

def propose(shape, cad_report, do_sweep: bool = True):
    """Measure the model and write down what the pipeline should be told."""
    from aox_g3 import brep, cad
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.BRep import BRep_Tool
    from OCP.TopoDS import TopoDS

    proposal = Proposal()
    lo = np.asarray(cad_report.bbox[:3], dtype=float)
    hi = np.asarray(cad_report.bbox[3:], dtype=float)
    diag = float(np.linalg.norm(hi - lo))
    proposal.diagonal = round(diag, 1)

    proposal.units, why = detect_units(diag)
    proposal.rationale.append(f"단위: {why}")

    # Vertices are enough to judge sidedness
    pts = []
    for v in brep._explore(shape, TopAbs_VERTEX):
        p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(v))
        pts.append((p.X(), p.Y(), p.Z()))
    half, plane, why = detect_half_model(np.asarray(pts))
    proposal.half_model, proposal.symmetry_plane_y = half, plane
    proposal.rationale.append(f"대칭: {why}")

    if do_sweep:
        stages, rows, why = sweep_sewing(shape, cad_report.bbox, diag)
        proposal.sweeps["sewing"] = rows
    else:
        stages = [round(diag / 5000, 4), round(diag / 1000, 4), round(diag / 500, 4)]
        why = "쓸기 생략 — 기본 사다리 대각선/5000, /1000, /500"
    proposal.sew_stages = stages
    proposal.rationale.append(f"꿰맴 사다리 {stages}: {why}")

    sewn, _ = cad.sew_progressive(shape, cad.CadReport(bbox=cad_report.bbox),
                                  tolerances=stages)
    holes, _ = brep.free_boundaries(sewn)
    sizes = [b.size for b, _ in holes]
    proposal.sweeps["hole_sizes"] = [round(s, 1) for s in sorted(sizes, reverse=True)]
    seal, why = propose_sealing_size(sizes, diag)
    proposal.seal_below = round(seal, 1)
    proposal.rationale.append(f"봉합 크기 {seal:.0f}: {why}")

    points, whys = detect_wheels(holes, cad_report.bbox)
    proposal.close_near = points
    proposal.rationale += whys or ["휠 후보 없음 — close-near 제안 없음"]
    if points:
        proposal.questions.append(
            "휠을 닫힌 림으로 단순화할지, 열린 스포크가 필요한지 (Cd에 직접 영향)")

    proposal.stitch_tolerance = round(diag / 2500.0, 4)
    proposal.curvature_angle = 15.0
    big = [s for s in sizes if s > seal]
    if big:
        proposal.questions.append(
            f"봉합 크기 초과 개구부 {len(big)}개 (최대 {max(big):.0f}) — 닫을지, 무엇으로 닫을지")
    if half:
        proposal.questions.append(f"대칭면 y={plane:.0f} 이 맞는지 (모델이 y {lo[1]:.0f}~{hi[1]:.0f})")
    return proposal, sewn
