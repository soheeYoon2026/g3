"""Detect a half model's symmetry plane and mirror it back to a full body.

A half car cannot be sealed by any wrapper: the symmetry plane is a hole the size
of the whole silhouette, so every tier walks in through it and hollows the result.
On the CAS-A CATIA export all three tiers failed exactly this way (alpha wrap
returned 1% of the volume), and the parameters were never the problem.

Detection rather than a hardcoded y=0, because the plane's axis and offset are a
CAD convention that varies by supplier. The test is the one that actually matters:
essentially all geometry sits on one side of the plane, and a large share of the
open boundary lies *in* it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import trimesh


@dataclass
class SymmetryReport:
    is_half: bool = False
    axis: int = -1
    axis_name: str = ""
    plane: float = 0.0
    boundary_on_plane: int = 0
    boundary_total: int = 0
    boundary_fraction: float = 0.0
    one_sided_fraction: float = 0.0
    plane_span: list = None
    silhouette_fraction: float = 0.0
    mirrored_faces: int = 0
    welded_vertices: int = 0
    reason: str = ""

    def as_dict(self):
        return asdict(self)


def boundary_vertices(mesh) -> np.ndarray:
    """Indices of vertices touching an edge used by exactly one face."""
    faces = np.asarray(mesh.faces)
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    counts = Counter(map(tuple, edges))
    open_edges = np.array([k for k, v in counts.items() if v == 1], dtype=np.int64)
    if len(open_edges) == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(open_edges.ravel())


def detect(mesh, tol: Optional[float] = None) -> SymmetryReport:
    """Find the symmetry plane of a half model, or report that there is none."""
    report = SymmetryReport()
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) == 0:
        report.reason = "empty mesh"
        return report

    extents = vertices.max(axis=0) - vertices.min(axis=0)
    tol = tol if tol is not None else float(np.linalg.norm(extents)) * 1e-3

    open_idx = boundary_vertices(mesh)
    report.boundary_total = int(len(open_idx))
    if report.boundary_total == 0:
        report.reason = "mesh is already closed"
        return report
    open_pts = vertices[open_idx]

    best = None
    for axis in range(3):
        # A symmetry plane is flat in one axis and spans the other two, so look
        # for a plane position where the open boundary piles up.
        values = open_pts[:, axis]
        for candidate in (0.0, float(np.median(values)),
                          float(vertices[:, axis].min()), float(vertices[:, axis].max())):
            on_plane = np.abs(values - candidate) < tol
            count = int(on_plane.sum())
            if count < 20:
                continue
            # Nearly everything must lie on one side, or it is not a boundary plane
            below = float((vertices[:, axis] < candidate - tol).mean())
            above = float((vertices[:, axis] > candidate + tol).mean())
            one_sided = max(below, above)
            if one_sided < 0.95:
                continue
            others = [a for a in range(3) if a != axis]
            span = [float(open_pts[on_plane, a].max() - open_pts[on_plane, a].min())
                    for a in others]
            # The loop has to be silhouette-sized; a small flat patch is not it
            silhouette = float(np.mean([span[i] / extents[others[i]] for i in range(2)]))
            if silhouette < 0.7:
                continue
            score = (count / report.boundary_total) * silhouette * one_sided
            if best is None or score > best[0]:
                best = (score, axis, candidate, count, one_sided, span, silhouette)

    if best is None:
        report.reason = "no plane holds the open boundary — not a half model"
        return report

    _, axis, plane, count, one_sided, span, silhouette = best
    report.is_half = True
    report.axis = axis
    report.axis_name = "xyz"[axis]
    report.plane = plane
    report.boundary_on_plane = count
    report.boundary_fraction = count / report.boundary_total
    report.one_sided_fraction = one_sided
    report.plane_span = [round(s, 1) for s in span]
    report.silhouette_fraction = silhouette
    report.reason = (f"open boundary lies in {report.axis_name}={plane:.1f}, spanning "
                     f"{silhouette:.0%} of the silhouette")
    return report


def mirror(mesh, report: SymmetryReport, weld_tol: Optional[float] = None):
    """Reflect a half model across its plane and weld the seam.

    The reflection has determinant -1, so the copy's winding is reversed to keep
    normals pointing the same way; without that the two halves disagree and every
    downstream sign test (winding number, volume) is meaningless.
    """
    if not report.is_half:
        return mesh, report

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    axis, plane = report.axis, report.plane

    reflected = vertices.copy()
    reflected[:, axis] = 2.0 * plane - reflected[:, axis]

    merged_vertices = np.vstack([vertices, reflected])
    # reversed winding on the mirrored copy
    merged_faces = np.vstack([faces, faces[:, ::-1] + len(vertices)])

    out = trimesh.Trimesh(vertices=merged_vertices, faces=merged_faces, process=False)
    before = len(out.vertices)
    if weld_tol is None:
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        weld_tol = float(np.linalg.norm(extents)) * 1e-5
    out.merge_vertices(digits_vertex=max(0, int(round(-np.log10(weld_tol)))))
    out.remove_unreferenced_vertices()

    report.mirrored_faces = int(len(out.faces))
    report.welded_vertices = int(before - len(out.vertices))
    return out, report


def make_full(mesh, tol: Optional[float] = None):
    """Detect and mirror in one call. Returns (mesh, report) unchanged if not half."""
    report = detect(mesh, tol=tol)
    if not report.is_half:
        return mesh, report
    return mirror(mesh, report)
