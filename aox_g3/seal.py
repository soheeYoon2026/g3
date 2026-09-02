"""Turn a dirty STL into a watertight one, independently of any solver.

This is the sealing cascade that lives inside the LES runner's `stl_loader.py`,
lifted out so anything can call it. That module imports jax at the top, so using
its cascade meant pulling in the whole solver stack; nothing here imports more
than numpy and trimesh at module level.

The cascade follows the "reconstruct, don't repair" line of Portaneri et al.,
"Alpha Wrapping with an Offset" (ACM TOG 2022): a triangle soup with holes, gaps,
self-intersections and non-manifold edges is not patched up, a new closed surface
is built around it. Three tiers, in order of how little they change the shape:

  1. OpenVDB level set (`vdb_tool`), offset chosen per shape by bisection.
     Preferred because it closes gaps at a shorter distance than Alpha Wrap
     (24 mm vs 29 mm on the GT-R), which keeps narrow features like a wing
     support alive.
  2. CGAL `alpha_wrap_3`. Takes a triangle soup, so it handles non-manifold
     input the other tiers choke on -- on one GT-R model two faces shared a
     half-edge in the same direction, and Alpha Wrap was the only thing that
     loaded it at all.
  3. Voxel flood-fill seal (`fix_shell`, Warp GPU distance field). Last because
     its closing distance is coarse enough to bridge a wing to the body.

**Every tier reports whether its tool was even available.** The v8 label campaign
lost seven LES runs because `vdb_tool` was missing on the host, tier 1 silently
fell through, and the open-shell result looked plausible enough that nobody
noticed until the run-to-run scatter came back four times too large. A caller
that cannot see which tier ran cannot detect that, so `seal_report` always says.

Constants carry their provenance in comments; they were measured, not chosen.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import trimesh

# A sealed result whose volume collapses relative to the input is a hollow shell,
# not a seal: the closing distance was smaller than the holes. Measured on a golf
# head where sealing took 95.1 cc to 10.5 cc and flow passed straight through.
SEAL_MIN_VOLUME_RATIO = 0.5

# Alpha Wrap: alpha's lower bound is set by the input's hole size — go below it
# and the algorithm walks in through a hole and carves out the interior.
# GT-R 2026-08-21: alpha 29 mm gave 6.65 m3, 17 mm gave 1.03 m3, 10 mm gave 0.63.
# diag/180 is the practical floor found there; CGAL recommends offset as a small
# fraction of alpha.
AW_ALPHA_DIV = 180.0
AW_OFFSET_FRAC = 1.0 / 30.0

# OpenVDB: the offset both jumps holes and fills narrow gaps, so smaller is better
# but it must exceed the holes. The floor is shape-dependent, hence the bisection.
# GT-R 2026-08-21 at a fixed 6 mm voxel: 36 mm gave volume ratio 0.541 (fine),
# 24 mm gave 0.535 (the floor), 18 mm gave 0.085 (a shell).
VDB_D_HI_DIV = 100.0
VDB_D_LO_DIV = 600.0
VDB_BISECT = 4
VDB_VOXEL_FRAC = 0.25

# Openness = boundary length / sqrt(area); dimensionless, so shape-size invariant.
# Below the threshold a generalised winding number handles the holes without
# touching the geometry, which is always better than sealing. Measured:
#   welded GT-R      16 boundary edges   openness  0.01   GWN fine
#   DrivAer notchback 3,641              openness  2.47   GWN fine (24k steps)
#   raw Audi/GT-R    55,514              openness 62.33   GWN fails, diverged
# 5.0 is the threshold; 5.0-62.33 is still an unmeasured band.
OPENNESS_SEAL_THRESHOLD = 5.0
OPENNESS_UNVERIFIED_HI = 62.33


@dataclass
class SealReport:
    """What happened, in enough detail that a silent downgrade is visible."""

    watertight_in: bool = False
    watertight_out: bool = False
    openness_in: float = 0.0
    openness_out: float = 0.0
    boundary_edges_in: int = 0
    boundary_edges_out: int = 0
    faces_in: int = 0
    faces_out: int = 0
    volume_in: float = 0.0
    volume_out: float = 0.0
    volume_ratio: float = 0.0
    method: str = "none"
    tools_available: dict = field(default_factory=dict)
    tiers_attempted: list = field(default_factory=list)
    normals_flipped: bool = False
    warnings: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def mesh_openness(mesh) -> tuple[float, int]:
    """Boundary length over sqrt(area), plus the boundary-edge count."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(faces) == 0:
        return 0.0, 0
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    counts = Counter(map(tuple, edges))
    boundary = np.array([k for k, v in counts.items() if v == 1])
    if len(boundary) == 0:
        return 0.0, 0
    length = float(np.linalg.norm(vertices[boundary[:, 0]] - vertices[boundary[:, 1]], axis=1).sum())
    area = float(mesh.area)
    return (length / np.sqrt(area) if area > 0 else 0.0), len(boundary)


def available_tools() -> dict:
    """Which tiers can actually run here. Check this before trusting a result."""
    tools = {"vdb_tool": shutil.which("vdb_tool") is not None}
    try:
        from CGAL import CGAL_Alpha_wrap_3  # noqa: F401
        tools["cgal_alpha_wrap"] = True
    except Exception:
        tools["cgal_alpha_wrap"] = False
    try:
        from fix_shell_stl import seal_mesh  # noqa: F401
        tools["fix_shell"] = True
    except Exception:
        tools["fix_shell"] = False
    return tools


def _fix_orientation(mesh, report: SealReport):
    """Flip inward normals. Signed volume < 0 is the test.

    vdb_tool's ls2mesh emits inward normals: the magnitude is right and only the
    sign is wrong, which inverts the winding number everywhere and leaves the
    solver with no solid at all. Watertightness and winding consistency both pass,
    so this is the only check that catches it.
    """
    try:
        volume = float(mesh.volume)
    except Exception:
        return mesh
    if volume < 0:
        try:
            trimesh.repair.fix_normals(mesh)
            report.normals_flipped = True
        except Exception:
            report.warnings.append("normal flip failed")
    return mesh


# A body that encloses nothing fills a few percent of its box; a car fills about
# half. Measured: carA_base, a solved G2 surface, comes to 55.8% by voxel count and
# 60.6% by divergence, the two agreeing because it is genuinely closed.
SEAL_MIN_BOX_FILL = 0.15


def _volume_ok(sealed, original) -> tuple[bool, float]:
    """Is the sealed result a body, or a shell wrapped around a sheet?

    Comparing against the input's volume only works when the input is closed. For
    an open mesh `.volume` is a divergence-theorem artefact that can land anywhere:
    on CAS-A it read 8.96 m3, a plausible-looking 59% of the bounding box, while a
    voxel flood fill from outside reached every interior cell - the surface encloses
    nothing at all. Trusting that number as a denominator would pass or fail a seal
    for no reason, so an open input is judged on how much of its own bounding box
    the result fills instead.
    """
    try:
        new = abs(float(sealed.volume))
    except Exception:
        return True, 0.0

    if not bool(getattr(original, "is_watertight", False)):
        extents = np.asarray(sealed.extents, dtype=float)
        box = float(extents[0] * extents[1] * extents[2])
        if box <= 0.0:
            return True, 0.0
        fill = new / box
        return fill >= SEAL_MIN_BOX_FILL, fill

    try:
        old = abs(float(original.volume))
    except Exception:
        return True, 0.0
    if old <= 0.0:
        return True, 0.0
    ratio = new / old
    return ratio >= SEAL_MIN_VOLUME_RATIO, ratio


def _vdb_once(stl_path, offset, voxel, out_path):
    """One vdb_tool invocation -> (mesh, volume fill ratio) or None.

    mesh2ls alone gives a narrow band around the surface, which for an open mesh
    is still a shell. iso2ls builds the distance-`offset` isosurface first, which
    IS closed, and only then do inside and outside mean anything; flood fills the
    interior and erode walks the surface back onto the original.
    """
    radius = offset / voxel
    width = radius + 4
    cmd = ["vdb_tool", "-quiet", "-read", f"files={stl_path}",
           "-mesh2ls", f"voxel={voxel}", f"width={width}",
           "-iso2ls", f"iso={offset}", f"width={width}", "-flood",
           "-erode", f"radius={radius}", "-ls2mesh", "adapt=0.25",
           "-write", f"files={out_path}"]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=900)
    except Exception:
        return None
    if result.returncode != 0 or not os.path.exists(out_path):
        return None
    try:
        mesh = trimesh.load(out_path, force="mesh")
    except Exception:
        return None
    # ls2mesh leaves zero-volume shards at the band edge; keep the body only
    parts = sorted(mesh.split(only_watertight=False), key=lambda p: abs(p.volume), reverse=True)
    if not parts:
        return None
    body = parts[0]
    extents = np.asarray(body.extents, dtype=float)
    box = float(extents[0] * extents[1] * extents[2])
    return body, (abs(body.volume) / box if box > 0 else 0.0)


def _try_vdb(mesh, report: SealReport):
    report.tiers_attempted.append("vdb")
    if not report.tools_available.get("vdb_tool"):
        report.warnings.append("vdb_tool not installed — tier 1 skipped")
        return None
    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, "in.stl")
        mesh.export(src)
        diag = float(np.linalg.norm(np.asarray(mesh.extents, dtype=float)))
        hi, lo = diag / VDB_D_HI_DIV, diag / VDB_D_LO_DIV
        first = _vdb_once(src, hi, hi * VDB_VOXEL_FRAC, os.path.join(tmp, "hi.obj"))
        if not first or first[1] < SEAL_MIN_VOLUME_RATIO:
            # No unit is printed: the cascade is scale-invariant (everything is a
            # fraction of the diagonal) and the mesh may be in metres or in
            # millimetres. The LES copy hard-codes "mm" and misreports metre meshes.
            report.warnings.append(
                f"vdb failed even at its largest offset ({hi:.4g}, diagonal/{VDB_D_HI_DIV:.0f}) "
                f"— the holes are larger than this cascade closes")
            return None
        best = first[0]
        for i in range(VDB_BISECT):
            mid = (lo + hi) / 2.0
            attempt = _vdb_once(src, mid, mid * VDB_VOXEL_FRAC, os.path.join(tmp, f"b{i}.obj"))
            if attempt and attempt[1] >= SEAL_MIN_VOLUME_RATIO:
                best, hi = attempt[0], mid
            else:
                lo = mid
        return _fix_orientation(best, report)
    except Exception as exc:
        report.warnings.append(f"vdb raised {type(exc).__name__}: {exc}")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _try_alpha_wrap(mesh, report: SealReport, alpha_div=AW_ALPHA_DIV):
    report.tiers_attempted.append("alpha_wrap")
    if not report.tools_available.get("cgal_alpha_wrap"):
        report.warnings.append("CGAL alpha_wrap not installed — tier 2 skipped")
        return None
    try:
        from CGAL import CGAL_Alpha_wrap_3 as AW
        from CGAL.CGAL_Kernel import Point_3
        from CGAL.CGAL_Polyhedron_3 import Polyhedron_3

        diag = float(np.linalg.norm(np.asarray(mesh.extents, dtype=float)))
        alpha = diag / float(alpha_div)
        offset = alpha * AW_OFFSET_FRAC

        points = AW.Point_3_Vector()
        points.reserve(len(mesh.vertices))
        for v in np.asarray(mesh.vertices, dtype=float):
            points.append(Point_3(float(v[0]), float(v[1]), float(v[2])))
        polygons = AW.Polygon_Vector()
        polygons.reserve(len(mesh.faces))
        for f in np.asarray(mesh.faces):
            indices = AW.Int_Vector()
            indices.reserve(3)
            for k in f:
                indices.append(int(k))
            polygons.append(indices)

        wrapped = Polyhedron_3()
        AW.alpha_wrap_3(points, polygons, alpha, offset, wrapped)
        handle, tmp = tempfile.mkstemp(suffix=".off")
        os.close(handle)
        wrapped.write_to_file(tmp)
        out = trimesh.load(tmp, force="mesh")
        os.unlink(tmp)
        return _fix_orientation(out, report)
    except Exception as exc:
        report.warnings.append(f"alpha_wrap raised {type(exc).__name__}: {exc}")
        return None


def _try_voxel_seal(mesh, report: SealReport, pitch=None, close=None):
    report.tiers_attempted.append("fix_shell")
    if not report.tools_available.get("fix_shell"):
        report.warnings.append("fix_shell not importable — tier 3 skipped")
        return None
    try:
        from fix_shell_stl import seal_mesh
    except Exception as exc:
        report.warnings.append(f"fix_shell import failed: {type(exc).__name__}")
        return None
    for multiplier in (1, 3):
        closing = close * multiplier if close is not None else (None if multiplier == 1 else 9)
        try:
            sealed = seal_mesh(mesh, pitch=pitch, close=closing, verbose=False)
        except Exception as exc:
            report.warnings.append(f"fix_shell raised {type(exc).__name__}: {exc}")
            return None
        ok, ratio = _volume_ok(sealed, mesh)
        if ok:
            return _fix_orientation(sealed, report)
        report.warnings.append(f"fix_shell x{multiplier} gave a hollow result "
                               f"(volume ratio {ratio:.2f})")
    return None


def seal(mesh, force: bool = False) -> tuple[Optional[trimesh.Trimesh], SealReport]:
    """Seal a mesh, returning (mesh or None, report).

    Returns None when every tier failed. That is deliberate: an earlier version of
    this pipeline fell back to the raw open mesh, the solver built a hollow shell,
    flow ran through the inside of the car, and it neither diverged nor looked
    wrong -- it just quietly produced an incorrect Cd for four hours.

    force=True seals even when the mesh is closed enough that a winding number
    would do; sealing inflates volume by a few percent and fills narrow gaps, so
    it is otherwise a last resort.
    """
    report = SealReport()
    report.tools_available = available_tools()
    report.faces_in = int(len(mesh.faces))
    report.watertight_in = bool(mesh.is_watertight)
    report.openness_in, report.boundary_edges_in = mesh_openness(mesh)
    try:
        report.volume_in = float(abs(mesh.volume))
    except Exception:
        report.volume_in = 0.0

    threshold = float(os.environ.get("AOX_SEAL_OPENNESS", OPENNESS_SEAL_THRESHOLD))
    if report.watertight_in and not force:
        report.method = "already_watertight"
        report.watertight_out = True
        report.faces_out = report.faces_in
        report.volume_out = report.volume_in
        report.volume_ratio = 1.0
        report.openness_out = report.openness_in
        return mesh, report
    if report.openness_in < threshold and not force:
        # A generalised winding number handles this without touching the shape
        report.method = "below_threshold"
        report.warnings.append(
            f"openness {report.openness_in:.2f} < {threshold} — a winding number "
            "suffices; sealing would change the geometry for nothing")
        report.faces_out = report.faces_in
        report.volume_out = report.volume_in
        report.volume_ratio = 1.0
        report.openness_out = report.openness_in
        return mesh, report
    if threshold <= report.openness_in < OPENNESS_UNVERIFIED_HI:
        report.warnings.append(
            f"openness {report.openness_in:.2f} sits in the unmeasured band "
            f"({threshold}-{OPENNESS_UNVERIFIED_HI}); sealing is untested here")

    for tier, attempt in (("vdb", _try_vdb),
                          ("alpha_wrap", _try_alpha_wrap),
                          ("fix_shell", _try_voxel_seal)):
        result = attempt(mesh, report)
        if result is None or not result.is_watertight:
            if result is not None:
                report.warnings.append(f"{tier} produced a non-watertight result")
            continue
        ok, ratio = _volume_ok(result, mesh)
        if not ok:
            report.warnings.append(f"{tier} gave a hollow result (volume ratio {ratio:.2f})")
            continue
        report.method = tier
        report.watertight_out = True
        report.faces_out = int(len(result.faces))
        report.volume_out = float(abs(result.volume))
        report.volume_ratio = ratio
        report.openness_out, report.boundary_edges_out = mesh_openness(result)
        return result, report

    report.method = "failed"
    report.warnings.append("every tier failed — do not feed this to a solver")
    return None, report


def seal_file(path, out_path=None, force: bool = False):
    """Seal an STL on disk. Returns (report, output path or None)."""
    mesh = trimesh.load(path, force="mesh")
    sealed, report = seal(mesh, force=force)
    if sealed is None:
        return report, None
    if out_path:
        sealed.export(out_path)
    return report, out_path
