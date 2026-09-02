"""Keep only the surface the external flow can actually reach.

A production car model carries geometry the outside air never touches: wheel-arch
liners, cabin trim, engine-bay parts. On CAS-A that shows up as 30.4 m² of surface
where a car's skin is 12-20, and it is why alpha wrap returns a shell holding 3% of
the bounding box - the wrap reaches those inner panels through the openings and
faithfully wraps them too.

The fix is the step Flexcompute calls "Remove hidden geometry": flood fill the
empty space from outside the bounding box, then keep the faces that touch the
filled region and drop the rest. Its one real parameter is the same one they
expose as "Min passage size" - the voxel pitch decides how narrow an opening the
fill can pass, so a gap thinner than the pitch reads as closed and whatever hides
behind it is classified as interior.

Nothing here is CFD-specific: it is a reachability question about a surface.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import trimesh


@dataclass
class VisibilityReport:
    pitch: float = 0.0
    min_passage: float = 0.0
    grid_shape: tuple = ()
    wall_voxels: int = 0
    outside_voxels: int = 0
    faces_in: int = 0
    faces_kept: int = 0
    faces_removed: int = 0
    area_in: float = 0.0
    area_kept: float = 0.0
    components_kept: int = 0
    warnings: list = None

    def as_dict(self):
        d = asdict(self)
        d["grid_shape"] = list(self.grid_shape)
        return d


def _voxelise(mesh, pitch, origin, shape):
    """Mark every voxel a triangle passes through.

    trimesh's own voxeliser rasterises through its cached scene graph and is slow
    on a six-figure face count; walking each triangle's own bounding box is direct
    and needs no extra dependency.
    """
    wall = np.zeros(shape, dtype=bool)
    triangles = np.asarray(mesh.triangles, dtype=float)
    lo = np.floor((triangles.min(axis=1) - origin) / pitch).astype(np.int64)
    hi = np.floor((triangles.max(axis=1) - origin) / pitch).astype(np.int64)
    np.clip(lo, 0, np.array(shape) - 1, out=lo)
    np.clip(hi, 0, np.array(shape) - 1, out=hi)
    for (i0, j0, k0), (i1, j1, k1) in zip(lo, hi):
        wall[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = True
    return wall


def remove_hidden(mesh, min_passage: Optional[float] = None,
                  dilate: int = 1, keep_largest_only: bool = False):
    """Drop faces the outside cannot reach. Returns (mesh, report)."""
    from scipy import ndimage

    report = VisibilityReport(warnings=[])
    report.faces_in = int(len(mesh.faces))
    report.area_in = float(mesh.area)

    lo = np.asarray(mesh.bounds[0], dtype=float)
    hi = np.asarray(mesh.bounds[1], dtype=float)
    diag = float(np.linalg.norm(hi - lo))
    if min_passage is None:
        # 1/300 of the diagonal is about 17 mm on a 5 m car: narrow enough to keep
        # a wheel arch or a duct open, wide enough that a panel gap reads as closed
        min_passage = diag / 300.0
    pitch = float(min_passage)
    report.min_passage = min_passage
    report.pitch = pitch

    pad = 3 * pitch
    origin = lo - pad
    shape = tuple(int(np.ceil((hi[i] + pad - origin[i]) / pitch)) + 1 for i in range(3))
    if np.prod(shape) > 6e8:
        report.warnings.append(f"grid {shape} is very large; raise min_passage")
    report.grid_shape = shape

    wall = _voxelise(mesh, pitch, origin, shape)
    if dilate > 0:
        # Close pinholes left by the tessellation so the fill does not leak into
        # a cavity through a defect that is not a real opening
        wall = ndimage.binary_dilation(wall, iterations=dilate)
    report.wall_voxels = int(wall.sum())

    free = ~wall
    labels, count = ndimage.label(free)
    if count == 0:
        report.warnings.append("no free space — pitch is too coarse")
        return mesh, report
    # The component touching the padded boundary is the outside
    outside_label = labels[0, 0, 0]
    if outside_label == 0:
        report.warnings.append("the box corner is inside the wall — cannot seed outside")
        return mesh, report
    outside = labels == outside_label
    if dilate > 0:
        # Undo the dilation so faces sitting on the true surface still touch it
        outside = ndimage.binary_dilation(outside, iterations=dilate)
    report.outside_voxels = int(outside.sum())

    centres = np.asarray(mesh.triangles_center, dtype=float)
    idx = np.floor((centres - origin) / pitch).astype(np.int64)
    np.clip(idx, 0, np.array(shape) - 1, out=idx)

    # A face counts as reachable when the outside region touches the voxel it sits
    # in or any of the 26 around it; a face lies inside a wall voxel by definition,
    # so only its neighbours can carry the answer.
    reach = ndimage.binary_dilation(outside, iterations=1)
    keep = reach[idx[:, 0], idx[:, 1], idx[:, 2]]

    report.faces_kept = int(keep.sum())
    report.faces_removed = int(report.faces_in - report.faces_kept)
    if report.faces_kept == 0:
        report.warnings.append("every face was classified as hidden — check min_passage")
        return mesh, report

    out = trimesh.Trimesh(vertices=mesh.vertices,
                          faces=np.asarray(mesh.faces)[keep], process=False)
    out.remove_unreferenced_vertices()

    if keep_largest_only:
        pieces = out.split(only_watertight=False)
        if len(pieces) > 1:
            out = max(pieces, key=lambda p: p.area)
            report.warnings.append(
                f"kept only the largest of {len(pieces)} components")
    try:
        report.components_kept = len(out.split(only_watertight=False))
    except Exception:
        report.components_kept = -1
    report.area_kept = float(out.area)
    return out, report
