"""Repair the geometry as B-rep and hand back STEP, without going through triangles.

The mesh route - tessellate, then wrap or voxel-remesh - cannot return STEP. Its
output is a triangle soup, and turning that back into CAD means either writing one
planar face per triangle (a 137,000-face STEP no CAD system will open) or refitting
NURBS to it, which silently replaces the supplier's surfaces with an approximation.
A CAD engineer receiving that back cannot work with it.

Staying in B-rep avoids the round trip and, as it turns out, is also the better
diagnosis. The mesh route had to infer that CAS-A leaks - flood fill from outside,
notice it reaches everywhere, conclude there are openings wider than the voxel
pitch, and then have nothing to say about where. ShapeAnalysis_FreeBounds answers
the same question directly and instantly: 109 closed free boundaries, the largest
4.85 m across with a 16 m perimeter. No sealing tolerance measured in millimetres
was ever going to close that.

What the boundaries turn out to be matters more than their count. Sorted by size
they are the underbody, the cabin opening, the door glass, and the wheel arches -
an exterior surface model, missing the parts that are not styled surfaces. Those
are not defects to heal silently. Closing an underbody is a modelling decision (is
the floor flat or detailed?) and closing a cooling inlet is a mistake, which is why
this module fills what is below the sealing size and reports the rest with position
and size for the engineer to decide, rather than sealing everything it can reach.

Filling uses BRepOffsetAPI_MakeFilling, which builds a GeomPlate surface satisfying
the boundary as a constraint - the B-rep form of the Blend Surface step - with a
planar face as a fast path when the loop happens to be flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# A hole spanning d closes with something of order d^2. Measured on CAS-A's 42
# fillable holes the good patches land between 0.08 and 1.15 - every planar one at
# exactly 0.39 - and the runaways start at 1.90 and reach 204, where a single
# 871 mm hole produced 155 m2 of surface against a 15 m2 car. The gap between the
# two populations is wide and empty, so the cut sits in the middle of it.
MAX_PATCH_RATIO = 1.5

# Area is not enough. A patch can be perfectly sized and still be somewhere else:
# a 238 mm hole produced one reaching 9.3 m away - twice the length of the car - at
# an area ratio of 0.26, because a long thin sliver has hardly any area. Measuring
# instead how far the patch escapes the hole's own bounding box, relative to the
# hole's size, separates the two populations across an empty gap:
#
#     kept      0.00 0.00 0.01 ... 0.40 0.49 0.52
#     rejected  1.18 1.18 1.20 ... 3.23 7.44 17.90 42.04
#
# 0.75 sits in the gap and is defensible on its own: a cap over a hole of extent d
# is a hemisphere at worst, which rises d/2.
MAX_PATCH_REACH = 0.75

# A free boundary does not always bound an opening. Styling CAD models panels
# oversized and overlapping and trims them later, so a panel laid on another has a
# free loop the whole way round with solid surface underneath. CAS-A's rear side
# glass is exactly that - the loop sits 4 mm above a continuous panel, and looking
# through it shows body, not sky. Capping one welds a second skin a few millimetres
# above the first, which no rendering will show and which hands the mesher two
# surfaces where the flow sees one.
#
# The median distance to existing surface barely separates the two, because a real
# opening also has surface near its rim. What separates them is how much of the
# patch has something under it: measured on CAS-A the genuine holes run 21-47% and
# the overlaps 61-100%, with nothing in between.
#
# "Under" has to mean along the normal, not merely nearby. The hood's leading-edge
# slit taught this: a sliver patch has no interior, every sample sits millimetres
# from the slit's own walls, and plain distance called it 61% backed and refused to
# close it - the gap the user then photographed. An overlapping panel has surface
# behind it; a slit has surface beside it. So a sample only counts as backed when
# the nearest surface lies along the patch normal.
MAX_BACKED_FRACTION = 0.55
BACKING_PROBE_FRAC = 1.0 / 1000.0   # about 5 mm on a 5 m car
BACKING_NORMAL_DOT = 0.6            # how aligned with the normal "behind" must be

# A backed patch is only a second skin if it has real area. The hood's leading
# edge slit sits over a cowl panel 5 mm below along its normal - genuinely
# "backed", 96% - yet the cap bridging it is a strip at 0.09 of the hole's own
# scale, and a strip over a shallow step closes a panel gap rather than doubling a
# surface. Overlays that matter are wide: glass 0.33, wheel rims 0.39, the flank
# panel 0.19. Below this ratio the backing verdict is not applied.
SLIT_PATCH_RATIO = 0.12


@dataclass
class Boundary:
    """One free boundary: a hole, with enough about it to decide what to do."""
    size: float = 0.0          # bounding box diagonal of the loop
    length: float = 0.0        # perimeter
    centre: tuple = (0.0, 0.0, 0.0)
    bbox: tuple = ()           # (lo, hi) of the loop, for the reach test
    edges: int = 0
    planar: bool = False
    plane_origin: tuple = ()   # best-fit plane through the loop
    plane_normal: tuple = ()
    plane_residual: float = 0.0   # worst distance from the loop to that plane
    filled: bool = False
    fill_method: str = ""
    patch_area: float = 0.0
    patch_ratio: float = 0.0
    patch_reach: float = 0.0
    backed_fraction: float = 0.0
    note: str = ""

    def as_dict(self):
        d = asdict(self)
        d["centre"] = [round(float(v), 1) for v in self.centre]
        d["bbox"] = [[round(float(v), 1) for v in corner] for corner in self.bbox]
        return d


@dataclass
class HealReport:
    sealing_size: float = 0.0
    boundaries_found: int = 0
    boundaries_filled: int = 0
    boundaries_left: int = 0            # holes not filled — bookkeeping
    free_boundaries_after: int = 0      # loops the shape still has — measured
    floating_caps: int = 0              # caps placed but not joined (two loops each)
    faces_after_fine_sew: int = 0
    free_after_fine_sew: int = 0
    patch_sew_tolerance: float = 0.0
    faces_before: int = 0
    faces_after: int = 0
    shells_after: int = 0
    solids_after: int = 0
    closed: bool = False
    valid: bool = False
    volume: float = 0.0
    area_before: float = 0.0
    area: float = 0.0
    patch_area: float = 0.0
    step_written: str = ""
    filled: list = field(default_factory=list)
    left_open: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def as_dict(self):
        d = asdict(self)
        # Both lists matter. What was left open says what still needs a decision;
        # what was filled is the part nobody looks at until it is wrong.
        for key in ("filled", "left_open"):
            d[key] = [b.as_dict() if isinstance(b, Boundary) else b
                      for b in getattr(self, key)]
        return d


def _explore(shape, kind):
    from OCP.TopExp import TopExp_Explorer
    out, ex = [], TopExp_Explorer(shape, kind)
    while ex.More():
        out.append(ex.Current())
        ex.Next()
    return out


def _wire_metrics(wire):
    """Perimeter, extent and centre of a boundary loop.

    Length alone does not say how big an opening is: a long thin slot and a round
    port can share a perimeter. The bounding box diagonal of the loop is what
    decides whether it is a crack to close or a feature to keep.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_AbscissaPoint
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS

    length, points, n = 0.0, [], 0
    for e in _explore(wire, TopAbs_EDGE):
        edge = TopoDS.Edge_s(e)
        n += 1
        curve = BRepAdaptor_Curve(edge)
        try:
            length += GCPnts_AbscissaPoint.Length_s(curve)
        except Exception:
            pass
        first, last = BRep_Tool.Range_s(edge)
        for t in np.linspace(first, last, 5):
            p = curve.Value(float(t))
            points.append((p.X(), p.Y(), p.Z()))
    if not points:
        return (length, 0.0, np.zeros(3), n, False, (np.zeros(3), np.zeros(3)),
                np.array([0.0, 0.0, 1.0]), float("inf"))
    pts = np.asarray(points)
    centre = pts.mean(axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = float(np.linalg.norm(hi - lo))
    # The best-fit plane: its normal is the least-variance direction of the loop,
    # and the worst distance to it says whether capping on that plane is fair. On
    # CAS-A a 155 mm hole sits within 6.99 mm of its own plane, so this is a
    # millimetre-scale approximation on a five-metre car - nothing next to a patch
    # that reaches 9.3 m out of the same hole.
    centred = pts - centre
    normal = np.array([0.0, 0.0, 1.0])
    residual = float("inf")
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        normal = vt[2]
        residual = float(np.abs(centred @ normal).max())
    except Exception:
        pass
    planar = size > 0 and residual / max(size, 1e-9) < 1e-3
    return length, size, centre, n, planar, (lo, hi), normal, residual


def free_boundaries(shape, split_closed: bool = False):
    """Every free boundary in the shape, largest first.

    Closed wires are holes. Open wires - edges that do not even close into a loop -
    mean the topology is broken in a way filling cannot address, so they are
    reported separately rather than mixed in.
    """
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.TopAbs import TopAbs_WIRE
    from OCP.TopoDS import TopoDS

    finder = ShapeAnalysis_FreeBounds(shape, split_closed, False)
    holes, dangling = [], []
    for compound, bucket in ((finder.GetClosedWires(), holes),
                             (finder.GetOpenWires(), dangling)):
        for w in _explore(compound, TopAbs_WIRE):
            wire = TopoDS.Wire_s(w)
            (length, size, centre, n, planar, box,
             normal, residual) = _wire_metrics(wire)
            bucket.append((Boundary(size=size, length=length, centre=tuple(centre),
                                    bbox=(tuple(box[0]), tuple(box[1])),
                                    edges=n, planar=planar,
                                    plane_origin=tuple(centre),
                                    plane_normal=tuple(normal),
                                    plane_residual=residual), wire))
    holes.sort(key=lambda pair: -pair[0].size)
    dangling.sort(key=lambda pair: -pair[0].size)
    return holes, dangling


def _accept_patch(face, boundary: Boundary, method: str) -> bool:
    """Is this patch a repair, or has the surface solver run away?

    IsDone cannot be trusted here. MakeFilling reports success for boundaries it
    has not solved, and the failures are not subtle: on CAS-A it returned a 155 m2
    face to close an 871 mm hole in a 15 m2 car, and another with negative area,
    which means an inverted face. Left unchecked those go into the STEP and change
    the flow answer instead of fixing the model, so every patch is measured against
    the hole it claims to close before it is accepted.

    Two measurements are needed, not one. Area catches the patches that balloon.
    Distance catches the ones that are the right size in the wrong place, which
    area cannot see at all: a sliver reaching 9.3 m out of a 238 mm hole scored
    0.26 on area and looked healthy. Rendering the result is what exposed them -
    the healed body was 269 mm wider than the input and grew fins near the wheels -
    and with both tests 18 of the 32 patches that had been accepted are rejected.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    try:
        BRepGProp.SurfaceProperties_s(face, props)
        area = float(props.Mass())
    except Exception as exc:
        boundary.note = f"could not measure the patch ({type(exc).__name__})"
        return False

    boundary.patch_area = area
    scale = boundary.size ** 2
    boundary.patch_ratio = area / scale if scale else float("inf")

    if area <= 0.0:
        boundary.note = f"{method} returned a face of area {area:.4g} — inverted"
        return False
    if boundary.patch_ratio > MAX_PATCH_RATIO:
        boundary.note = (f"{method} patch is {boundary.patch_ratio:.1f}x the hole's "
                         f"own scale ({area / 1e6:.2f} m2) — the surface ran away")
        return False

    if boundary.bbox and boundary.size > 0:
        try:
            box = Bnd_Box()
            BRepBndLib.Add_s(face, box)
            plo = np.array([box.CornerMin().X(), box.CornerMin().Y(),
                            box.CornerMin().Z()])
            phi = np.array([box.CornerMax().X(), box.CornerMax().Y(),
                            box.CornerMax().Z()])
            hlo = np.asarray(boundary.bbox[0], dtype=float)
            hhi = np.asarray(boundary.bbox[1], dtype=float)
            escape = float(np.maximum(
                np.concatenate([hlo - plo, phi - hhi]), 0.0).max())
            boundary.patch_reach = escape / boundary.size
        except Exception:
            boundary.patch_reach = 0.0
        if boundary.patch_reach > MAX_PATCH_REACH:
            boundary.note = (
                f"{method} patch reaches {escape:.0f} beyond the hole "
                f"({boundary.patch_reach:.1f}x its size) — it is not over the hole")
            return False

    boundary.fill_method = method
    return True


class _Backing:
    """Asks what is already underneath a proposed patch.

    Built once per heal: the shape is tessellated with each triangle remembering
    its B-rep face, so a patch can be tested against everything except the faces
    its own boundary belongs to. Without that exclusion the answer is always zero -
    the nearest surface to a hole's rim is the face the rim is part of.
    """

    def __init__(self, shape, probe):
        from aox_g3 import topology
        self.ok = False
        self.probe = probe
        try:
            import trimesh  # noqa: F401
            self.mesh, self.owner, faces = topology.tessellate_with_owner(shape)
            if len(self.mesh.faces) == 0:
                return
            self.index_of = {face: i for i, face in enumerate(faces)}
            self.ok = True
        except Exception:
            self.ok = False

    def own_faces(self, wire, supports):
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.TopoDS import TopoDS
        own = set()
        if not self.ok or supports is None:
            return own
        for e in _explore(wire, TopAbs_EDGE):
            edge = TopoDS.Edge_s(e)
            if supports.Contains(edge):
                for f in supports.FindFromKey(edge):
                    index = self.index_of.get(TopoDS.Face_s(f))
                    if index is not None:
                        own.add(index)
        return own

    def backed_fraction(self, patch, own):
        """What share of the patch has existing surface behind it, along its normal.

        Distance alone misreads thin slits. A sliver patch bridging a panel gap has
        no interior - every sample sits millimetres from the slit's own walls - so
        plain proximity calls it backed and the gap never closes. The walls are
        beside the patch; an overlapping panel is behind it. The direction to the
        nearest surface, dotted with the patch normal, tells them apart.
        """
        if not self.ok:
            return 0.0
        import trimesh
        from aox_g3 import topology
        try:
            keep = (~np.isin(self.owner, list(own)) if own
                    else np.ones(len(self.owner), bool))
            if not keep.any():
                return 0.0
            other = trimesh.Trimesh(vertices=self.mesh.vertices,
                                    faces=np.asarray(self.mesh.faces)[keep],
                                    process=False)
            patch_mesh = topology.tessellate_by_curvature(patch, 20.0)
            if len(patch_mesh.faces) == 0:
                return 0.0
            points = np.asarray(patch_mesh.triangles_center, dtype=float)
            normals = np.asarray(patch_mesh.face_normals, dtype=float)
            if len(points) > 300:
                pick = np.linspace(0, len(points) - 1, 300).astype(int)
                points, normals = points[pick], normals[pick]
            closest, distance, _ = trimesh.proximity.closest_point(other, points)
            towards = closest - points
            length = np.linalg.norm(towards, axis=1)
            safe = length > 1e-9
            alignment = np.zeros(len(points))
            alignment[safe] = np.abs(
                (towards[safe] / length[safe, None] * normals[safe]).sum(axis=1))
            backed = (distance < self.probe) & (alignment > BACKING_NORMAL_DOT)
            return float(backed.mean())
        except Exception:
            return 0.0


def face_supports(shape):
    """Map every edge to the faces it belongs to, so a patch can be told them.

    This is what turns a filling constraint from "pass through this curve" into
    "continue this surface", and without it the solver has no idea what shape it is
    supposed to make.
    """
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    table = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, table)
    return table


# A cap on the loop's own best-fit plane is only honest while the loop is near
# that plane. On CAS-A 43 of 44 holes sit within 0.20 of their own size, and the
# absolute worst is under 10 mm on a 5 m car, so this rejects almost nothing while
# refusing the one case where a plane would be a lie.
MAX_PLANE_RESIDUAL_FRAC = 0.20


def _fitted_plane_face(boundary: Boundary):
    """A bounded planar face on the loop's best-fit plane, or None.

    Used as MakeFilling's initial surface, which it deforms to meet the boundary
    rather than solving from nothing. The plane's own extent has to cover the hole,
    so the loop is projected onto the plane's axes to size it.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt

    if not boundary.plane_normal or not boundary.bbox or boundary.size <= 0:
        return None
    if boundary.plane_residual > MAX_PLANE_RESIDUAL_FRAC * boundary.size:
        return None

    origin = np.asarray(boundary.plane_origin, dtype=float)
    normal = np.asarray(boundary.plane_normal, dtype=float)
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
        return None
    try:
        axes = gp_Ax3(gp_Pnt(*origin), gp_Dir(*normal))
        plane = gp_Pln(axes)
        x = np.array([axes.XDirection().X(), axes.XDirection().Y(),
                      axes.XDirection().Z()])
        y = np.array([axes.YDirection().X(), axes.YDirection().Y(),
                      axes.YDirection().Z()])
        corners = np.array([[a, b, c] for a in boundary.bbox[0][:1] + boundary.bbox[1][:1]
                            for b in boundary.bbox[0][1:2] + boundary.bbox[1][1:2]
                            for c in boundary.bbox[0][2:3] + boundary.bbox[1][2:3]])
        local = corners - origin
        u, v = local @ x, local @ y
        # A margin so the deformed surface has room to reach the boundary
        pad = 0.25 * boundary.size
        maker = BRepBuilderAPI_MakeFace(plane, float(u.min() - pad),
                                        float(u.max() + pad),
                                        float(v.min() - pad),
                                        float(v.max() + pad))
        return maker.Face() if maker.IsDone() else None
    except Exception:
        return None


def _shared_edge_cap(wire, boundary: Boundary):
    """Build the cap on the boundary's own edges, so it never needs sewing.

    Every other cap here owns its edges - projected curves or a polygon - and then
    has to be sewn to the body, and that seam is where the loops came back. At
    10.5 mm about half the caps missed and the shape carried 51 free boundaries
    while reporting 8; at 40 mm they joined but 480 edges went sloppy and the STEP
    reader inflated one to 249 mm. Sewing is the wrong tool for a seam we made.

    CAD bounds a face by edges it shares with its neighbours. Give MakeFace the
    boundary wire itself and the cap shares those TopoDS_Edge objects with the
    body: the edge now has two faces and the loop is simply gone. The wire is not
    on the plane; ShapeFix_Face adds the pcurves and raises the tolerance on
    exactly those edges to the plane deviation, and nothing else is touched.
    Measured on CAS-A: 33 of 43 fillable loops close this way, 46 free boundaries
    become 13 after only a 1 mm sew, and the largest edge tolerance is the largest
    plane residual (26.9 mm) rather than 249. The ten that fail are the wheel spoke
    gaps, whose 8-10 edge loops sit 7-12% of their size off any plane and come back
    as invalid faces; those fall through to the polygon cap.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.ShapeFix import ShapeFix_Face
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt

    if not boundary.plane_normal or boundary.size <= 0:
        return None
    if boundary.plane_residual > MAX_PLANE_RESIDUAL_FRAC * boundary.size:
        return None
    origin = np.asarray(boundary.plane_origin, dtype=float)
    normal = np.asarray(boundary.plane_normal, dtype=float)
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
        return None
    try:
        plane = gp_Pln(gp_Ax3(gp_Pnt(*origin), gp_Dir(*normal)))
        maker = BRepBuilderAPI_MakeFace(plane, wire, True)
        if not maker.IsDone():
            return None
        face = maker.Face()
        fixer = ShapeFix_Face(face)
        fixer.SetPrecision(max(boundary.plane_residual, 1e-3))
        fixer.SetMaxTolerance(max(2.0 * boundary.plane_residual, 1e-2))
        fixer.Perform()
        face = fixer.Face()
        if not BRepCheck_Analyzer(face).IsValid():
            return None
        return face
    except Exception:
        return None


def _polygon_cap(wire, boundary: Boundary):
    """Cap the loop with a polygon of its own samples, projected to its plane.

    The curve-projection route fragments on the wheel spoke gaps: eight edges come
    back as four disconnected arcs, the largest 'loop' is open, and the face built
    on it fails every acceptance test. None of that can happen to a polygon. The
    wire is walked in order (BRepTools_WireExplorer gives traversal order, plain
    TopExp does not), each edge sampled, every point dropped onto the fitted
    plane, and the polygon closed by construction. The cost is polygonal edges
    instead of the true curves - a few millimetres of chord error on a 200 mm
    spoke gap, under the sewing tolerance that joins the cap in.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt
    from OCP.TopoDS import TopoDS

    origin = np.asarray(boundary.plane_origin, dtype=float)
    normal = np.asarray(boundary.plane_normal, dtype=float)
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
        return None
    normal = normal / np.linalg.norm(normal)

    try:
        points = []
        explorer = BRepTools_WireExplorer(wire)
        while explorer.More():
            edge = explorer.Current()
            reversed_ = str(edge.Orientation()).endswith("REVERSED")
            curve = BRepAdaptor_Curve(edge)
            first, last = BRep_Tool.Range_s(edge)
            ts = np.linspace(first, last, 8)
            if reversed_:
                ts = ts[::-1]
            for t in ts[:-1]:   # the edge's last point is the next edge's first
                p = curve.Value(float(t))
                points.append((p.X(), p.Y(), p.Z()))
            explorer.Next()
        if len(points) < 3:
            return None
        pts = np.asarray(points, dtype=float)
        flat = pts - np.outer((pts - origin) @ normal, normal)
        # Collapse consecutive duplicates, which MakePolygon refuses
        keep = np.ones(len(flat), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(flat, axis=0), axis=1) > 1e-6
        flat = flat[keep]
        if len(flat) < 3:
            return None

        polygon = BRepBuilderAPI_MakePolygon()
        for p in flat:
            polygon.Add(gp_Pnt(*p))
        polygon.Close()
        if not polygon.IsDone():
            return None
        plane = gp_Pln(gp_Ax3(gp_Pnt(*origin), gp_Dir(*normal)))
        maker = BRepBuilderAPI_MakeFace(plane, polygon.Wire(), True)
        return maker.Face() if maker.IsDone() else None
    except Exception:
        return None


def _plane_cap(wire, boundary: Boundary, plane_face):
    """Project the loop onto its own plane and cap it there. Cannot wander.

    Every attempt to *solve* for this surface failed the same way, and measuring
    all four combinations showed why. Anchoring the solve on the plane helps a
    little (reach 2.98 to 1.56 on one hole) but not enough, and asking for tangency
    to the neighbouring faces makes it far worse - reach 1.18 becomes 42, and
    plane-anchored G1 reaches 119,000 times the hole size. A free boundary edge
    borders exactly one face, so demanding tangency to a dozen unrelated panels
    around a convoluted loop over-constrains the problem, and the solver satisfies
    it by leaving.

    So do not solve. Take the plane, project the boundary onto it, and build the
    face there. The patch is planar by construction, so its reach is zero and no
    acceptance test can be surprised by it.

    What it costs is a step where the real boundary is off the plane. On CAS-A that
    is at most 10 mm, and under 14 mm for the group this actually fills - on a
    4.8 m car, inside a wheel-spoke gap, and below the tolerance the sewing ladder
    already closes. A millimetre step is a fair price for never producing a patch
    that reaches across the car.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepAlgo import BRepAlgo_NormalProjection
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.ShapeFix import ShapeFix_Face
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_WIRE
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_HSequenceOfShape

    if plane_face is None:
        return None
    try:
        projector = BRepAlgo_NormalProjection(plane_face)
        projector.Add(wire)
        projector.SetDefaultParams()
        projector.Build()
        if not projector.IsDone():
            return None
        edges = TopTools_HSequenceOfShape()
        for e in _explore(projector.Projection(), TopAbs_EDGE):
            edges.Append(e)
        if edges.Size() == 0:
            return None

        # The projection comes back as loose edges; reassemble them into loops.
        # There is usually more than one. A wheel rim projects to an outer ring and
        # an inner one, and capping only the outer covers the spokes as well - the
        # first version did exactly that and produced patches 3x the area of the
        # exact planar caps on comparable holes. The largest loop bounds the face;
        # the rest are its holes.
        wires = TopTools_HSequenceOfShape()
        ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(
            edges, 1e-4 * max(boundary.size, 1.0), False, wires)
        if wires.Size() == 0:
            return None

        loops = []
        for i in range(1, wires.Size() + 1):
            loop = TopoDS.Wire_s(wires.Value(i))
            box = Bnd_Box()
            BRepBndLib.Add_s(loop, box)
            if box.IsVoid():
                continue
            lo = np.array([box.CornerMin().X(), box.CornerMin().Y(),
                           box.CornerMin().Z()])
            hi = np.array([box.CornerMax().X(), box.CornerMax().Y(),
                           box.CornerMax().Z()])
            loops.append((float(np.linalg.norm(hi - lo)), loop))
        if not loops:
            return None
        loops.sort(key=lambda pair: -pair[0])

        # Build on the infinite plane with Inside=True, not on the padded face.
        # A projected wire arrives with arbitrary orientation, and MakeFace on a
        # bounded face keeps whatever side the winding points at - on the wheel
        # spoke rings that was the padding minus the loop, a complement whose area
        # exceeded the loop's own bounding box (ratio 1.7-1.9, geometrically
        # impossible for the loop's inside) and every cap was rejected for it.
        # Inside=True pins the face to the loop's interior regardless of winding.
        from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt
        origin = np.asarray(boundary.plane_origin, dtype=float)
        normal = np.asarray(boundary.plane_normal, dtype=float)
        plane = gp_Pln(gp_Ax3(gp_Pnt(*origin), gp_Dir(*normal)))
        maker = BRepBuilderAPI_MakeFace(plane, loops[0][1], True)
        if not maker.IsDone():
            return None
        for _, inner in loops[1:]:
            try:
                maker.Add(inner)
            except Exception:
                pass
        face = maker.Face()
        fixer = ShapeFix_Face(face)
        fixer.FixOrientationMode = 1
        fixer.Perform()
        return fixer.Face()
    except Exception:
        return None


def fill_boundary(wire, boundary: Boundary, supports=None, degree: int = 3,
                  max_segments: int = 12):
    """Build a face closing this loop. Returns the face, or None.

    A flat loop gets a planar face, which is exact and instant. Anything else goes
    to MakeFilling, which solves for a GeomPlate surface constrained by the
    boundary - the B-rep form of a blend patch.

    **Give it the neighbouring faces.** Constrained by the edges alone the solver
    knows where the boundary is and nothing about what the surface should look
    like, so it is free to wander, and it does: 18 of 32 patches left the hole
    entirely, one reaching 9.3 m out of a 238 mm opening. `Add(edge, support, G1)`
    asks instead for tangency with the face the edge already lies on, which is the
    same idea as projecting the patch onto the surrounding surface - the shape is
    taken from the geometry that is there rather than invented.

    G1 needs the edge to have a 2D representation on the face it is given, so each
    edge falls back to C0 on its own when that is missing, and the whole patch
    falls back to edges-only if the constrained solve fails. Every result is
    checked by _accept_patch regardless of how it was made.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
    from OCP.GeomAbs import GeomAbs_C0, GeomAbs_G1
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS

    if boundary.planar:
        try:
            maker = BRepBuilderAPI_MakeFace(wire, True)
            if maker.IsDone():
                face = maker.Face()
                if _accept_patch(face, boundary, "planar"):
                    return face
        except Exception:
            pass

    edges = [TopoDS.Edge_s(e) for e in _explore(wire, TopAbs_EDGE)]

    def solve(use_supports, init_surface=None):
        filler = BRepOffsetAPI_MakeFilling(degree, 15, 2, False, 1e-5, 1e-4,
                                           1e-2, 1e-1, 8, max_segments)
        if init_surface is not None:
            filler.LoadInitSurface(init_surface)
        tangent = 0
        for edge in edges:
            support = None
            if use_supports and supports is not None:
                try:
                    if supports.Contains(edge):
                        faces = supports.FindFromKey(edge)
                        if faces.Size():
                            support = TopoDS.Face_s(faces.First())
                except Exception:
                    support = None
            if support is not None:
                try:
                    filler.Add(edge, support, GeomAbs_G1)
                    tangent += 1
                    continue
                except Exception:
                    pass
            filler.Add(edge, GeomAbs_C0)
        filler.Build()
        return filler, tangent

    # Projection first: a planar cap cannot run away, and on this geometry that
    # matters more than the millimetres of step it introduces.
    # On the boundary's own edges first: no seam to sew, no tolerance to spend
    cap = _shared_edge_cap(wire, boundary)
    if cap is not None and _accept_patch(cap, boundary, "shared"):
        return cap

    plane_face = _fitted_plane_face(boundary)
    cap = _plane_cap(wire, boundary, plane_face)
    if cap is not None and _accept_patch(cap, boundary, "projected"):
        return cap
    if plane_face is not None:
        cap = _polygon_cap(wire, boundary)
        if cap is not None and _accept_patch(cap, boundary, "polygon"):
            return cap

    # Then the solved surface, for boundaries too far from any plane to project.
    # G1 to the neighbouring faces is deliberately not offered: measured on every
    # hole it made the result worse in every case, by up to five orders of
    # magnitude, because a free edge borders one face and tangency to a ring of
    # unrelated panels has no solution near the hole.
    attempts = [("plane-anchored", False, plane_face)] if plane_face is not None else []
    attempts.append(("free", False, None))

    for label, use_supports, init_surface in attempts:
        try:
            filler, tangent = solve(use_supports, init_surface)
        except Exception as exc:
            boundary.note = f"{label} filling raised {type(exc).__name__}"
            continue
        if not filler.IsDone():
            boundary.note = f"{label}: the surface solver did not converge"
            continue
        method = f"{label}({tangent}/{len(edges)})" if tangent else label
        face = filler.Shape()
        if _accept_patch(face, boundary, method):
            return face
    return None


def heal(shape, sealing_size: Optional[float] = None,
         report: Optional[HealReport] = None, fill_all: bool = False,
         sew_tolerance: Optional[float] = None, close_near=None):
    """Close the holes below the sealing size, leave the rest, build a solid.

    The sealing size is the parameter the product has to expose, and the reason it
    cannot default to "as large as possible" is physical: a cooling inlet, a duct
    mouth or a grille slot is a hole the flow is supposed to go through. Closing it
    changes the answer rather than fixing the model. So the threshold has to sit
    below the smallest opening that is real, and everything above it is reported
    instead of sealed.

    close_near is the explicit override for deliberate simplifications, and exists
    because of wheels. A modelled wheel has spokes lying just behind the rim ring,
    so the backing test rightly refuses to cap it - a cap over spokes is a second
    skin. Closing the rim anyway is a standard aerodynamic simplification (a closed
    rim), but it is a modelling decision, so it has to be asked for: each entry is
    (x, y, z, radius), every boundary whose centre falls inside skips the backing
    rejection, and the patch is labelled a simplification in the report. The size
    and shape checks still apply - a deliberate cap may not run away either.
    """
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.ShapeFix import ShapeFix_Shell, ShapeFix_Solid
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopoDS import TopoDS, TopoDS_Compound

    report = report or HealReport()
    report.faces_before = len(_explore(shape, TopAbs_FACE))
    before = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, before)
    report.area_before = float(before.Mass())

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    lo, hi = box.CornerMin(), box.CornerMax()
    diag = float(np.linalg.norm([hi.X() - lo.X(), hi.Y() - lo.Y(),
                                 hi.Z() - lo.Z()]))
    if sealing_size is None:
        # 1/100 of the body diagonal is about 50 mm on a car: larger than a panel
        # gap or a sewing remnant, smaller than any inlet worth resolving
        sealing_size = diag / 100.0
    report.sealing_size = sealing_size

    holes, dangling = free_boundaries(shape)
    report.boundaries_found = len(holes)
    if dangling:
        report.warnings.append(
            f"{len(dangling)} free boundaries do not close into a loop; "
            "the topology is broken there and filling cannot fix it")

    builder = BRep_Builder()
    # Caps that share the body's edges need only a fine sew; caps that own their
    # edges need a tolerance covering their plane residual, applied in context
    shared_patches = TopoDS_Compound()
    builder.MakeCompound(shared_patches)
    loose_patches = TopoDS_Compound()
    builder.MakeCompound(loose_patches)
    # Built once: the patches take their shape from the faces they border, and are
    # then checked against what is already behind them
    supports = face_supports(shape)
    backing = _Backing(shape, diag * BACKING_PROBE_FRAC)
    if not backing.ok:
        report.warnings.append(
            "could not tessellate to check what is behind each patch; overlapping "
            "panels may be capped instead of left to the intersection step")
    n_patches = 0
    for boundary, wire in holes:
        if not fill_all and boundary.size > sealing_size:
            boundary.note = boundary.note or "larger than the sealing size"
            report.left_open.append(boundary)
            continue
        face = fill_boundary(wire, boundary, supports)
        if face is None:
            boundary.note = boundary.note or "filling failed"
            report.left_open.append(boundary)
            continue
        centre = np.asarray(boundary.centre, dtype=float)
        requested = any(
            float(np.linalg.norm(centre - np.asarray(point[:3]))) <= float(point[3])
            for point in (close_near or ()))
        boundary.backed_fraction = backing.backed_fraction(
            face, backing.own_faces(wire, supports))
        narrow = boundary.patch_ratio < SLIT_PATCH_RATIO
        if boundary.backed_fraction > MAX_BACKED_FRACTION and narrow:
            boundary.note = (
                f"narrow strip ({boundary.patch_ratio:.2f} of the hole's scale) "
                f"bridging a step — {100 * boundary.backed_fraction:.0f}% backed "
                "but too thin to double a surface")
        if boundary.backed_fraction > MAX_BACKED_FRACTION and not narrow:
            if requested:
                boundary.note = (
                    f"closed on request as a simplification — "
                    f"{100 * boundary.backed_fraction:.0f}% of the patch has "
                    "surface behind it (e.g. spokes behind a rim)")
            else:
                boundary.note = (
                    f"{100 * boundary.backed_fraction:.0f}% of the patch has "
                    "surface under it — this is an overlapping panel, not an "
                    "opening; it belongs to the intersection step")
                report.left_open.append(boundary)
                continue
        boundary.filled = True
        report.filled.append(boundary)
        report.patch_area += boundary.patch_area
        # "planar" caps are MakeFace(wire) on the original wire, so they share
        # edges exactly as "shared" ones do; only projected and polygon caps own
        # their edges and need the coarser, context-limited sew
        method = boundary.fill_method.split("(")[0]
        builder.Add(shared_patches if method in ("shared", "planar")
                    else loose_patches, face)
        n_patches += 1
    report.boundaries_filled = n_patches
    report.boundaries_left = len(report.left_open)
    # Per-patch checks catch a runaway surface; this catches many small ones adding
    # up to a body that is no longer the body that came in
    if report.area_before and report.patch_area > 0.5 * report.area_before:
        report.warnings.append(
            f"the patches add {report.patch_area / 1e6:.2f} m2 to a "
            f"{report.area_before / 1e6:.2f} m2 model — check them before using it")

    if n_patches:
        # Two kinds of patch, two ways of joining them.
        #
        # Shared-edge caps already share their edges with the body, so a fine pass
        # (diagonal/5000, about 1 mm) is enough to fold them into the shells and
        # cannot add slop anywhere. Caps that own their edges - projected or
        # polygon - sit off their loop by the loop's plane residual and need a
        # tolerance that covers it; sewing the whole body at that tolerance is what
        # inflated 480 edges and let the STEP reader reach 249 mm, so those are
        # sewn in context mode: the body is loaded as context and only the added
        # patches are matched against it.
        loose = [b for b in report.filled
                 if b.fill_method.split("(")[0] not in ("shared", "planar")]
        fine = diag / 5000.0
        sewer = BRepBuilderAPI_Sewing(fine)
        sewer.Add(shape)
        sewer.Add(shared_patches)
        sewer.Perform()
        shape = sewer.SewedShape()
        # Instrumented: the shared caps went missing somewhere between here and the
        # end (v15: 38 patches, +19 faces), so record the state after each pass
        report.faces_after_fine_sew = len(_explore(shape, TopAbs_FACE))
        report.free_after_fine_sew = len(free_boundaries(shape)[0])

        report.patch_sew_tolerance = float(fine)
        if loose:
            # Caps that own their edges are left floating, deliberately.
            #
            # Every way of sewing them in was measured and every one damages the
            # body: a global pass at 1.5x their residual (40 mm) inflates 480 edge
            # tolerances, context mode (Load/Add) silently drops the caps from
            # the result, and local-tolerance mode - tolerance set on the cap
            # edges only - still propagates through shared vertices to 460 edges.
            # Sewing carries slop wherever the topology connects, and these seams
            # need 14-27 mm of it.
            #
            # So the cap is placed but not joined. It sits within its loop's plane
            # residual of the rim, which closes the opening for the wrap that
            # produces the CFD mesh downstream, and the STEP that goes back to the
            # CAD engineer shows a face there instead of a hole. The count of
            # floating caps is reported so nobody mistakes them for joined ones.
            report.floating_caps = len(loose)
            assembly = TopoDS_Compound()
            builder.MakeCompound(assembly)
            builder.Add(assembly, shape)
            builder.Add(assembly, loose_patches)
            folder = BRepBuilderAPI_Sewing(fine)
            folder.Add(assembly)
            folder.Perform()
            shape = folder.SewedShape()

    shells = _explore(shape, TopAbs_SHELL)
    for s in shells:
        fixer = ShapeFix_Shell()
        fixer.Init(TopoDS.Shell_s(s))
        fixer.Perform()

    solid_fixer = ShapeFix_Solid()
    try:
        solid = solid_fixer.SolidFromShell(TopoDS.Shell_s(shells[0])) \
            if len(shells) == 1 else shape
    except Exception:
        solid = shape
    shape = solid if solid is not None else shape

    report.faces_after = len(_explore(shape, TopAbs_FACE))
    report.shells_after = len(_explore(shape, TopAbs_SHELL))
    report.solids_after = len(_explore(shape, TopAbs_SOLID))
    # Measured, not bookkept: boundaries_left counts holes that were not filled,
    # which is a different number from the free boundaries the shape still has.
    # A cap that did not sew in adds two loops the hole count never sees.
    remaining, _ = free_boundaries(shape)
    report.free_boundaries_after = len(remaining)
    report.closed = len(remaining) == 0
    report.valid = bool(BRepCheck_Analyzer(shape).IsValid())

    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    report.area = float(props.Mass())
    if report.closed:
        vprops = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, vprops)
        report.volume = float(vprops.Mass())
    return shape, report


def write_step(shape, path, report: Optional[HealReport] = None,
               units: str = "MM", pcurves: bool = False):
    """Write STEP AP214, the format the CAD engineer gets back.

    Pcurves are off by default. OCC writes the parametric curve of every edge on
    every adjacent face, which on a body of a few thousand trimmed surfaces is the
    bulk of the file, and the receiving CAD system recomputes them on import
    regardless. Turning them on is worth it only when the consumer is another OCC
    program that would rather load them than recompute.
    """
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs

    Interface_Static.SetCVal_s("write.step.unit", units)
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
    Interface_Static.SetIVal_s("write.surfacecurve.mode", 1 if pcurves else 0)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    ok = writer.Write(str(path)) == IFSelect_RetDone
    if report is not None:
        report.step_written = str(path) if ok else ""
        if not ok:
            report.warnings.append("STEP writer refused to write")
    return ok
