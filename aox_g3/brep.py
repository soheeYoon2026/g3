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


@dataclass
class Boundary:
    """One free boundary: a hole, with enough about it to decide what to do."""
    size: float = 0.0          # bounding box diagonal of the loop
    length: float = 0.0        # perimeter
    centre: tuple = (0.0, 0.0, 0.0)
    bbox: tuple = ()           # (lo, hi) of the loop, for the reach test
    edges: int = 0
    planar: bool = False
    filled: bool = False
    fill_method: str = ""
    patch_area: float = 0.0
    patch_ratio: float = 0.0
    patch_reach: float = 0.0
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
    boundaries_left: int = 0
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
    left_open: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def as_dict(self):
        d = asdict(self)
        d["left_open"] = [b.as_dict() if isinstance(b, Boundary) else b
                          for b in self.left_open]
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
        return length, 0.0, np.zeros(3), n, False, (np.zeros(3), np.zeros(3))
    pts = np.asarray(points)
    centre = pts.mean(axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = float(np.linalg.norm(hi - lo))
    # Planar to within a thousandth of its own size: the smallest singular value of
    # the centred point cloud is the out-of-plane spread
    centred = pts - centre
    try:
        flatness = float(np.linalg.svd(centred, compute_uv=False)[2])
    except Exception:
        flatness = float("inf")
    planar = size > 0 and flatness / max(size, 1e-9) < 1e-3
    return length, size, centre, n, planar, (lo, hi)


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
            length, size, centre, n, planar, box = _wire_metrics(wire)
            bucket.append((Boundary(size=size, length=length, centre=tuple(centre),
                                    bbox=(tuple(box[0]), tuple(box[1])),
                                    edges=n, planar=planar), wire))
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


def fill_boundary(wire, boundary: Boundary, degree: int = 3,
                  max_segments: int = 12):
    """Build a face closing this loop. Returns the face, or None.

    A flat loop gets a planar face, which is exact and instant. Anything else goes
    to MakeFilling, which solves for a GeomPlate surface constrained to pass
    through the boundary edges - the same operation as a CAD blend patch. C0 is
    asked for rather than G1 because tangency to the neighbouring faces needs those
    faces as constraints too, and on a boundary assembled from unrelated panels
    that request usually fails outright instead of degrading.

    Whatever comes back is checked before it is handed on: see _accept_patch.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
    from OCP.GeomAbs import GeomAbs_C0
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

    try:
        filler = BRepOffsetAPI_MakeFilling(degree, 15, 2, False, 1e-5, 1e-4,
                                           1e-2, 1e-1, 8, max_segments)
        for e in _explore(wire, TopAbs_EDGE):
            filler.Add(TopoDS.Edge_s(e), GeomAbs_C0)
        filler.Build()
        if filler.IsDone():
            face = filler.Shape()
            if _accept_patch(face, boundary, "geomplate"):
                return face
        elif not boundary.note:
            boundary.note = "the surface solver did not converge"
    except Exception as exc:
        boundary.note = f"filling raised {type(exc).__name__}"
    return None


def heal(shape, sealing_size: Optional[float] = None,
         report: Optional[HealReport] = None, fill_all: bool = False,
         sew_tolerance: Optional[float] = None):
    """Close the holes below the sealing size, leave the rest, build a solid.

    The sealing size is the parameter the product has to expose, and the reason it
    cannot default to "as large as possible" is physical: a cooling inlet, a duct
    mouth or a grille slot is a hole the flow is supposed to go through. Closing it
    changes the answer rather than fixing the model. So the threshold has to sit
    below the smallest opening that is real, and everything above it is reported
    instead of sealed.
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
    patches = TopoDS_Compound()
    builder.MakeCompound(patches)
    n_patches = 0
    for boundary, wire in holes:
        if not fill_all and boundary.size > sealing_size:
            boundary.note = boundary.note or "larger than the sealing size"
            report.left_open.append(boundary)
            continue
        face = fill_boundary(wire, boundary)
        if face is None:
            boundary.note = boundary.note or "filling failed"
            report.left_open.append(boundary)
            continue
        boundary.filled = True
        report.patch_area += boundary.patch_area
        builder.Add(patches, face)
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
        # Re-sew rather than splicing faces into the shell by hand: the patches are
        # built on the boundary edges, so sewing rejoins them at the right places
        sewer = BRepBuilderAPI_Sewing(sew_tolerance or diag * 0.002)
        sewer.Add(shape)
        sewer.Add(patches)
        sewer.Perform()
        shape = sewer.SewedShape()

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
    remaining, _ = free_boundaries(shape)
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
