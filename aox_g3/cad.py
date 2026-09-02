"""Read STEP, diagnose it while it is still B-rep, then tessellate.

The order matters. Tessellating first and diagnosing the triangles afterwards
throws away exactly the information worth having: which faces are adjacent, where
edges are free rather than shared, how many shells and solids there are, and what
the parts are called. Those are the ANSA-style checks (naked edges, open shells,
invalid faces) and they only exist at the B-rep level.

So: read with OCP keeping topology, report what is wrong, and only then hand a
triangle soup to the sealing cascade in aox_g3.seal.

CATPart is deliberately not supported. It is CATIA V5 native and no open-source
reader exists; the options are a CATIA licence, a commercial converter, or asking
the supplier to export STEP. STEP carries part names, colours and assembly
structure - enough of the semantics to be useful - but no feature history.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CadReport:
    path: str = ""
    read_ok: bool = False
    solids: int = 0
    shells: int = 0
    faces: int = 0
    edges: int = 0
    free_edges: int = 0
    open_shells: int = 0
    invalid_faces: int = 0
    part_names: list = field(default_factory=list)
    colours: int = 0            # colours actually assigned to faces
    colour_palette: int = 0     # colours defined in the file, assigned or not
    bbox: list = field(default_factory=list)
    units_hint: str = ""
    triangles: int = 0
    tessellation_deflection: float = 0.0
    sewn: bool = False
    sew_tolerance: float = 0.0
    sew_stages: list = field(default_factory=list)
    shells_after_sew: int = 0
    free_edges_after_sew: int = 0
    warnings: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def _require_ocp():
    try:
        import OCP  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "OpenCascade bindings missing. Install with: pip install cadquery-ocp"
        ) from exc


def read_step(path, report: Optional[CadReport] = None):
    """Read a STEP file into an OCC shape, preserving names and colours."""
    _require_ocp()
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDF import TDF_LabelSequence
    from OCP.TDataStd import TDataStd_Name
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder

    report = report or CadReport()
    report.path = str(path)

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)

    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    reader.SetLayerMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        report.warnings.append("STEP reader refused the file")
        return None, report
    reader.Transfer(doc)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(labels)
    if labels.Length() == 0:
        report.warnings.append("STEP file contains no shapes")
        return None, report

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for i in range(1, labels.Length() + 1):
        label = labels.Value(i)
        builder.Add(compound, shape_tool.GetShape_s(label))
        name = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), name):
            report.part_names.append(name.Get().ToExtString())

    # GetColors returns the palette, which is not the same as colours actually
    # assigned to geometry: CAS-A carries 15 palette entries and assigns none of
    # them. Since colours are how a CFD engineer picks boundary patches, the number
    # worth reporting is how many faces carry one.
    colour_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    palette = TDF_LabelSequence()
    colour_tool.GetColors(palette)
    report.colour_palette = palette.Length()
    from OCP.Quantity import Quantity_Color
    from OCP.XCAFDoc import XCAFDoc_ColorSurf, XCAFDoc_ColorGen
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    assigned, explorer = 0, TopExp_Explorer(compound, TopAbs_FACE)
    while explorer.More():
        colour = Quantity_Color()
        if colour_tool.GetColor(explorer.Current(), XCAFDoc_ColorSurf, colour) or \
                colour_tool.GetColor(explorer.Current(), XCAFDoc_ColorGen, colour):
            assigned += 1
        explorer.Next()
    report.colours = assigned
    if palette.Length() and not assigned:
        report.warnings.append(
            f"{palette.Length()} colours in the palette but none assigned to a "
            "face; boundary patches cannot be taken from the CAD and have to come "
            "from the geometry")
    report.read_ok = True
    return compound, report


def diagnose(shape, report: CadReport) -> CadReport:
    """Count topology and find the defects that matter before tessellation."""
    _require_ocp()
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import (TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_EDGE)
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRep import BRep_Tool
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.ShapeAnalysis import ShapeAnalysis_Shell

    def count(kind):
        n, explorer = 0, TopExp_Explorer(shape, kind)
        while explorer.More():
            n += 1
            explorer.Next()
        return n

    report.solids = count(TopAbs_SOLID)
    report.shells = count(TopAbs_SHELL)
    report.faces = count(TopAbs_FACE)
    report.edges = count(TopAbs_EDGE)

    # An open shell is the B-rep form of "this will not be watertight"
    explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    while explorer.More():
        analyser = ShapeAnalysis_Shell()
        analyser.LoadShells(explorer.Current())
        analyser.CheckOrientedShells(explorer.Current(), True)
        if analyser.HasFreeEdges():
            report.open_shells += 1
            # FreeEdges() returns a compound of the edges, so count them
            free = analyser.FreeEdges()
            sub = TopExp_Explorer(free, TopAbs_EDGE)
            while sub.More():
                report.free_edges += 1
                sub.Next()
        if analyser.HasBadEdges():
            report.warnings.append("shell has edges shared by more than two faces")
        explorer.Next()

    # Faces OCC itself considers invalid - the closest analogue of ANSA's
    # "unchecked faces" and the ones most likely to break a mesher
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        if not BRepCheck_Analyzer(explorer.Current()).IsValid():
            report.invalid_faces += 1
        explorer.Next()

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if not box.IsVoid():
        lo = box.CornerMin()
        hi = box.CornerMax()
        report.bbox = [round(v, 4) for v in
                       (lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z())]
        span = max(hi.X() - lo.X(), hi.Y() - lo.Y(), hi.Z() - lo.Z())
        # STEP has no reliable unit tag in practice; guess from a car-sized body
        report.units_hint = "mm" if span > 100 else "m"
    return report


def sew(shape, report: CadReport, tolerance: Optional[float] = None):
    """Stitch faces that are adjacent but not topologically joined.

    A CATIA surface export arrives as one shell per face with essentially every
    edge free — on CAS-A, 12,724 of 12,728. They are unstitched, not missing.

    The tolerance used to default to diagonal x 0.002, about 10.5 mm on a car,
    justified by a measured median gap of 5.7 mm. Sweeping it showed both the
    number and the choice were wrong. At 0.1 mm the shells already collapse from
    2,847 to 36 and the free edges from 12,724 to 1,874, so the gaps between
    adjacent panels are overwhelmingly sub-millimetre; the 5.7 mm median was
    measuring the genuine openings mixed in with them.

    Coarsening past that buys little and costs real damage, which only shows up
    when the result has to go back out as CAD. OCC stores its tolerance on the
    edges it creates, so 10 mm of slop stays in the model:

        tolerance   shells   free edges   invalid faces
             0.10       36        1,874              61
             1.00       26        1,672              46
             5.00       22        1,511              83
            10.51       16        1,203              96

    Surface area moves by less than 0.1% across that whole range, so nothing is
    being reshaped either way — the choice is purely how much topological slop to
    accept. 1 mm sits at the minimum, and diagonal/5000 puts it there for a car
    while scaling to other sizes.

    This is a single pass and is kept for when the tolerance is known. Prefer
    sew_progressive, which reaches a coarse tolerance without paying for it on
    every edge.
    """
    _require_ocp()
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing

    if tolerance is None:
        if report.bbox:
            lo = np.array(report.bbox[:3])
            hi = np.array(report.bbox[3:])
            tolerance = float(np.linalg.norm(hi - lo)) / 5000.0
        else:
            tolerance = 1.0

    sewer = BRepBuilderAPI_Sewing(tolerance)
    sewer.SetFloatingEdgesMode(True)
    sewer.Add(shape)
    sewer.Perform()
    sewn = sewer.SewedShape()
    report.sew_tolerance = tolerance
    report.sewn = True
    return sewn, report


def sew_progressive(shape, report: CadReport, tolerances=None):
    """Sew in stages, fine first, so a coarse tolerance costs only what it must.

    A single pass at 10.5 mm is not equivalent to reaching 10.5 mm in steps, and
    the difference is large. OCC records its tolerance on every edge it merges, so
    one coarse pass stamps 10 mm of slop on all 12,724 free edges of a CATIA
    surface export. Sewing again on an already-sewn shape cannot do that: the edges
    joined at 1 mm are no longer free, so each later pass can only reach what is
    still open.

    Measured on CAS-A, against a single coarse pass:

        stages                 free edges   invalid faces   holes   time
        1.05                        1,591              33     140    7.3s
        10.51                       1,203              96     109   19.6s
        1.05 -> 10.51                 792              48      57    8.6s
        1.05 -> 5 -> 10.51            765              46      56    8.9s
        1.05 -> 21                    613              67      30    9.6s

    Better on every axis at once - a third fewer free edges, half the invalid
    faces, half the holes, and less than half the time - because most of the work
    happens cheaply at the fine tolerance and the coarse pass has little left to
    search. Surface area is 15.02 m2 in every row, so none of this reshapes
    anything; it only decides how much topological slop ends up in the file.

    Pushing the last stage beyond about diagonal/500 keeps closing holes but starts
    costing invalid faces again, which is why the ladder stops there.
    """
    _require_ocp()
    if tolerances is None:
        if report.bbox:
            lo = np.array(report.bbox[:3])
            hi = np.array(report.bbox[3:])
            diag = float(np.linalg.norm(hi - lo))
        else:
            diag = 5000.0
        tolerances = [diag / 5000.0, diag / 1000.0, diag / 500.0]

    for tolerance in sorted(tolerances):
        stage = CadReport(bbox=report.bbox)
        shape, stage = sew(shape, stage, tolerance)
    report.sewn = True
    report.sew_tolerance = max(tolerances)
    report.sew_stages = [round(float(t), 4) for t in sorted(tolerances)]
    return shape, report


def tessellate(shape, report: CadReport, deflection_frac: float = 0.001):
    """Triangulate the B-rep into a trimesh, deflection scaled to the body."""
    _require_ocp()
    import trimesh
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location

    if report.bbox:
        lo = np.array(report.bbox[:3])
        hi = np.array(report.bbox[3:])
        deflection = float(np.linalg.norm(hi - lo)) * deflection_frac
    else:
        deflection = 0.1
    report.tessellation_deflection = deflection
    # BRepMesh caches its triangulation on the faces and reuses it whenever the new
    # linear deflection is no tighter, so an earlier coarser pass would come back
    # unchanged and the parameter would appear to do nothing
    BRepTools.Clean_s(shape)
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.5, True)

    vertices, faces = [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            report.warnings.append("a face produced no triangulation")
            explorer.Next()
            continue
        transform = location.Transformation()
        offset = len(vertices)
        for i in range(1, triangulation.NbNodes() + 1):
            p = triangulation.Node(i).Transformed(transform)
            vertices.append((p.X(), p.Y(), p.Z()))
        for i in range(1, triangulation.NbTriangles() + 1):
            a, b, c = triangulation.Triangle(i).Get()
            faces.append((offset + a - 1, offset + b - 1, offset + c - 1))
        explorer.Next()

    if not faces:
        report.warnings.append("tessellation produced no triangles")
        return None, report
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float),
                           faces=np.asarray(faces, dtype=np.int64), process=False)
    # STEP writes each face's triangulation independently, so shared edges arrive
    # as duplicate vertices; without welding every face looks like an island.
    mesh.merge_vertices()
    report.triangles = int(len(mesh.faces))
    return mesh, report


def step_to_mesh(path, deflection_frac: float = 0.001, sew_tolerance=None,
                 do_sew: bool = True):
    """STEP file -> (trimesh, report). Returns (None, report) on failure.

    Order is deliberate: diagnose while the topology still exists, sew what is
    only unstitched, and tessellate last. Sewing after tessellation cannot use
    the surface definitions and is a strictly worse operation.
    """
    shape, report = read_step(path)
    if shape is None:
        return None, report
    diagnose(shape, report)
    if do_sew:
        shape, report = (sew(shape, report, sew_tolerance) if sew_tolerance
                         else sew_progressive(shape, report))
        after = CadReport()
        diagnose(shape, after)
        report.shells_after_sew = after.shells
        report.free_edges_after_sew = after.free_edges
    return tessellate(shape, report, deflection_frac)
