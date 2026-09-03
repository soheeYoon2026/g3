"""Steps 7, 8 and 9 of the heal order: intersections, hidden faces, orientation.

These are the stages that decide whether a mesher will accept the model and
whether the surface it meshes is the one the flow actually sees. All three work on
the B-rep so the result can still leave as STEP.

The order in the checklist hides a dependency worth stating, because ignoring it
produces confident nonsense. **Step 8 needs the body to be closed first.** Hidden
geometry is defined by flooding the empty space from outside and seeing what it
reaches, and that only means "outside" if the flood cannot get in. On CAS-A it
can: the underbody opening is 4.85 m across, the flood fills the cabin, and every
panel with the cabin behind it comes back as a zero-thickness baffle. Measured that
way the model reports 524 baffles covering 6.52 m2 - more than any other class on a
15 m2 car - and every one of them is an artefact.

So classify_faces measures whether it is entitled to an answer before giving one,
and says so when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# A closed car encloses 30-55% of its bounding box; carA_base, a solved G2 surface,
# measures 55.8%. When the flood fill leaves essentially nothing enclosed the
# surface is leaking and "outside" has lost its meaning, so the classification is
# reported as untrustworthy rather than returned as fact.
MIN_ENCLOSED_FRACTION = 0.05


@dataclass
class IntersectionReport:
    self_intersections: int = 0
    other_faults: int = 0
    faulty_faces: int = 0
    seconds: float = 0.0
    checked: bool = False
    warnings: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


@dataclass
class FaceClassification:
    trustworthy: bool = False
    enclosed_fraction: float = 0.0
    pitch: float = 0.0
    min_passage: float = 0.0
    grid_shape: tuple = ()
    skin_faces: int = 0
    baffle_faces: int = 0
    hidden_faces: int = 0
    skin_area: float = 0.0
    baffle_area: float = 0.0
    hidden_area: float = 0.0
    total_area: float = 0.0
    verdict: object = None      # per-face codes, 0 skin / 1 baffle / 2 hidden
    warnings: list = field(default_factory=list)

    def as_dict(self):
        d = asdict(self)
        d.pop("verdict")
        d["grid_shape"] = list(self.grid_shape)
        return d


@dataclass
class OrientationReport:
    faces: int = 0
    forward: int = 0
    reversed_: int = 0
    shells_with_bad_edges: int = 0
    shells_fixed: int = 0
    warnings: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def _explore(shape, kind):
    from OCP.TopExp import TopExp_Explorer
    out, ex = [], TopExp_Explorer(shape, kind)
    while ex.More():
        out.append(ex.Current())
        ex.Next()
    return out


def _diagonal(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    lo, hi = box.CornerMin(), box.CornerMax()
    return float(np.linalg.norm([hi.X() - lo.X(), hi.Y() - lo.Y(),
                                 hi.Z() - lo.Z()]))


# --------------------------------------------------------------- 7 intersections

def check_intersections(shape, fuzzy: float = 0.0, use_obb: bool = True,
                        report: Optional[IntersectionReport] = None):
    """Count faces that cut through each other. Slow, and worth the wait.

    A mesher meeting intersecting surfaces either refuses the model or resolves the
    crossing its own way, and neither is something to discover after the run. On
    CAS-A this finds 8,830 self-intersecting pairs among 2,847 faces in about eight
    minutes, so it is not a step to put in an inner loop.

    BOPAlgo_CheckerSI is the lower-level route and performs fine, but reading its
    result set through DS() segfaults the interpreter. BRepAlgoAPI_Check wraps the
    same algorithm and hands back the offending shapes, so it is the one used here.
    Edge self-interference is left off by default: on a surface export it fires on
    almost every seam and buries the face crossings that matter.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Check
    import time

    report = report or IntersectionReport()
    checker = BRepAlgoAPI_Check(shape, False, True)
    if fuzzy:
        checker.SetFuzzyValue(fuzzy)
    checker.SetUseOBB(use_obb)
    t0 = time.time()
    checker.Perform()
    report.seconds = time.time() - t0
    report.checked = True

    faulty = set()
    for result in checker.Result():
        status = str(result.GetCheckStatus())
        if status.endswith("SelfIntersect"):
            report.self_intersections += 1
        else:
            report.other_faults += 1
        for getter in (result.GetFaultyShapes1, result.GetFaultyShapes2):
            try:
                shapes = getter()
            except Exception:
                continue
            if shapes is None:
                continue
            for s in shapes:
                faulty.add(s)
    report.faulty_faces = len(faulty)
    return report


def split_at_intersections(shape):
    """Cut every face where another face crosses it, so no crossing is implicit.

    This is the Boolean-Exact step of the checklist. General Fuse with a single
    argument intersects the shape against itself and rebuilds it with the crossing
    curves as real edges, which is what a mesher needs: it can then mesh each piece
    instead of guessing what happens along an invisible seam. UnifySameDomain puts
    back together the pieces that were split for no reason, so the face count does
    not explode.

    It does not decide which side of a crossing is material - that needs solids,
    and a surface model has none. What it removes is the ambiguity.

    The arguments have to be handed over separately. General Fuse intersects its
    arguments with each other, so passing one compound holding everything asks it
    to intersect a shape with nothing and it declines - IsDone stays false and the
    shape comes back untouched, which is what happened on two overlapping boxes
    until they were added one at a time. A compound is therefore unpacked into its
    children, and a lone shell into its faces.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_BuilderAlgo
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_FACE
    from OCP.TopoDS import TopoDS_Iterator
    from OCP.TopTools import TopTools_ListOfShape

    pieces = []
    if shape.ShapeType() == TopAbs_COMPOUND:
        iterator = TopoDS_Iterator(shape)
        while iterator.More():
            pieces.append(iterator.Value())
            iterator.Next()
    if len(pieces) < 2:
        pieces = _explore(shape, TopAbs_FACE)
    if len(pieces) < 2:
        return shape, ["nothing to intersect against; the shape is a single face"]

    arguments = TopTools_ListOfShape()
    for piece in pieces:
        arguments.Append(piece)
    builder = BRepAlgoAPI_BuilderAlgo()
    builder.SetArguments(arguments)
    builder.SetRunParallel(False)
    builder.Build()
    if not builder.IsDone():
        return shape, ["general fuse failed; the shape is unchanged"]
    out = builder.Shape()

    unifier = ShapeUpgrade_UnifySameDomain(out, True, True, True)
    unifier.Build()
    return unifier.Shape(), []


# --------------------------------------------------- 8 hidden faces and baffles

def tessellate_with_owner(shape, deflection_frac: float = 0.001,
                          angular: float = 0.5):
    """Triangulate, remembering which B-rep face each triangle came from.

    The classification is a question about faces, but it can only be answered about
    triangles, so the link between them has to survive the tessellation. Keeping it
    is what lets a verdict reached per triangle be applied back to the CAD.
    """
    import trimesh
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    deflection = _diagonal(shape) * deflection_frac
    # BRepMesh keeps its result on the faces and reuses it unless asked for
    # something finer, so a previous coarser pass would silently be handed back
    BRepTools.Clean_s(shape)
    BRepMesh_IncrementalMesh(shape, deflection, False, angular, True)

    from OCP.TopAbs import TopAbs_REVERSED

    vertices, triangles, owner, faces = [], [], [], []
    for f in _explore(shape, TopAbs_FACE):
        face = TopoDS.Face_s(f)
        index = len(faces)
        faces.append(face)
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            continue
        transform = location.Transformation()
        offset = len(vertices)
        for i in range(1, triangulation.NbNodes() + 1):
            p = triangulation.Node(i).Transformed(transform)
            vertices.append((p.X(), p.Y(), p.Z()))
        # A REVERSED face's triangulation is stored in the surface's own
        # parametrisation; the face points the other way, so the winding has to
        # be flipped or 1,717 of CAS-A's faces come out inside-out and any seam
        # angle against them reads as a fold
        flip = face.Orientation() == TopAbs_REVERSED
        for i in range(1, triangulation.NbTriangles() + 1):
            a, b, c = triangulation.Triangle(i).Get()
            tri = (offset + a - 1, offset + b - 1, offset + c - 1)
            triangles.append((tri[0], tri[2], tri[1]) if flip else tri)
            owner.append(index)

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float),
                           faces=np.asarray(triangles, dtype=np.int64),
                           process=False)
    return mesh, np.asarray(owner), faces


def classify_faces(shape, min_passage: Optional[float] = None,
                   deflection_frac: float = 0.001):
    """Sort faces into skin, baffle and hidden — or refuse, and say why.

    Flood the empty space from outside the bounding box, then probe each triangle
    two voxels off its front and its back. Outside on one side only is skin;
    outside on both is a baffle, a zero-thickness sheet with flow on each face;
    outside on neither is buried where the flow never reaches.

    The voxel pitch is the checklist's Minimum Passage Size under another name: a
    gap narrower than the pitch reads as closed, and whatever sits behind it counts
    as interior.

    Before any of that means anything the flood has to be kept out of the body, so
    the enclosed volume is measured first. If the surface leaks the flood fills the
    cabin, the cabin becomes "outside", and every panel in front of it is declared
    a baffle. That is not a rare edge case - it is exactly what CAS-A does - so the
    result carries `trustworthy` and the caller is expected to honour it.
    """
    from scipy import ndimage

    result = FaceClassification()
    mesh, owner, faces = tessellate_with_owner(shape, deflection_frac)
    if len(mesh.faces) == 0:
        result.warnings.append("the shape produced no triangles")
        return result, mesh, owner, faces

    diag = _diagonal(shape)
    if min_passage is None:
        # A car's real openings - grille slots, duct mouths - start around 20 mm,
        # so diagonal/300 (about 17 mm here) stays under them
        min_passage = diag / 300.0
    pitch = float(min_passage)
    result.min_passage = min_passage
    result.pitch = pitch

    lo = np.asarray(mesh.bounds[0], dtype=float)
    hi = np.asarray(mesh.bounds[1], dtype=float)
    pad = 4 * pitch
    origin = lo - pad
    grid = tuple(int(np.ceil((hi[i] + pad - origin[i]) / pitch)) + 1
                 for i in range(3))
    result.grid_shape = grid
    if np.prod(grid) > 6e8:
        result.warnings.append(f"grid {grid} is very large; raise min_passage")
        return result, mesh, owner, faces

    wall = np.zeros(grid, dtype=bool)
    tris = np.asarray(mesh.triangles, dtype=float)
    limit = np.array(grid) - 1
    a = np.clip(np.floor((tris.min(axis=1) - origin) / pitch).astype(np.int64),
                0, limit)
    b = np.clip(np.floor((tris.max(axis=1) - origin) / pitch).astype(np.int64),
                0, limit)
    for (i0, j0, k0), (i1, j1, k1) in zip(a, b):
        wall[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = True

    labels, _ = ndimage.label(~wall)
    outside = labels == labels[0, 0, 0]
    enclosed = int((~wall & ~outside).sum())
    box_voxels = int(np.prod(grid))
    result.enclosed_fraction = enclosed / box_voxels if box_voxels else 0.0
    result.trustworthy = result.enclosed_fraction >= MIN_ENCLOSED_FRACTION
    if not result.trustworthy:
        result.warnings.append(
            f"the flood fill encloses only {100 * result.enclosed_fraction:.1f}% "
            f"of the bounding box, so it is reaching the inside of the body "
            f"through an opening. Close the large holes before classifying: until "
            f"then every panel with an interior behind it reads as a baffle.")

    def probe(points):
        idx = np.floor((points - origin) / pitch).astype(np.int64)
        np.clip(idx, 0, limit, out=idx)
        return outside[idx[:, 0], idx[:, 1], idx[:, 2]]

    centres = np.asarray(mesh.triangles_center, dtype=float)
    normals = np.asarray(mesh.face_normals, dtype=float)
    # Two voxels clears the wall the triangle painted for itself
    step = 2.0 * pitch
    front = probe(centres + normals * step)
    back = probe(centres - normals * step)
    per_triangle = np.where(front ^ back, 0, np.where(front & back, 1, 2))

    areas = np.asarray(mesh.area_faces)
    per_face = np.zeros((len(faces), 3))
    for code in (0, 1, 2):
        pick = per_triangle == code
        np.add.at(per_face[:, code], owner[pick], areas[pick])
    total = per_face.sum(axis=1)
    verdict = np.full(len(faces), -1)
    alive = total > 0
    verdict[alive] = per_face[alive].argmax(axis=1)
    result.verdict = verdict

    face_area = np.zeros(len(faces))
    np.add.at(face_area, owner, areas)
    result.skin_faces = int((verdict == 0).sum())
    result.baffle_faces = int((verdict == 1).sum())
    result.hidden_faces = int((verdict == 2).sum())
    result.skin_area = float(face_area[verdict == 0].sum())
    result.baffle_area = float(face_area[verdict == 1].sum())
    result.hidden_area = float(face_area[verdict == 2].sum())
    result.total_area = float(mesh.area)
    return result, mesh, owner, faces


def rebuild_without(shape, faces, drop_mask, sew_tolerance: Optional[float] = None):
    """Return a shape with the marked faces gone, still sewn into shells."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.TopoDS import TopoDS_Compound

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    kept = 0
    for face, drop in zip(faces, drop_mask):
        if drop:
            continue
        builder.Add(compound, face)
        kept += 1
    if kept == 0:
        return shape, 0
    sewer = BRepBuilderAPI_Sewing(sew_tolerance or _diagonal(shape) / 5000.0)
    sewer.Add(compound)
    sewer.Perform()
    return sewer.SewedShape(), kept


# ----------------------------------------------- 9 orientation and curvature mesh

def check_orientation(shape, fix: bool = False):
    """Are the face normals consistent, and can they be made so?

    Within a shell OCC carries direction as each face's FORWARD or REVERSED flag,
    so a mixture is normal and means nothing on its own - what matters is whether
    neighbours agree across their shared edges. ShapeAnalysis_Shell answers that;
    ShapeFix_Shell's orientation pass is what repairs it.

    On an open shell there is no global outside to point away from, so consistency
    is all that can be asked for. Deciding which way is out needs the body closed,
    the same precondition step 8 has.
    """
    from OCP.ShapeAnalysis import ShapeAnalysis_Shell
    from OCP.ShapeFix import ShapeFix_Shell
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS

    report = OrientationReport()
    for f in _explore(shape, TopAbs_FACE):
        report.faces += 1
        if f.Orientation() == TopAbs_REVERSED:
            report.reversed_ += 1
        elif f.Orientation() == TopAbs_FORWARD:
            report.forward += 1

    shells = _explore(shape, TopAbs_SHELL)
    for s in shells:
        analyser = ShapeAnalysis_Shell()
        analyser.LoadShells(s)
        analyser.CheckOrientedShells(s, True, True)
        if analyser.HasBadEdges():
            report.shells_with_bad_edges += 1
            if fix:
                fixer = ShapeFix_Shell()
                fixer.Init(TopoDS.Shell_s(s))
                fixer.FixFaceOrientation(TopoDS.Shell_s(s), True, True)
                report.shells_fixed += 1
    if report.shells_with_bad_edges and not fix:
        report.warnings.append(
            f"{report.shells_with_bad_edges} of {len(shells)} shells have edges "
            "their neighbours disagree about; pass fix=True to repair them")
    return report


def tessellate_by_curvature(shape, curvature_angle_deg: float = 15.0,
                            linear_deflection: Optional[float] = None,
                            max_edge: Optional[float] = None,
                            relative: bool = False):
    """Surface mesh whose density follows curvature, with the checklist's knobs.

    Curvature Angle is BRepMesh's angular deflection: it bounds how far the surface
    normal may turn across one element, so a tight radius gets small elements and a
    flat panel gets large ones with nobody painting regions by hand.

    Maximum Edge Length is not the same thing as BRepMesh's linear deflection, and
    conflating them is a mistake worth naming. Linear deflection bounds how far a
    chord sags from the surface, and on a planar face the sag is zero at any size -
    which is why the box test case returned exactly 36 triangles at every setting.
    A flat floor would come back as two enormous triangles no matter how small the
    number. So the sag is left to BRepMesh and the edge length is enforced
    afterwards by subdividing whatever still exceeds it.

    None of this changes the B-rep. The STEP going back to the CAD engineer is the
    same either way; these settings only decide what the solver is handed.

    The existing triangulation has to be cleared first. BRepMesh stores its result
    on the faces and reuses it whenever the new linear deflection is no tighter, so
    without the Clean the angular setting appears to do nothing: 30, 15 and 5
    degrees all returned exactly 41,090 triangles on CAS-A.
    """
    import trimesh
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    diag = _diagonal(shape)
    if linear_deflection is None:
        linear_deflection = diag / 2000.0
    angular = float(np.radians(curvature_angle_deg))
    BRepTools.Clean_s(shape)
    BRepMesh_IncrementalMesh(shape, float(linear_deflection), relative,
                             angular, True)

    from OCP.TopAbs import TopAbs_REVERSED

    vertices, triangles = [], []
    for f in _explore(shape, TopAbs_FACE):
        face = TopoDS.Face_s(f)
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            continue
        transform = location.Transformation()
        offset = len(vertices)
        for i in range(1, triangulation.NbNodes() + 1):
            p = triangulation.Node(i).Transformed(transform)
            vertices.append((p.X(), p.Y(), p.Z()))
        # See tessellate_with_owner: REVERSED faces need their winding flipped
        flip = face.Orientation() == TopAbs_REVERSED
        for i in range(1, triangulation.NbTriangles() + 1):
            a, b, c = triangulation.Triangle(i).Get()
            tri = (offset + a - 1, offset + b - 1, offset + c - 1)
            triangles.append((tri[0], tri[2], tri[1]) if flip else tri)

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float),
                           faces=np.asarray(triangles, dtype=np.int64),
                           process=False)
    mesh.merge_vertices()

    if max_edge:
        vertices, triangles = trimesh.remesh.subdivide_to_size(
            mesh.vertices, mesh.faces, max_edge=float(max_edge))
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        mesh.merge_vertices()
    return mesh
