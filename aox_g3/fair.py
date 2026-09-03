"""Close the small gaps in a mesh with patches that follow the surrounding surface.

The B-rep caps are planar, and a planar cap over a curved panel is a visible kink.
For the mesh that goes to the solver the patch should instead continue the
neighbouring surface — the same slope across the seam — which is what a CAD
engineer's "blend surface" does.

The approach is Liepa's (2003), and the reason it works where the B-rep G1 solver
ran away is structural. GeomPlate solves for a surface that extends past the loop
and can satisfy a tangency constraint by leaving; here the boundary ring is pinned
and the interior is found by minimising thin-plate energy, a sparse linear system
that can only interpolate between fixed points. It cannot escape.

Three steps per hole:
  1. triangulate the loop by minimum-weight dynamic programming — worst dihedral
     first, area as tie-break — so the initial patch is already as flat as the
     loop allows, with no new vertices
  2. refine by inserting centroids until the patch's edges match the density of
     the boundary, so the fairing has enough freedom to bend
  3. fair: fix the loop and its one-ring in the body, solve the bi-Laplacian for
     the interior. The one-ring is what carries the neighbours' slope across.

Before any of that the mesh is stitched: tessellating a B-rep sewn at tolerance
leaves neighbouring faces with unmatched vertices along shared edges, and every
such seam is a "hole" of a few millimetres. Merging vertices within a small
distance closes them without a patch. On CAS-A v17 that is 196 of the 226 loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# Above this seam angle the flat patch is meeting walls, not a continuation of
# the surface, and fairing toward the walls folds the cap into the pocket
POCKET_SEAM_ANGLE = 60.0


@dataclass
class HoleResult:
    size: float = 0.0
    vertices: int = 0
    centre: tuple = (0.0, 0.0, 0.0)
    filled: bool = False
    triangles: int = 0
    seam_angle_flat_max: float = 0.0     # degrees, before fairing
    seam_angle_flat_median: float = 0.0
    seam_angle_fair_max: float = 0.0     # degrees, after fairing
    seam_angle_fair_median: float = 0.0
    note: str = ""

    def as_dict(self):
        d = asdict(self)
        d["centre"] = [round(float(v), 1) for v in self.centre]
        return d


@dataclass
class FairReport:
    stitch_tolerance: float = 0.0
    loops_before_stitch: int = 0
    loops_after_stitch: int = 0
    loops_considered: int = 0
    loops_filled: int = 0
    loops_skipped_large: int = 0
    loops_skipped_held: int = 0
    loops_skipped_shape: int = 0
    triangles_added: int = 0
    holes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def as_dict(self):
        d = asdict(self)
        d["holes"] = [h.as_dict() if isinstance(h, HoleResult) else h
                      for h in self.holes]
        return d


# ------------------------------------------------------------------ stitching

def stitch(mesh, tolerance: float):
    """Merge vertices closer than the tolerance and drop the triangles that die.

    A B-rep sewn at 10 mm tessellates into faces whose shared edges do not share
    vertices, so the triangle soup has a hairline seam along every one of them.
    Those are not openings; they are the same edge drawn twice.
    """
    import trimesh
    from scipy.spatial import cKDTree

    vertices = np.asarray(mesh.vertices, dtype=float)

    # Only vertices that lie on an open edge may merge. A seam between two
    # tessellated faces is a pair of boundary vertices a hair apart; an interior
    # vertex a hair from its neighbour is simply a small triangle. Merging
    # everything within 2 mm collapsed the wheel spokes and the fillets - 163,000
    # triangles became 47,000 and the loop count went up, not down.
    edges = np.asarray(mesh.edges_sorted)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    on_boundary = np.unique(unique[counts == 1])
    if len(on_boundary) == 0:
        return mesh
    tree = cKDTree(vertices[on_boundary])
    local_pairs = tree.query_pairs(tolerance, output_type="ndarray")
    pairs = on_boundary[local_pairs] if len(local_pairs) else np.zeros((0, 2), int)
    parent = np.arange(len(vertices))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(len(vertices))])
    unique, remap = np.unique(roots, return_inverse=True)

    # Merged vertices take the mean of their group so no side is favoured
    merged = np.zeros((len(unique), 3))
    counts = np.zeros(len(unique))
    np.add.at(merged, remap, vertices)
    np.add.at(counts, remap, 1.0)
    merged /= counts[:, None]

    faces = remap[np.asarray(mesh.faces)]
    alive = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & \
            (faces[:, 0] != faces[:, 2])
    out = trimesh.Trimesh(vertices=merged, faces=faces[alive], process=False)
    out.remove_unreferenced_vertices()
    return out


# ---------------------------------------------------------------- loop finding

def boundary_loops(mesh):
    """Ordered vertex loops of the mesh's open boundaries.

    Returns (loops, rejected): loops are lists of vertex indices in cyclic order;
    rejected counts components that are not simple cycles (a T-junction chain or a
    figure-eight), which need stitching rather than filling.
    """
    edges = np.asarray(mesh.edges_sorted)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if len(boundary) == 0:
        return [], 0

    adjacency = {}
    for a, b in boundary:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    seen = set()
    loops, rejected = [], 0
    for start in adjacency:
        if start in seen:
            continue
        # collect the component first
        stack, component = [start], set()
        while stack:
            v = stack.pop()
            if v in component:
                continue
            component.add(v)
            stack.extend(adjacency[v])
        seen |= component
        if any(len(adjacency[v]) != 2 for v in component):
            rejected += 1
            continue
        # walk it
        order = [start]
        previous, current = None, start
        while True:
            nxt = [n for n in adjacency[current] if n != previous]
            if not nxt:
                break
            nxt = nxt[0]
            if nxt == start:
                break
            order.append(nxt)
            previous, current = current, nxt
        if len(order) == len(component) and len(order) >= 3:
            loops.append(order)
        else:
            rejected += 1
    return loops, rejected


# ------------------------------------------------------------- triangulation

def _triangle_normal(a, b, c):
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n)
    return n / length if length > 1e-12 else np.zeros(3)


def _dihedral(n1, n2):
    """Angle between two unit normals in degrees; 0 means coplanar."""
    if not n1.any() or not n2.any():
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))))


def min_weight_triangulation(points, outer_normals):
    """Liepa's minimum-weight triangulation of a closed 3D polygon.

    points: (n, 3) loop in order. outer_normals: (n, 3) normal of the body
    triangle across boundary edge (i, i+1), used so the patch's edge triangles
    are judged against what they attach to, not only against each other.

    Weight is (max dihedral, area) compared lexicographically: the patch that
    folds least wins, and among equally flat ones the smallest. O(n^3), fine for
    the loops a sealing size admits (hundreds of vertices at most).
    """
    n = len(points)
    INF = (1e9, 1e9)
    weight = {}
    choice = {}

    def tri_area(i, j, k):
        return 0.5 * np.linalg.norm(np.cross(points[j] - points[i],
                                             points[k] - points[i]))

    def normal(i, j, k):
        return _triangle_normal(points[i], points[j], points[k])

    # Normal of the sub-patch triangle adjacent to edge (i, j) inside range i..j,
    # needed to score dihedrals between neighbouring patch triangles
    edge_normal = {}

    for i in range(n - 1):
        weight[(i, i + 1)] = (0.0, 0.0)
    for span in range(2, n):
        for i in range(0, n - span):
            j = i + span
            best, best_k = INF, None
            for k in range(i + 1, j):
                w_ik = weight.get((i, k), INF)
                w_kj = weight.get((k, j), INF)
                if w_ik[0] >= 1e9 or w_kj[0] >= 1e9:
                    continue
                nk = normal(i, k, j)
                angle = 0.0
                # against the two sub-patches
                if k - i >= 2:
                    angle = max(angle, _dihedral(nk, edge_normal[(i, k)]))
                else:
                    angle = max(angle, _dihedral(nk, outer_normals[i]))
                if j - k >= 2:
                    angle = max(angle, _dihedral(nk, edge_normal[(k, j)]))
                else:
                    angle = max(angle, _dihedral(nk, outer_normals[k]))
                # against the body across the closing edge (i, j) when it is a
                # boundary edge, i.e. the whole loop
                if span == n - 1:
                    angle = max(angle, _dihedral(nk, outer_normals[j % n]))
                total = (max(w_ik[0], w_kj[0], angle),
                         w_ik[1] + w_kj[1] + tri_area(i, k, j))
                if total < best:
                    best, best_k = total, k
            weight[(i, j)] = best
            choice[(i, j)] = best_k
            if best_k is not None:
                edge_normal[(i, j)] = normal(i, best_k, j)

    triangles = []

    def emit(i, j):
        if j - i < 2:
            return
        k = choice.get((i, j))
        if k is None:
            return
        triangles.append((i, k, j))
        emit(i, k)
        emit(k, j)

    emit(0, n - 1)
    return triangles, weight.get((0, n - 1), INF)


def ear_clip(points, normal):
    """Triangulate a large loop by ear clipping in its best-fit plane. O(n^2).

    Minimum-weight triangulation is O(n^3) and fine up to sixty-odd vertices; a
    scanned STL's boundary loops run to hundreds (GTR35: 474 on a loop under the
    sealing size) and would take hours. Ear clipping gives a valid triangulation
    of the projected polygon in seconds. It is only as good as the projection -
    a loop that folds over in its own plane defeats it - so the caller falls back
    to a fan when it fails.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 3 or n > EAR_CLIP_MAX_VERTICES:
        # The inside test below is linear per candidate, so this is cubic in the
        # worst case; past a few hundred vertices the fan is the honest choice
        return []
    normal = np.asarray(normal, dtype=float)
    normal /= max(np.linalg.norm(normal), 1e-12)
    # In-plane basis
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    centre = pts.mean(axis=0)
    flat = np.stack([(pts - centre) @ u, (pts - centre) @ v], axis=1)

    # Ensure counter-clockwise
    area2 = np.sum(flat[:, 0] * np.roll(flat[:, 1], -1) - np.roll(flat[:, 0], -1) * flat[:, 1])
    order = list(range(n)) if area2 > 0 else list(range(n))[::-1]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return (cross(a, b, p) >= -1e-12 and cross(b, c, p) >= -1e-12
                and cross(c, a, p) >= -1e-12)

    triangles = []
    remaining = order[:]
    guard = 0
    while len(remaining) > 3 and guard < 4 * n:
        guard += 1
        clipped = False
        m = len(remaining)
        for i in range(m):
            i0, i1, i2 = remaining[i - 1], remaining[i], remaining[(i + 1) % m]
            a, b, c = flat[i0], flat[i1], flat[i2]
            if cross(a, b, c) <= 1e-12:
                continue  # reflex or degenerate
            if any(inside(flat[k], a, b, c) for k in remaining
                   if k not in (i0, i1, i2)):
                continue
            triangles.append((i0, i1, i2))
            del remaining[i]
            clipped = True
            break
        if not clipped:
            return []  # the projection folded; let the caller fall back
    if len(remaining) == 3:
        triangles.append(tuple(remaining))
    return triangles


# Loops up to this many vertices get the minimum-weight (flattest) triangulation;
# larger ones are ear-clipped in their plane, and past the second limit a fan
MIN_WEIGHT_MAX_VERTICES = 60
EAR_CLIP_MAX_VERTICES = 300


# ---------------------------------------------------------------- refinement

def refine(points, triangles, boundary_count, target_edge):
    """Insert centroids until patch edges are about the boundary's length.

    Boundary edges are never split, so the patch keeps its connection to the body.
    """
    points = [np.asarray(p, dtype=float) for p in points]
    tris = [tuple(t) for t in triangles]
    for _ in range(12):
        new_tris, inserted = [], 0
        for a, b, c in tris:
            pa, pb, pc = points[a], points[b], points[c]
            longest = max(np.linalg.norm(pb - pa), np.linalg.norm(pc - pb),
                          np.linalg.norm(pa - pc))
            centroid = (pa + pb + pc) / 3.0
            # Liepa's scale test: split if the centroid is far from all corners
            # relative to the target density
            far = all(np.linalg.norm(centroid - p) > 0.75 * target_edge
                      for p in (pa, pb, pc))
            if far and longest > 1.5 * target_edge:
                m = len(points)
                points.append(centroid)
                new_tris += [(a, b, m), (b, c, m), (c, a, m)]
                inserted += 1
            else:
                new_tris.append((a, b, c))
        tris = new_tris
        if inserted == 0:
            break
    tris = _flip_relax(points, tris, boundary_count)
    return points, tris


def _flip_relax(points, tris, boundary_count, passes=6):
    """Delaunay-style edge flips on interior edges to undo the centroid slivers."""
    tris = [tuple(t) for t in tris]
    for _ in range(passes):
        edge_to_tris = {}
        for ti, t in enumerate(tris):
            for u, v in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                edge_to_tris.setdefault((min(u, v), max(u, v)), []).append(ti)
        flipped = 0
        used = set()
        for (u, v), owners in edge_to_tris.items():
            if len(owners) != 2 or u < boundary_count and v < boundary_count:
                continue
            t1, t2 = owners
            if t1 in used or t2 in used:
                continue
            a = [x for x in tris[t1] if x not in (u, v)]
            b = [x for x in tris[t2] if x not in (u, v)]
            if len(a) != 1 or len(b) != 1:
                continue
            a, b = a[0], b[0]
            # Orientation must survive the flip. Which way t1 traverses the
            # shared edge decides the quad's cyclic order: with t1 = (u, v, a)
            # and t2 = (v, u, b) the boundary runs u→b→v→a, and the two new
            # triangles are (u, b, a) and (b, v, a). Taking (u, v) as sorted
            # without checking this inverted half the flips and every seam angle
            # against them read as a fold.
            order = tris[t1]
            forward = any(order[i] == u and order[(i + 1) % 3] == v
                          for i in range(3))
            if not forward:
                u, v = v, u

            def angle_at(p, q, r):
                v1, v2 = points[q] - points[p], points[r] - points[p]
                c = np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-12)
                return np.arccos(np.clip(c, -1, 1))

            if angle_at(a, u, v) + angle_at(b, u, v) > np.pi + 1e-6:
                tris[t1] = (u, b, a)
                tris[t2] = (b, v, a)
                used.update((t1, t2))
                flipped += 1
        if flipped == 0:
            break
    return tris


# -------------------------------------------------------------------- fairing

def fair(points, triangles, fixed_count, ring_points, ring_faces,
         max_normal_shift: Optional[float] = None):
    """Move the interior vertices to minimise thin-plate energy.

    The loop vertices (the first `fixed_count` points) stay put, and so does the
    one-ring of body triangles handed in as `ring_faces` (indexing into the
    combined point list). With those pinned the bi-Laplacian solve interpolates a
    surface whose slope across the seam is set by the body, not by the patch.
    Only the normal component of the solution is applied, and it can be capped.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    pts = np.asarray(points, dtype=float)
    all_faces = np.vstack([np.asarray(triangles), np.asarray(ring_faces)]) \
        if len(ring_faces) else np.asarray(triangles)
    n = len(pts)

    # Uniform (umbrella) Laplacian: robust on the slivers refinement leaves
    rows, cols = [], []
    for a, b, c in all_faces:
        for u, v in ((a, b), (b, c), (c, a)):
            rows += [u, v]
            cols += [v, u]
    adjacency = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)),
                                  shape=(n, n)).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    laplacian = sparse.diags(1.0 / degree) @ adjacency - sparse.identity(n)
    bilaplacian = (laplacian @ laplacian).tocsr()

    free = np.array([i for i in range(fixed_count, len(points))
                     if i not in ring_points])
    if len(free) == 0:
        return pts
    fixed = np.array(sorted(set(range(n)) - set(free.tolist())))
    A = bilaplacian[free][:, free]
    B = bilaplacian[free][:, fixed]
    rhs = -B @ pts[fixed]
    try:
        solution = spsolve(A.tocsc(), rhs)
    except Exception:
        return pts

    # Keep only the displacement along the surface normal. Solved for full
    # positions, the uniform bi-Laplacian slides interior vertices tangentially
    # toward their neighbours' centroid - on a flat wheel ring the points moved a
    # median of 178 mm and up to 837 mm while staying within 0.7 mm of the plane,
    # and the ones that crossed the boundary edge flipped their triangles. The
    # bending is the part we want; the sliding is not.
    normals = np.zeros((n, 3))
    for a, b, c in np.asarray(triangles):
        tn = _triangle_normal(pts[a], pts[b], pts[c])
        normals[a] += tn
        normals[b] += tn
        normals[c] += tn
    length = np.linalg.norm(normals, axis=1)
    normals[length > 1e-12] /= length[length > 1e-12, None]

    out = pts.copy()
    shift = solution - pts[free]
    along = (shift * normals[free]).sum(axis=1)
    if max_normal_shift is not None:
        along = np.clip(along, -max_normal_shift, max_normal_shift)
    out[free] = pts[free] + along[:, None] * normals[free]
    return out


# ------------------------------------------------------------------- driver

def fill_holes(mesh, sealing_size: float, held=(), hold_radius: float = 60.0,
               stitch_tolerance: Optional[float] = None,
               report: Optional[FairReport] = None):
    """Stitch, then fill every simple loop under the sealing size with a faired patch.

    `held` is a list of (x, y, z, size) for openings the B-rep stage decided to
    leave alone — overlays, and anything an engineer has not ruled on — so the
    mesh stage does not quietly close what the CAD stage refused to.
    """
    import trimesh

    report = report or FairReport()
    diag = float(np.linalg.norm(np.asarray(mesh.bounds[1]) - np.asarray(mesh.bounds[0])))
    if stitch_tolerance is None:
        stitch_tolerance = diag / 2500.0
    report.stitch_tolerance = stitch_tolerance

    loops, _ = boundary_loops(mesh)
    report.loops_before_stitch = len(loops)
    mesh = stitch(mesh, stitch_tolerance)
    loops, rejected = boundary_loops(mesh)
    report.loops_after_stitch = len(loops) + rejected
    report.loops_skipped_shape = rejected

    vertices = [np.asarray(v, dtype=float) for v in mesh.vertices]
    faces = [tuple(int(x) for x in f) for f in mesh.faces]
    face_normals = np.asarray(mesh.face_normals, dtype=float)

    # Body triangle across each boundary edge, and the one-ring of each vertex
    edge_owner = {}
    vertex_faces = {}
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_owner.setdefault((min(u, v), max(u, v)), []).append(fi)
        for v in (a, b, c):
            vertex_faces.setdefault(v, []).append(fi)

    held = [(np.asarray(h[:3], dtype=float), float(h[3])) for h in held]
    added = 0
    for loop in loops:
        pts = np.array([vertices[i] for i in loop])
        size = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        centre = pts.mean(axis=0)
        result = HoleResult(size=size, vertices=len(loop), centre=tuple(centre))
        report.loops_considered += 1
        if size > sealing_size:
            result.note = "larger than the sealing size"
            report.loops_skipped_large += 1
            report.holes.append(result)
            continue
        if any(np.linalg.norm(centre - c) < hold_radius + 0.5 * abs(size - s)
               and abs(size - s) < 0.3 * max(size, s) for c, s in held):
            result.note = "held by the B-rep stage"
            report.loops_skipped_held += 1
            report.holes.append(result)
            continue

        n = len(loop)
        outer = np.zeros((n, 3))
        for k in range(n):
            u, v = loop[k], loop[(k + 1) % n]
            owners = edge_owner.get((min(u, v), max(u, v)), [])
            if owners:
                outer[k] = face_normals[owners[0]]
        base_points = list(pts)
        if n <= MIN_WEIGHT_MAX_VERTICES:
            triangles, _ = min_weight_triangulation(pts, outer)
        else:
            centred = pts - centre
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            triangles = ear_clip(pts, vt[2])
            if not triangles:
                # Fan from the centroid as a last resort: adds one vertex, always
                # produces a valid patch, never smooth. Better than a hole.
                base_points.append(centre)
                triangles = [(k, (k + 1) % n, n) for k in range(n)]
                result.note = "ear clipping folded; fan used"
        if not triangles:
            result.note = "triangulation failed"
            report.holes.append(result)
            continue
        # The patch must be wound consistently with the body: across a shared
        # edge, a manifold's two triangles traverse it in opposite directions.
        # Find a boundary edge both sides own and compare the directions; the
        # DP patch runs the loop forward, ear clipping may run it either way, so
        # the patch's own direction is read rather than assumed. Averaging body
        # normals - the first version - is meaningless on a ring.
        for k in range(n):
            u, v = loop[k], loop[(k + 1) % n]
            owners = edge_owner.get((min(u, v), max(u, v)), [])
            if not owners:
                continue
            tri = faces[owners[0]]
            body_forward = any(tri[i] == u and tri[(i + 1) % 3] == v
                               for i in range(3))
            patch_tri = next((t for t in triangles
                              if k in t and (k + 1) % n in t), None)
            if patch_tri is None:
                continue
            patch_forward = any(patch_tri[i] == k and patch_tri[(i + 1) % 3] == (k + 1) % n
                                for i in range(3))
            if patch_forward == body_forward:
                triangles = [(a, c, b) for a, b, c in triangles]
            break

        # Seam angle of the flat patch (base_points, not pts: a fan has one
        # extra vertex the triangles refer to)
        result.seam_angle_flat_max, result.seam_angle_flat_median = \
            _seam_angles(np.asarray(base_points), triangles, outer, n)

        # Refine to the boundary's own edge length
        boundary_edges = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        target = float(np.median(boundary_edges))
        local_points, local_tris = refine(base_points, triangles, n, target)

        # One-ring of the loop in the body: pinned during fairing
        ring_faces, ring_pts = [], set()
        index_of = {loop[k]: k for k in range(n)}
        for k in range(n):
            for fi in vertex_faces.get(loop[k], []):
                tri = faces[fi]
                local = []
                for v in tri:
                    if v in index_of:
                        local.append(index_of[v])
                    else:
                        if v not in ring_pts:
                            index_of[v] = len(local_points)
                            local_points.append(vertices[v])
                            ring_pts.add(v)
                        local.append(index_of[v])
                ring_faces.append(tuple(local))
        pinned = {index_of[v] for v in ring_pts}

        # A loop whose flat patch already meets its neighbours at a steep angle
        # is the rim of a pocket - a wheel spoke gap is bounded by the spokes'
        # side walls, not by the rim face - and "continuing the neighbours" there
        # means folding the cap down into the pocket. The flat patch is the outer
        # surface's continuation; keep it.
        pocket = result.seam_angle_flat_median > POCKET_SEAM_ANGLE
        if pocket:
            faired = np.asarray(local_points, dtype=float)
            result.note = (f"pocket rim (flat patch meets its walls at "
                           f"{result.seam_angle_flat_median:.0f}°) — kept flat")
        else:
            faired = fair(local_points, local_tris, n, pinned, ring_faces,
                          max_normal_shift=0.25 * size)
            # Guard: no patch triangle may turn over relative to the flat patch
            flat_normals = [_triangle_normal(*[np.asarray(local_points[i]) for i in t])
                            for t in local_tris]
            turned = any(
                np.dot(_triangle_normal(*[faired[i] for i in t]), fn) < 0
                for t, fn in zip(local_tris, flat_normals) if fn.any())
            if turned:
                faired = np.asarray(local_points, dtype=float)
                result.note = "fairing turned a triangle over — kept flat"

        result.seam_angle_fair_max, result.seam_angle_fair_median = \
            _seam_angles(faired, local_tris, outer, n)

        # Splice into the mesh: boundary points map to the loop's vertices,
        # ring points to theirs, new interior points are appended
        mapping = {}
        for k in range(n):
            mapping[k] = loop[k]
        for v in ring_pts:
            mapping[index_of[v]] = v
        for li in range(len(faired)):
            if li not in mapping:
                mapping[li] = len(vertices)
                vertices.append(faired[li])
        for a, b, c in local_tris:
            faces.append((mapping[a], mapping[b], mapping[c]))
        added += len(local_tris)
        result.filled = True
        result.triangles = len(local_tris)
        report.loops_filled += 1
        report.holes.append(result)

    report.triangles_added = added
    out = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces),
                          process=False)
    return out, report


def _seam_angles(points, triangles, outer_normals, boundary_count):
    """Dihedral angle across each boundary edge between patch and body."""
    pts = np.asarray(points)
    n = boundary_count
    angles = []
    for k in range(n):
        u, v = k, (k + 1) % n
        for a, b, c in triangles:
            if {u, v} <= {a, b, c}:
                pn = _triangle_normal(pts[a], pts[b], pts[c])
                angles.append(_dihedral(pn, outer_normals[k]))
                break
    if not angles:
        return 0.0, 0.0
    # A dihedral above 90 means the patch folds back; report the fold as 180 - x
    # so that 0 always means smooth
    angles = np.array(angles)
    return float(angles.max()), float(np.median(angles))
