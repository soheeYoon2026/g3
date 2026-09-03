"""Render the body and draw its holes on it, so the report can be looked at.

Numbers say CAS-A has 46 openings and that the largest spans 4.85 m, but a list of
sizes and centroids does not show that the missing pieces are the floor, the glass
and the wheel arches. A picture does, and the decision that is currently blocking
the pipeline - what the underbody should be - is one a person makes by looking.

Rendered here rather than through a viewer because the geometry is confidential and
stays on this machine. A z-buffer and flat shading in numpy is enough for that, and
avoids needing a display or a GPU.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from aox_g3 import brep, cad, topology  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--step", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--width", type=int, default=760)
ap.add_argument("--height", type=int, default=520)
ap.add_argument("--no-holes", action="store_true")
ap.add_argument("--mirror", action="store_true",
                help="mirror the half model about y=0 for the picture only")
ap.add_argument("--label-top", type=int, default=6,
                help="give this many of the largest holes their own colour")
ap.add_argument("--focus-hole", type=int,
                help="centre the views on the Nth largest hole (1 = largest)")
ap.add_argument("--focus-point", metavar="X,Y,Z",
                help="centre the views on this point (for a filled hole, which is "
                     "no longer in the list)")
ap.add_argument("--focus-span", type=float,
                help="how much of the model to show around the focus point")
ap.add_argument("--title", default="")
args = ap.parse_args()


def load_font(size):
    """A CJK face, because the labels are Korean and DejaVu draws them as boxes."""
    from PIL import ImageFont
    for path in ("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Medium.ttc",
                 "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# Distinct hues for the largest openings, so each can be named in the legend
# rather than described by a centroid
HUES = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (148, 103, 189),
        (255, 127, 14), (23, 190, 207), (227, 119, 194), (188, 189, 34)]
REST = (120, 120, 128)


VIEWS = [
    ("앞쪽 3/4", np.array([-0.75, -0.55, 0.36]), np.array([0.0, 0.0, 1.0])),
    ("옆면", np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    ("아래에서", np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])),
    ("뒤쪽 3/4", np.array([0.75, -0.55, 0.30]), np.array([0.0, 0.0, 1.0])),
]


def basis(direction, up):
    """Camera axes: forward toward the model, right and up across the image."""
    forward = direction / np.linalg.norm(direction)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return right, true_up, forward


def project(points, right, true_up, forward, centre, scale, size):
    local = points - centre
    x = local @ right
    y = local @ true_up
    z = local @ forward
    width, height = size
    px = width * 0.5 + x * scale
    # image rows run downward, so the up axis is negated
    py = height * 0.5 - y * scale
    return np.stack([px, py], axis=1), z


def rasterise(mesh, view, size, scale, centre, light=np.array([-0.4, -0.6, 0.7])):
    """Flat-shaded z-buffer render. Returns an RGB array and the depth buffer."""
    right, true_up, forward = basis(*view)
    width, height = size
    screen, depth_v = project(np.asarray(mesh.vertices, dtype=float),
                              right, true_up, forward, centre, scale, size)

    colour = np.full((height, width, 3), 246, dtype=np.uint8)
    zbuf = np.full((height, width), np.inf)

    light = light / np.linalg.norm(light)
    normals = np.asarray(mesh.face_normals, dtype=float)
    shade = np.abs(normals @ light)
    # A flat lambert term alone reads as a silhouette; lifting the floor keeps the
    # panel breaks visible without washing the shape out
    shade = 0.30 + 0.70 * shade

    tri_screen = screen[mesh.faces]
    tri_depth = depth_v[mesh.faces]
    # Painter order is not enough where surfaces interpenetrate, which this model
    # does 8,830 times, so depth is compared per pixel
    order = np.argsort(-tri_depth.mean(axis=1))

    for index in order:
        pts = tri_screen[index]
        zs = tri_depth[index]
        x0 = max(int(np.floor(pts[:, 0].min())), 0)
        x1 = min(int(np.ceil(pts[:, 0].max())) + 1, width)
        y0 = max(int(np.floor(pts[:, 1].min())), 0)
        y1 = min(int(np.ceil(pts[:, 1].max())) + 1, height)
        if x1 <= x0 or y1 <= y0:
            continue

        ax, ay = pts[0]
        bx, by = pts[1]
        cx, cy = pts[2]
        area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if abs(area) < 1e-12:
            continue

        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        w0 = ((bx - ax) * (gy - ay) - (gx - ax) * (by - ay)) / area
        w1 = ((gx - ax) * (cy - ay) - (cx - ax) * (gy - ay)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w2 * zs[0] + w1 * zs[1] + w0 * zs[2]

        window = zbuf[y0:y1, x0:x1]
        nearer = inside & (z < window)
        if not nearer.any():
            continue
        window[nearer] = z[nearer]
        tone = shade[index]
        base = np.array([118, 134, 152]) * tone + np.array([40, 40, 44]) * (1 - tone)
        colour[y0:y1, x0:x1][nearer] = np.clip(base, 0, 255).astype(np.uint8)
    return colour, zbuf


def hole_lines(shape):
    """Every free boundary as polylines, with the size that classifies it."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS

    holes, _ = brep.free_boundaries(shape)
    out = []
    for rank, (boundary, wire) in enumerate(holes):
        for e in topology._explore(wire, TopAbs_EDGE):
            edge = TopoDS.Edge_s(e)
            curve = BRepAdaptor_Curve(edge)
            first, last = BRep_Tool.Range_s(edge)
            pts = []
            for t in np.linspace(first, last, 12):
                p = curve.Value(float(t))
                pts.append((p.X(), p.Y(), p.Z()))
            out.append((rank, boundary.size, np.asarray(pts)))
    return out, holes


if args.step.suffix.lower() in (".stl", ".obj", ".ply"):
    # A mesh has no B-rep to query; its open loops are drawn from the triangles
    import trimesh
    mesh = trimesh.load(args.step, force="mesh")
    holes, lines = [], []
    if not args.no_holes:
        from aox_g3 import fair as _fair
        loops, _ = _fair.boundary_loops(mesh)
        for rank, loop in enumerate(sorted(
                loops, key=lambda l: -np.linalg.norm(
                    mesh.vertices[l].max(0) - mesh.vertices[l].min(0)))):
            pts = np.asarray(mesh.vertices[loop + [loop[0]]], dtype=float)
            size = float(np.linalg.norm(pts.max(0) - pts.min(0)))
            lines.append((rank, size, pts))
            holes.append((brep.Boundary(size=size, length=float(
                np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()),
                centre=tuple(pts.mean(0))), None))
else:
    shape, report = cad.read_step(args.step)
    if shape is None:
        raise SystemExit(f"읽기 실패: {report.warnings}")
    cad.diagnose(shape, report)
    shape, report = cad.sew_progressive(shape, report)
    mesh = topology.tessellate_by_curvature(shape, 20.0)
    lines, holes = ([], []) if args.no_holes else hole_lines(shape)
print(f"삼각형 {len(mesh.faces):,}   구멍 {len(holes)}개   "
     f"경계상자 {np.round(np.asarray(mesh.bounds[1]) - np.asarray(mesh.bounds[0]), 0)}")

if args.mirror:
    flipped = mesh.copy()
    flipped.vertices[:, 1] *= -1.0
    flipped.faces = flipped.faces[:, ::-1]
    mesh = mesh + flipped
    lines = lines + [(rank, size, pts * np.array([1.0, -1.0, 1.0]))
                     for rank, size, pts in lines]

lo = np.asarray(mesh.bounds[0], dtype=float)
hi = np.asarray(mesh.bounds[1], dtype=float)
centre = 0.5 * (lo + hi)
extent = float(np.linalg.norm(hi - lo))

# Zooming in on one opening is how a hole list becomes a thing you can identify:
# a frame with nothing behind it and a frame with a panel sitting 4 mm back look
# identical in a table of sizes and centroids.
if args.focus_point:
    centre = np.array([float(v) for v in
                       args.focus_point.replace(" ", "").split(",")])
    span = args.focus_span or extent * 0.25
    lo = centre - span * 0.5
    hi = centre + span * 0.5
    print(f"확대: 점 {np.round(centre, 0)}  범위 {span:.0f}")
elif args.focus_hole and holes:
    picked = holes[min(args.focus_hole, len(holes)) - 1][0]
    centre = np.asarray(picked.centre, dtype=float)
    span = args.focus_span or (2.0 * picked.size)
    lo = centre - span * 0.5
    hi = centre + span * 0.5
    print(f"확대: {args.focus_hole}번째 구멍 크기 {picked.size:.0f}  "
          f"중심 {np.round(centre, 0)}  범위 {span:.0f}")

size = (args.width, args.height)
title_font = load_font(19)
label_font = load_font(16)
tiles = []
for name, direction, up in VIEWS:
    right, true_up, forward = basis(direction, up)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    local = corners - centre
    span_x = (local @ right).ptp() if hasattr(local @ right, "ptp") else \
        (local @ right).max() - (local @ right).min()
    span_y = (local @ true_up).max() - (local @ true_up).min()
    scale = 0.86 * min(args.width / max(span_x, 1e-9),
                       args.height / max(span_y, 1e-9))

    colour, _ = rasterise(mesh, (direction, up), size, scale, centre)
    image = Image.fromarray(colour)
    draw = ImageDraw.Draw(image)

    # Draw the small ones first so the named openings are not painted over
    for rank, hole_size, pts in sorted(lines, key=lambda row: row[1]):
        screen, _ = project(pts, right, true_up, forward, centre, scale, size)
        named = rank < args.label_top
        draw.line([tuple(p) for p in screen],
                  fill=HUES[rank % len(HUES)] if named else REST,
                  width=4 if named else 2)

    draw.rectangle([0, 0, args.width - 1, args.height - 1], outline=(190, 190, 194))
    draw.text((14, 10), name, fill=(30, 30, 34), font=label_font)
    tiles.append(image)

gap = 10
legend_h = 26 + 22 * min(args.label_top, len(holes)) if holes else 0
sheet = Image.new("RGB",
                  (args.width * 2 + gap * 3,
                   args.height * 2 + gap * 3 + 34 + legend_h),
                  (255, 255, 255))
for i, tile in enumerate(tiles):
    sheet.paste(tile, (gap + (i % 2) * (args.width + gap),
                       gap + 34 + (i // 2) * (args.height + gap)))

draw = ImageDraw.Draw(sheet)
draw.text((gap + 4, 10), args.title or args.step.name, fill=(20, 20, 24),
          font=title_font)

if holes:
    y = args.height * 2 + gap * 3 + 40
    draw.text((gap + 4, y - 18), f"자유경계 {len(holes)}개 — 큰 것부터",
              fill=(60, 60, 66), font=label_font)
    for rank, (boundary, _) in enumerate(holes[:args.label_top]):
        row = y + 4 + rank * 22
        draw.rectangle([gap + 6, row + 4, gap + 30, row + 12],
                       fill=HUES[rank % len(HUES)])
        cx, cy, cz = boundary.centre
        draw.text((gap + 40, row),
                  f"크기 {boundary.size:7.0f}   둘레 {boundary.length:8.0f}   "
                  f"중심 ({cx:7.0f}, {cy:6.0f}, {cz:6.0f})",
                  fill=(30, 30, 34), font=label_font)
    row = y + 4 + args.label_top * 22
    draw.rectangle([gap + 6, row + 4, gap + 30, row + 12], fill=REST)
    draw.text((gap + 40, row), f"나머지 {max(len(holes) - args.label_top, 0)}개",
              fill=(30, 30, 34), font=label_font)

sheet.save(args.out)
print(f"저장: {args.out}  ({sheet.size[0]}x{sheet.size[1]})")
