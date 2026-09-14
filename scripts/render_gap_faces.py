"""Paint the faces a closed surface added across gaps, so the seams can be seen.

    render_gap_faces.py --reference original.stl --candidate closed.stl --out gaps.png

A face of the candidate whose centroid is farther than --web-distance from the
reference is material the closing step added: the bridge over a panel seam, the
assumed floor, a sealed slot. Those faces are painted red on four views of the
whole car, the largest patches are numbered, and each of the --closeups largest
patches gets its own close-up from the direction it faces. Same z-buffer renderer
as render_geometry.py (numpy, no display).
"""
import argparse
from pathlib import Path

import igl
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--reference", type=Path, required=True)
ap.add_argument("--candidate", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True, help="four-view picture; close-ups go next to it as <out>_closeup_N.png")
ap.add_argument("--web-distance", type=float, default=1.5)
ap.add_argument("--min-area", type=float, default=200.0, help="cm²; smaller patches are painted but not numbered")
ap.add_argument("--label-top", type=int, default=10)
ap.add_argument("--closeups", type=int, default=6)
ap.add_argument("--closeup-span", type=float, default=0.0, help="mm; 0 = 1.25x the patch's largest extent, capped at 1600")
ap.add_argument("--skip-overview", action="store_true")
ap.add_argument("--width", type=int, default=760)
ap.add_argument("--height", type=int, default=520)
ap.add_argument("--title", default="")
args = ap.parse_args()


def load_font(size):
    for cand in ("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def basis(direction, up):
    forward = direction / np.linalg.norm(direction)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return right, true_up, forward


def project(points, right, true_up, forward, centre, scale, size):
    local = points - centre
    width, height = size
    px = width * 0.5 + (local @ right) * scale
    py = height * 0.5 - (local @ true_up) * scale
    return np.stack([px, py], axis=1), local @ forward


def rasterise(V, F, N, colours, view, size, scale, centre, light=np.array([-0.4, -0.6, 0.7])):
    right, true_up, forward = basis(*view)
    width, height = size
    screen, depth_v = project(V, right, true_up, forward, centre, scale, size)
    image = np.full((height, width, 3), 246, dtype=np.uint8)
    zbuf = np.full((height, width), np.inf)
    light = light / np.linalg.norm(light)
    shade = 0.30 + 0.70 * np.abs(N @ light)
    tri = screen[F]
    dep = depth_v[F]
    # skip triangles fully outside the picture
    keep = (tri[:, :, 0].max(1) >= 0) & (tri[:, :, 0].min(1) < width) & (tri[:, :, 1].max(1) >= 0) & (tri[:, :, 1].min(1) < height)
    order = np.nonzero(keep)[0]
    order = order[np.argsort(-dep[order].mean(axis=1))]
    dark = np.array([40, 40, 44])
    for index in order:
        pts = tri[index]; zs = dep[index]
        x0 = max(int(np.floor(pts[:, 0].min())), 0); x1 = min(int(np.ceil(pts[:, 0].max())) + 1, width)
        y0 = max(int(np.floor(pts[:, 1].min())), 0); y1 = min(int(np.ceil(pts[:, 1].max())) + 1, height)
        if x1 <= x0 or y1 <= y0:
            continue
        (ax, ay), (bx, by), (cx, cy) = pts
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
        base = colours[index] * tone + dark * (1 - tone)
        image[y0:y1, x0:x1][nearer] = np.clip(base, 0, 255).astype(np.uint8)
    return image


ref = trimesh.load(args.reference, force="mesh")
cand = trimesh.load(args.candidate, force="mesh")
RV = np.ascontiguousarray(ref.vertices, np.float64); RF = np.ascontiguousarray(ref.faces, np.int64)
C = np.ascontiguousarray(cand.triangles_center, np.float64)
d = np.sqrt(igl.point_mesh_squared_distance(C, RV, RF)[0])
gap = d > args.web_distance
A = cand.area_faces
print(f"후보 {len(cand.faces):,}면, 덧댄 면 {int(gap.sum()):,} ({A[gap].sum()/A.sum()*100:.1f} % 면적)")

# patches = connected components of the gap faces
adj = cand.face_adjacency
both = gap[adj[:, 0]] & gap[adj[:, 1]]
labels = trimesh.graph.connected_component_labels(adj[both], node_count=len(cand.faces))
labels[~gap] = -1
ids, inv = np.unique(labels[gap], return_inverse=True)
areas = np.bincount(inv, weights=A[gap])
order = np.argsort(areas)[::-1]
patches = []
for k in order:
    if areas[k] / 100.0 < args.min_area:
        break
    sel = np.nonzero(labels == ids[k])[0]
    centre = (C[sel] * A[sel, None]).sum(0) / A[sel].sum()
    normal = (cand.face_normals[sel] * A[sel, None]).sum(0)
    n = np.linalg.norm(normal)
    normal = normal / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    extent = C[sel].max(0) - C[sel].min(0)
    patches.append(dict(faces=sel, area_cm2=areas[k] / 100.0, centre=centre, normal=normal, extent=extent,
                        gap_mm=2.0 * float(d[sel].max())))
print("번호  면적 cm²   중심 (x, y, z)            크기 (mm)          건너뛴 틈 ≈")
for i, p in enumerate(patches[: args.label_top], 1):
    print(f"{i:3d}  {p['area_cm2']:8.0f}   ({p['centre'][0]:6.0f},{p['centre'][1]:6.0f},{p['centre'][2]:5.0f})   "
          f"{p['extent'][0]:5.0f}×{p['extent'][1]:5.0f}×{p['extent'][2]:4.0f}   {p['gap_mm']:5.0f} mm")

V = np.asarray(cand.vertices, float); F = np.asarray(cand.faces); N = np.asarray(cand.face_normals, float)
colours = np.tile(np.array([118, 134, 152], float), (len(F), 1))
colours[gap] = np.array([225, 62, 48], float)
lo, hi = cand.bounds; centre = 0.5 * (lo + hi)
VIEWS = [("앞쪽 3/4", np.array([-1.0, -0.8, -0.55]), np.array([0, 0, 1.0])),
         ("옆면", np.array([0.0, -1.0, 0.0]), np.array([0, 0, 1.0])),
         ("아래에서", np.array([0.0, 0.0, 1.0]), np.array([1.0, 0, 0])),
         ("뒤쪽 3/4", np.array([1.0, 0.8, -0.55]), np.array([0, 0, 1.0]))]
size = (args.width, args.height)
title_font = load_font(19); label_font = load_font(15); small_font = load_font(13)


def frame(view, lo_, hi_, centre_, faces_mask=None):
    direction, up = view
    right, true_up, forward = basis(direction, up)
    corners = np.array([[x, y, z] for x in (lo_[0], hi_[0]) for y in (lo_[1], hi_[1]) for z in (lo_[2], hi_[2])]) - centre_
    span_x = (corners @ right).max() - (corners @ right).min()
    span_y = (corners @ true_up).max() - (corners @ true_up).min()
    scale = 0.86 * min(args.width / max(span_x, 1e-9), args.height / max(span_y, 1e-9))
    if faces_mask is None:
        Fm, Nm, Cm = F, N, colours
    else:
        Fm, Nm, Cm = F[faces_mask], N[faces_mask], colours[faces_mask]
    img = Image.fromarray(rasterise(V, Fm, Nm, Cm, view, size, scale, centre_))
    return img, (right, true_up, forward, scale)


def annotate(img, cam, centre_, pts_labels):
    right, true_up, forward, scale = cam
    draw = ImageDraw.Draw(img)
    for text, p in pts_labels:
        (px, py), _ = project(np.asarray([p], float), right, true_up, forward, centre_, scale, size)[0][0], None
        if 0 <= px < args.width and 0 <= py < args.height:
            draw.ellipse([px - 11, py - 11, px + 11, py + 11], fill=(255, 255, 255), outline=(120, 20, 10), width=2)
            draw.text((px - 5 if len(text) == 1 else px - 9, py - 9), text, fill=(120, 20, 10), font=label_font)


tiles = []
labels_pts = [(str(i), p["centre"]) for i, p in enumerate(patches[: args.label_top], 1)]
for name, direction, up in ([] if args.skip_overview else VIEWS):
    img, cam = frame((direction, up), lo, hi, centre)
    annotate(img, cam, centre, labels_pts)
    ImageDraw.Draw(img).text((10, 8), name, fill=(60, 60, 60), font=label_font)
    tiles.append(img)
    print(f"  {name} 완료")
W, H = size
sheet = Image.new("RGB", (2 * W + 30, 2 * H + 70), (255, 255, 255)) if tiles else None
args.out.parent.mkdir(parents=True, exist_ok=True)
if sheet is not None:
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), args.title, fill=(30, 30, 30), font=title_font)
    draw.text((10, 34), f"빨강 = 닫힌 면 중 원본에서 {args.web_distance:g} mm 넘게 떨어진 면(틈을 건너뛰어 덧댄 자리), 번호 = 면적 순위", fill=(90, 90, 90), font=small_font)
    for i, t in enumerate(tiles):
        sheet.paste(t, (10 + (i % 2) * (W + 10), 60 + (i // 2) * (H + 10)))
    sheet.save(args.out)
    print(f"저장 {args.out}")

# close-ups of the largest patches, looking along minus the patch normal, faces outside the box culled
for i, p in enumerate(patches[: args.closeups], 1):
    span = args.closeup_span or float(np.clip(1.25 * p["extent"].max(), 300.0, 1600.0))
    c = p["centre"]; lo_ = c - span / 2; hi_ = c + span / 2
    n = p["normal"]
    direction = -n if np.linalg.norm(n) > 0 else np.array([0, -1.0, 0])
    if abs(direction[2]) > 0.9:
        up = np.array([1.0, 0, 0])
    else:
        up = np.array([0, 0, 1.0])
    # tilt a little so the surface reads as a surface, not a silhouette
    direction = direction + 0.35 * up + 0.25 * np.cross(up, direction)
    inside = np.all(np.abs(C - c) < span * 0.75, axis=1)
    img, cam = frame((direction, up), lo_, hi_, c, faces_mask=inside)
    ImageDraw.Draw(img).text((10, 8), f"{i}번  면적 {p['area_cm2']:.0f} cm²  중심 ({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f})  틈 ≈ {p['gap_mm']:.0f} mm", fill=(60, 60, 60), font=small_font)
    out = args.out.with_name(args.out.stem + f"_closeup_{i}.png")
    img.save(out)
    print(f"  근접 {i}: {out.name}")
