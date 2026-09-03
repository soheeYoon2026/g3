"""Projected frontal area of a body, the way a CFD reference area is defined.

    frontal_area.py --in body.stl --mirror --ground-z -310

Every triangle is projected onto the plane normal to the flow and rasterised;
the union of the covered pixels is the area. Rasterising rather than summing
triangle areas is deliberate: the input overlaps itself (styling panels lie on
top of each other) and is not closed, and a sum would count the overlaps twice
and the mesh's back faces as well. A pixel is covered once no matter how many
triangles land on it.

Two things a reference area quietly depends on are exposed as options, because
BAIC's 2.52 m² will only be comparable if they match: whether the half model is
mirrored, and where the ground is - tyres below the ground line are not frontal
area. The bounding box W×H is printed alongside because the LES pipeline used
that as area_ref and the two differ by the corners.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="src", type=Path, required=True, nargs="+")
ap.add_argument("--flow", default="x", choices=("x", "y", "z"),
                help="flow direction; the projection plane is normal to it")
ap.add_argument("--mirror", action="store_true",
                help="mirror a half model about y=0 before projecting")
ap.add_argument("--ground-z", type=float,
                help="ignore everything below this height")
ap.add_argument("--pixel", type=float, default=0.5, help="raster pitch, mm")
ap.add_argument("--curvature-angle", type=float, default=15.0)
args = ap.parse_args()

from PIL import Image, ImageDraw  # noqa: E402
import trimesh  # noqa: E402


def load(path):
    if path.suffix.lower() in (".stl", ".obj", ".ply"):
        return trimesh.load(path, force="mesh")
    from aox_g3 import cad, topology
    shape, report = cad.read_step(path)
    if shape is None:
        raise SystemExit(f"읽기 실패: {report.warnings}")
    cad.diagnose(shape, report)
    shape, report = cad.sew_progressive(shape, report)
    return topology.tessellate_by_curvature(shape, args.curvature_angle)


axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[args.flow]

for path in args.src:
    mesh = load(path)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    if args.mirror:
        flipped = vertices.copy()
        flipped[:, 1] *= -1.0
        faces = np.vstack([faces, faces[:, ::-1] + len(vertices)])
        vertices = np.vstack([vertices, flipped])
    if args.ground_z is not None:
        # Clip at the ground: triangles entirely below it are dropped, and the
        # raster is cut at the ground line so partially-below ones stop there
        keep = (vertices[faces, 2] >= args.ground_z).any(axis=1)
        faces = faces[keep]

    uv = vertices[:, list(axes)]
    lo = uv.min(axis=0)
    hi = uv.max(axis=0)
    if args.ground_z is not None and axes[1] == 2:
        lo[1] = max(lo[1], args.ground_z)
    span = hi - lo
    width = int(np.ceil(span[0] / args.pixel)) + 2
    height = int(np.ceil(span[1] / args.pixel)) + 2

    canvas = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    px = (uv[:, 0] - lo[0]) / args.pixel + 1
    py = (hi[1] - uv[:, 1]) / args.pixel + 1
    for a, b, c in faces:
        draw.polygon([(px[a], py[a]), (px[b], py[b]), (px[c], py[c])], fill=1)
    covered = np.asarray(canvas, dtype=bool).sum()
    area = covered * args.pixel ** 2 / 1e6
    box = span[0] * span[1] / 1e6

    print(f"\n{path.name}")
    print(f"  삼각형 {len(faces):,}   투영면 {'yz' if args.flow == 'x' else 'xz' if args.flow == 'y' else 'xy'}   "
          f"픽셀 {args.pixel} mm   미러 {'예' if args.mirror else '아니오'}   "
          f"지면 {'없음' if args.ground_z is None else f'{args.ground_z:.0f}'}")
    print(f"  투영 폭 {span[0]:.0f} × 높이 {span[1]:.0f} mm   경계상자 W×H {box:.3f} m²")
    print(f"  전면 면적 (투영 합집합) {area:.3f} m²   상자 대비 {100 * area / box:.1f}%")
