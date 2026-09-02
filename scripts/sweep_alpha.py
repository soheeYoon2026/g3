"""Find the alpha at which this CAD actually closes, and what it costs.

Alpha must exceed the input's gaps or the wrap walks in and hollows the result;
raise it further and narrow features (diffuser fins, wheel spokes, the slot under
a wing) get bridged over. The usable value is the smallest one that closes, so
sweep upward and stop at the first success rather than picking a safe-looking
large number.

Reports volume against the wrap's own bounding box, not against the input soup's
volume - for a triangle soup that number is a divergence-theorem artefact.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh


def alpha_wrap(mesh, alpha, offset_frac=1.0 / 30.0):
    from CGAL import CGAL_Alpha_wrap_3 as AW
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    import os
    import tempfile

    points = AW.Point_3_Vector()
    points.reserve(len(mesh.vertices))
    for v in np.asarray(mesh.vertices, dtype=float):
        points.append(Point_3(float(v[0]), float(v[1]), float(v[2])))
    polygons = AW.Polygon_Vector()
    polygons.reserve(len(mesh.faces))
    for f in np.asarray(mesh.faces):
        iv = AW.Int_Vector()
        iv.reserve(3)
        for k in f:
            iv.append(int(k))
        polygons.append(iv)
    out = Polyhedron_3()
    AW.alpha_wrap_3(points, polygons, float(alpha), float(alpha * offset_frac), out)
    handle, tmp = tempfile.mkstemp(suffix=".off")
    os.close(handle)
    out.write_to_file(tmp)
    wrapped = trimesh.load(tmp, force="mesh")
    os.unlink(tmp)
    return wrapped


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--mesh", type=Path, required=True)
ap.add_argument("--divisors", type=float, nargs="+", default=[180, 140, 110, 90, 70, 50])
ap.add_argument("--out", type=Path)
args = ap.parse_args()

mesh = trimesh.load(args.mesh, force="mesh")
extents = np.asarray(mesh.extents, dtype=float)
diag = float(np.linalg.norm(extents))
box = float(extents[0] * extents[1] * extents[2])
print(f"입력: 삼각형 {len(mesh.faces):,}  대각선 {diag:.0f}  경계상자 {box / 1e9:.2f} m3")
print(f"차 한 대라면 체적은 상자의 30~45% 정도가 정상\n")

print(f"{'alpha':>8s} {'대각선/':>8s} {'초':>6s} {'삼각형':>10s} {'수밀':>5s} "
      f"{'체적 m3':>9s} {'상자대비':>8s}")
print("-" * 62)
best = None
for div in args.divisors:
    alpha = diag / div
    t = time.time()
    try:
        wrapped = alpha_wrap(mesh, alpha)
    except Exception as exc:
        print(f"{alpha:8.1f} {div:8.0f}   실패 {type(exc).__name__}")
        continue
    elapsed = time.time() - t
    volume = abs(float(wrapped.volume))
    fill = volume / box if box > 0 else 0.0
    ok = wrapped.is_watertight and fill > 0.15
    print(f"{alpha:8.1f} {div:8.0f} {elapsed:6.1f} {len(wrapped.faces):10,} "
          f"{str(wrapped.is_watertight):>5s} {volume / 1e9:9.3f} {100 * fill:7.1f}%")
    if ok and best is None:
        best = (alpha, div, wrapped)

if best:
    alpha, div, wrapped = best
    print(f"\n가장 작은 성공 alpha = {alpha:.1f} (대각선/{div:.0f})")
    print(f"  이보다 크면 미세 형상이 더 메워지고, 작으면 속이 빈다")
    if args.out:
        wrapped.export(args.out)
        print(f"  wrote {args.out}")
else:
    print("\n어떤 alpha 로도 닫히지 않았다")
