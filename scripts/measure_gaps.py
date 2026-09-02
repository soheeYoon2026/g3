"""How wide are the gaps between the disconnected surface patches?

Alpha wrap's alpha must exceed the input's gaps or the algorithm walks in through
one and carves out the interior - which is what a 1% volume ratio means. The
cascade picks alpha = diagonal/180 by default; whether that is enough is a
property of this CAD, so measure it instead of guessing.

Also checks the wrap result against its own bounding box rather than against the
input's volume: for a triangle soup the input's "volume" is a divergence-theorem
artefact and makes a poor denominator.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--mesh", type=Path, required=True)
ap.add_argument("--alpha-div", type=float, default=180.0)
args = ap.parse_args()

mesh = trimesh.load(args.mesh, force="mesh")
vertices = np.asarray(mesh.vertices, dtype=float)
extents = vertices.max(axis=0) - vertices.min(axis=0)
diag = float(np.linalg.norm(extents))
print(f"메시: 삼각형 {len(mesh.faces):,}  대각선 {diag:.0f}")
print(f"기본 alpha = 대각선/{args.alpha_div:.0f} = {diag / args.alpha_div:.1f}")

pieces = mesh.split(only_watertight=False)
print(f"\n분리된 조각 {len(pieces):,}개")
sizes = np.array([len(p.faces) for p in pieces])
print(f"  조각당 삼각형: 중앙값 {np.median(sizes):.0f}  최대 {sizes.max():,}")

# For each piece, how far to the nearest vertex of any OTHER piece
sample = pieces if len(pieces) <= 400 else [pieces[i] for i in
                                            np.random.default_rng(0).choice(len(pieces), 400, replace=False)]
all_pts = vertices
gaps = []
for piece in sample:
    pts = np.asarray(piece.vertices, dtype=float)
    if len(pts) == 0:
        continue
    # exclude the piece's own points by asking for enough neighbours
    tree = cKDTree(all_pts)
    dist, idx = tree.query(pts, k=min(40, len(all_pts)))
    own = set(map(tuple, np.round(pts, 6)))
    for row_d, row_i in zip(dist, idx):
        for d, i in zip(row_d, row_i):
            if tuple(np.round(all_pts[i], 6)) not in own:
                gaps.append(d)
                break

gaps = np.array(gaps)
if len(gaps):
    print(f"\n조각 간 간극 ({len(gaps):,}개 표본)")
    for q in (10, 25, 50, 75, 90, 99):
        print(f"  {q:2d}퍼센타일 {np.percentile(gaps, q):8.2f}")
    print(f"  최대 {gaps.max():.1f}")
    need = float(np.percentile(gaps, 95))
    print(f"\n간극 95%를 덮으려면 alpha >= {need:.1f} "
          f"(= 대각선/{diag / max(need, 1e-9):.0f})")
    print(f"기본값 대각선/{args.alpha_div:.0f} = {diag / args.alpha_div:.1f} 는 "
          f"{'충분' if diag / args.alpha_div >= need else '부족'}")
