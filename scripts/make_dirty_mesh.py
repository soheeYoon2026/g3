"""Punch holes in a watertight mesh so the seal can be checked against a known answer.

Testing a sealer on already-dirty geometry tells you it produced *something*
closed; it cannot tell you whether the shape survived. Starting from a watertight
mesh and removing faces gives a ground-truth volume to compare against.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--source", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--holes", type=int, default=6)
ap.add_argument("--radius-frac", type=float, default=0.06,
                help="hole radius as a fraction of the bounding-box diagonal")
ap.add_argument("--seed", type=int, default=20260902)
args = ap.parse_args()

mesh = trimesh.load(args.source, force="mesh")
print(f"원본: 삼각형 {len(mesh.faces):,}  수밀 {mesh.is_watertight}  "
      f"체적 {abs(mesh.volume):.4f}")

centres = np.asarray(mesh.triangles_center, dtype=float)
diag = float(np.linalg.norm(np.asarray(mesh.extents, dtype=float)))
radius = diag * args.radius_frac

rng = np.random.default_rng(args.seed)
picked = rng.choice(len(centres), size=args.holes, replace=False)
drop = np.zeros(len(centres), dtype=bool)
for idx in picked:
    drop |= np.linalg.norm(centres - centres[idx], axis=1) < radius

keep = ~drop
dirty = trimesh.Trimesh(vertices=mesh.vertices,
                        faces=np.asarray(mesh.faces)[keep], process=False)
dirty.remove_unreferenced_vertices()
dirty.export(args.out)
print(f"구멍 {args.holes}개 (반경 {radius:.1f}), 삼각형 {int(drop.sum()):,}개 제거")
print(f"결과: 삼각형 {len(dirty.faces):,}  수밀 {dirty.is_watertight}")
print(f"정답 체적 {abs(mesh.volume):.4f} — 봉합 결과가 이 값에 가까워야 한다")
