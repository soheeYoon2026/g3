"""Is this a half model?

A half car cannot be sealed into a closed volume no matter which wrapper runs -
the symmetry plane is a genuine hole the size of the whole silhouette, so every
tier walks in through it and hollows the result. That looks identical in the
report to "the holes are too big", which is why it is worth ruling in or out
before touching any wrap parameter.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--mesh", type=Path, required=True)
args = ap.parse_args()

mesh = trimesh.load(args.mesh, force="mesh")
v = np.asarray(mesh.vertices)

print("축별 범위 (mm)")
for axis, name in enumerate("xyz"):
    lo, hi = v[:, axis].min(), v[:, axis].max()
    print(f"  {name}: {lo:9.1f} ~ {hi:9.1f}   폭 {hi - lo:8.1f}")

print("\ny 분포 (정점 수)")
counts, edges = np.histogram(v[:, 1], bins=12)
peak = counts.max()
for i in range(len(counts)):
    bar = "#" * int(40 * counts[i] / peak)
    print(f"  {edges[i]:8.0f} ~ {edges[i + 1]:8.0f}  {bar} {counts[i]:,}")

# A full car is roughly symmetric about its own y centre; a half model piles up
# against one side, and that side is flat
centre = 0.5 * (v[:, 1].min() + v[:, 1].max())
left = int((v[:, 1] < centre).sum())
right = int((v[:, 1] >= centre).sum())
print(f"\n중심({centre:.1f}) 기준 좌 {left:,} / 우 {right:,}  "
      f"비대칭도 {abs(left - right) / len(v) * 100:.1f}%")

for side, value in (("y_min", v[:, 1].min()), ("y_max", v[:, 1].max())):
    near = np.abs(v[:, 1] - value) < 1.0
    print(f"  {side} 평면에 붙은 정점 {int(near.sum()):,}개 "
          f"({100 * near.mean():.1f}%) — 평면이면 대칭면일 가능성")

width = float(v[:, 1].max() - v[:, 1].min())
print(f"\n차 폭 {width:.0f} mm — 실차 전폭은 보통 1,700~2,000 mm")
print("절반이면 대칭면을 채우거나 미러링해야 어떤 래퍼로도 닫힌다" if width < 1400
      else "전폭으로 보인다")

# The decisive check: an open half model has a boundary loop lying in the
# symmetry plane, and its area is the whole silhouette - a hole no wrap closes.
for plane in (0.0, v[:, 1].max(), v[:, 1].min()):
    near = np.abs(v[:, 1] - plane) < 2.0
    if near.sum() < 50:
        continue
    pts = v[near]
    span_x = pts[:, 0].max() - pts[:, 0].min()
    span_z = pts[:, 2].max() - pts[:, 2].min()
    print(f"  y={plane:7.1f} 평면 근처 정점 {int(near.sum()):,}개, "
          f"x폭 {span_x:.0f} z폭 {span_z:.0f} — 실루엣 크기면 대칭면")
