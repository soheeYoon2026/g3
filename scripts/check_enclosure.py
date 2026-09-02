"""Does this surface enclose a volume at all?

Every alpha produced a watertight wrap holding 1-3% of the bounding box, and
sewing the B-rep changed the wrap's volume by less than a thousandth. That points
away from "the gaps are too wide" and toward the input not bounding a volume in
the first place - a wrap around a sheet is watertight too, it is just a flattened
balloon.

The test is a ray cast from points inside the silhouette: a surface that encloses
a volume gives an odd crossing count from an interior point, and an even one from
outside. A sheet gives even counts everywhere.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--mesh", type=Path, required=True)
ap.add_argument("--samples", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

mesh = trimesh.load(args.mesh, force="mesh")
lo, hi = mesh.bounds
print(f"삼각형 {len(mesh.faces):,}  표면적 {mesh.area / 1e6:.2f} m2")
print(f"경계상자 {np.round(hi - lo, 0).tolist()} mm")

rng = np.random.default_rng(args.seed)
# sample the middle of the box, where a car body should be solid
span = hi - lo
points = lo + span * rng.uniform(0.3, 0.7, size=(args.samples, 3))
direction = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(points), 1))

locations, index_ray, _ = mesh.ray.intersects_location(
    ray_origins=points, ray_directions=direction, multiple_hits=True)
counts = np.bincount(index_ray, minlength=len(points))

odd = int((counts % 2 == 1).sum())
zero = int((counts == 0).sum())
print(f"\n중앙부 {args.samples}점에서 +x 방향 광선")
print(f"  교차 홀수(내부로 판정) {odd:4d}  ({100 * odd / args.samples:.0f}%)")
print(f"  교차 짝수            {args.samples - odd:4d}")
print(f"  교차 0회             {zero:4d}")
print(f"  평균 교차 횟수 {counts.mean():.1f}  최대 {counts.max()}")

print()
if odd > 0.25 * args.samples:
    print("→ 표면이 부피를 감싼다. 봉합 실패 원인은 다른 데 있다.")
else:
    print("→ 표면이 부피를 감싸지 않는다 (판 형상). 어떤 래퍼도 얇은 껍질만 만든다.")
    print("  차체를 닫으려면 빠진 면을 채우거나, 솔버 쪽에서 지면·대칭면으로 막아야 한다.")
print(f"  참고: 평균 교차가 4회 이상이면 내부 패널이 여러 겹이라는 뜻 "
      f"(표면적 {mesh.area / 1e6:.1f} m2 가 외피치고 크면 그 방증)")
