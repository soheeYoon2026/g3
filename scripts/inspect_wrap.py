"""Is the alpha wrap a solid car body, or a shell around the panels?

Both are watertight, so the flag cannot tell them apart. Two things can:
surface area (a shell has roughly twice the input's, because it covers both
sides) and volume against the bounding box (a car body fills 30-45%, a shell a
few percent). Winding is checked too, since a wrap loaded from OFF with reversed
orientation reports a small or negative volume for a perfectly good solid.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--wrap", type=Path, required=True)
ap.add_argument("--input", type=Path, help="the mesh that was wrapped, for comparison")
args = ap.parse_args()

wrap = trimesh.load(args.wrap, force="mesh")
extents = np.asarray(wrap.extents, dtype=float)
box = float(extents[0] * extents[1] * extents[2])

print(f"래핑: 삼각형 {len(wrap.faces):,}  수밀 {wrap.is_watertight}")
print(f"  치수 {np.round(extents, 0).tolist()} mm   경계상자 {box / 1e9:.2f} m3")
print(f"  체적 {wrap.volume / 1e9:+.3f} m3  →  상자 대비 {100 * abs(wrap.volume) / box:.1f}%")
print(f"  표면적 {wrap.area / 1e6:.2f} m2")

fixed = wrap.copy()
trimesh.repair.fix_winding(fixed)
trimesh.repair.fix_normals(fixed)
print(f"  감김·법선 수정 후 체적 {fixed.volume / 1e9:+.3f} m3 "
      f"({100 * abs(fixed.volume) / box:.1f}%)")

if args.input and args.input.exists():
    source = trimesh.load(args.input, force="mesh")
    ratio = wrap.area / source.area if source.area else float("nan")
    print(f"\n입력 표면적 {source.area / 1e6:.2f} m2  →  래핑/입력 = {ratio:.2f}")
    print("  ~1.0 이면 고체를 감쌌고, ~2.0 이면 판 양면을 감싼 껍질이다")

print()
fill = abs(fixed.volume) / box
if fill > 0.20:
    print("→ 차체 고체로 보인다")
elif wrap.area > 1.6 * (trimesh.load(args.input, force='mesh').area
                        if args.input and args.input.exists() else 0):
    print("→ 얇은 껍질이다 (판의 양면을 감쌌다)")
else:
    print("→ 고체도 껍질도 아닌 중간 — 내부에 큰 빈 공간이 있다는 뜻")
