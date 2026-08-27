"""How much of the geometry latent grid actually covers the car?

The encoder samples an SDF on interp_res cells spanning bounding_box_surface. A box
much larger than the car spends most of that resolution on empty air, and the
resulting cell has to be smaller than a deformation for the encoder to see it.
This measures the car's real extent against the boxes in use and reports the cell
size each one gives, next to the deformations the product asks about.
"""

import argparse
import glob
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv

BOXES = {
    "DrivAerML 기본": ([-1.5, -1.4, -0.32], [5.0, 1.4, 1.4]),
    "우리 학습 설정": ([-3.3, -1.5, -0.7], [4.6, 1.5, 5.1]),
}
RES = [128, 64, 64]

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--pairs", type=Path, required=True)
ap.add_argument("--sample", type=int, default=25)
args = ap.parse_args()

lo = np.full(3, np.inf)
hi = np.full(3, -np.inf)
run_dirs = sorted((p for p in args.root.glob("run_*") if p.is_dir()),
                  key=lambda p: int(p.name.split("_")[1]))[:args.sample]
for run_dir in run_dirs:
    tag = run_dir.name.split("_", 1)[1]
    bounds = np.asarray(pv.read(run_dir / f"boundary_{tag}.vtp").bounds).reshape(3, 2)
    lo = np.minimum(lo, bounds[:, 0])
    hi = np.maximum(hi, bounds[:, 1])
car = hi - lo
print(f"형상 {len(run_dirs)}개의 실제 범위")
print(f"  min {np.round(lo,3).tolist()}  max {np.round(hi,3).tolist()}")
print(f"  치수 {np.round(car,3).tolist()} m\n")

manifest = json.loads(args.pairs.read_text())
displacements = [p.get("max_displacement") for p in manifest.get("pairs", [])
                 if p.get("max_displacement")]
if displacements:
    d = np.array(displacements)
    print(f"변형 최대변위: 중앙값 {np.median(d)*100:.2f} cm  "
          f"사분위 {np.percentile(d,25)*100:.2f}~{np.percentile(d,75)*100:.2f} cm\n")
else:
    d = None

print(f"{'상자':>16s} {'폭 (m)':>22s} {'셀 크기 (cm)':>22s} {'차가 차지하는 비율':>18s}")
print("-" * 84)
rows = {}
for name, (bmin, bmax) in BOXES.items():
    span = np.asarray(bmax) - np.asarray(bmin)
    cell = span / np.asarray(RES) * 100
    fill = car / span
    rows[name] = cell
    print(f"{name:>16s} {str(np.round(span,2).tolist()):>22s} "
          f"{str(np.round(cell,2).tolist()):>22s} {str(np.round(fill,2).tolist()):>18s}")

tight_span = car * 1.05
tight_cell = tight_span / np.asarray(RES) * 100
print(f"{'차에 딱 맞춘 상자':>16s} {str(np.round(tight_span,2).tolist()):>22s} "
      f"{str(np.round(tight_cell,2).tolist()):>22s} {str([0.95]*3):>18s}")

if d is not None:
    print(f"\n변형(중앙값 {np.median(d)*100:.2f} cm)이 격자 한 칸의 몇 배인가")
    for name, cell in list(rows.items()) + [("차에 딱 맞춘 상자", tight_cell)]:
        ratio = np.median(d) * 100 / cell
        verdict = "보임" if ratio.min() >= 1.0 else "안 보임"
        print(f"  {name:>16s}: {np.round(ratio,2).tolist()}  -> {verdict}")
    print("\n한 축이라도 1보다 작으면 그 방향 변형은 SDF 격자에 잡히지 않는다.")
