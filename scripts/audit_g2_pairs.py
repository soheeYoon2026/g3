"""Audit the stored G2 deformation pairs for usable ΔCd labels.

The pair set is large but its deltas run small, and a label below the solver's
own noise floor carries no direction information. G2 on a watertight
solver-derived surface resolves about |ΔCd| >= 0.0015 (three times the measured
run-to-run scatter), so this reports how many pairs clear that and how the
survivors split between drag up and drag down -- a benchmark that is nearly all
one direction cannot separate a real model from one that always answers the same
way.
"""

import argparse
import glob
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--noise-floor", type=float, default=0.0015)
args = ap.parse_args()

rows = []
for path in sorted(glob.glob(str(args.root / "*" / "RBF_DSN_*.npz"))):
    try:
        data = np.load(path, allow_pickle=True)
        rows.append({
            "job": Path(path).parent.name,
            "design": Path(path).stem,
            "base_cd": float(data["base_cd"]),
            "target_cd": float(data["target_cd"]),
            "delta_cd": float(data["delta_cd"]),
            "points": int(data["base_points"].shape[0]),
            "moved": int(data["moved_mask"].sum()),
            "max_disp": float(np.linalg.norm(data["displacement"], axis=1).max()),
        })
    except Exception as exc:
        print(f"{path}: 읽기 실패 {type(exc).__name__}")

print(f"쌍 {len(rows)}개\n")
delta = np.array([r["delta_cd"] for r in rows])
base = np.array([r["base_cd"] for r in rows])
finite = np.isfinite(delta) & np.isfinite(base)
print(f"유한 라벨 {int(finite.sum())}개")
delta, base = delta[finite], base[finite]
rows = [r for r, ok in zip(rows, finite) if ok]

print(f"base_cd  {base.min():.4f} ~ {base.max():.4f}  (중앙값 {np.median(base):.4f})")
print(f"|ΔCd|    중앙값 {np.median(np.abs(delta)):.5f}  최대 {np.abs(delta).max():.5f}")

print("\n노이즈 기준별 생존 쌍")
for floor in (0.0015, 0.003, 0.005, 0.010):
    keep = np.abs(delta) >= floor
    up, down = int((delta[keep] > 0).sum()), int((delta[keep] < 0).sum())
    print(f"  |ΔCd| >= {floor:.4f}: {int(keep.sum()):3d}쌍  "
          f"(증가 {up} / 감소 {down})")

keep = np.abs(delta) >= args.noise_floor
survivors = [r for r, ok in zip(rows, keep) if ok]
jobs = sorted({r["job"] for r in survivors})
print(f"\n기준 {args.noise_floor}: {len(survivors)}쌍, 서로 다른 형상 {len(jobs)}종")
print(f"  변위 최대 {np.median([r['max_disp'] for r in survivors]):.4f} m (중앙값), "
      f"이동점 {np.median([r['moved'] for r in survivors]):.0f}개")
print(f"  base_cd 범위 {min(r['base_cd'] for r in survivors):.4f} ~ "
      f"{max(r['base_cd'] for r in survivors):.4f}")
