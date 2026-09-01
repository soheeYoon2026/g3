"""Summarise the LES benchmark campaign: did the deformations increase drag, and
is the change resolvable?

The design target was drag-INCREASING deformations, which the existing 400-pair
pool could not supply. Each variant is compared against its own car's baseline on
the same frozen grid, so ΔCd is meaningful even though absolute LES Cd carries a
known +86% calibration offset in this implementation.
"""

import argparse
import json
from pathlib import Path

VARIANTS = ("blunt_tail", "wide_rear", "raise_roof")


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dir", type=Path, required=True)
ap.add_argument("--cars", nargs="+", default=["carA", "carB"])
args = ap.parse_args()

for car in args.cars:
    base = load(args.dir / f"les_{car}_base_result.json")
    if not base or base.get("skipped"):
        print(f"\n=== {car} === 기준 결과 없음\n")
        continue
    base_cd = base.get("cd")
    print(f"\n=== {car} ===")
    print(f"  기준 Cd {base_cd}  cd_std {base.get('cd_std')}  "
          f"셀 {base.get('cells_m')}M  {base.get('elapsed_minutes')}분")
    print(f"  {'변형':>12s} {'Cd':>8s} {'cd_std':>8s} {'ΔCd':>9s} {'S/N':>6s} {'판정':>6s}")
    print("  " + "-" * 56)
    for name in VARIANTS:
        row = load(args.dir / f"les_{car}_{name}_result.json")
        if not row or row.get("skipped") or row.get("cd") is None:
            print(f"  {name:>12s}   결과 없음")
            continue
        delta = row["cd"] - base_cd
        # combine both runs' scatter; the grid is frozen so this is the honest bar
        noise = (base.get("cd_std", 0) ** 2 + row.get("cd_std", 0) ** 2) ** 0.5
        sn = abs(delta) / noise if noise else float("inf")
        verdict = "증가" if delta > 0 and sn >= 2 else "감소" if delta < 0 and sn >= 2 else "판정불가"
        print(f"  {name:>12s} {row['cd']:8.4f} {row.get('cd_std', 0):8.4f} "
              f"{delta:+9.4f} {sn:6.1f} {verdict:>6s}")
