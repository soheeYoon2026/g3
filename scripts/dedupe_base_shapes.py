"""How many genuinely distinct cars are in the candidate list?

Several candidates share dimensions and drag to four decimals across different job
ids, which suggests the same vehicle uploaded repeatedly. A benchmark built from
copies of one car would measure less than its case count implies, so this hashes
the actual geometry and groups by it.
"""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--candidates", type=Path, required=True)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--out", type=Path)
args = ap.parse_args()

candidates = json.loads(args.candidates.read_text())["candidates"]
groups = {}
for row in candidates:
    run = row["run"]
    mesh = pv.read(args.root / f"run_{run}" / f"boundary_{run}.vtp").extract_surface()
    points = np.asarray(mesh.points, dtype=np.float64)
    # translation-invariant: hash the shape, not where it sits
    centred = np.round(points - points.mean(axis=0), 4)
    digest = hashlib.sha256(np.sort(centred, axis=0).tobytes()).hexdigest()[:16]
    groups.setdefault(digest, []).append(row)

print(f"후보 {len(candidates)}개 -> 서로 다른 형상 {len(groups)}개\n")
print(f"{'형상해시':>18s} {'중복':>5s} {'Cd':>8s} {'치수':>24s} {'대표 run':>9s}")
print("-" * 70)
unique = []
for digest, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    rep = min(rows, key=lambda r: r["run"])
    unique.append({**rep, "duplicates": len(rows),
                   "duplicate_runs": sorted(r["run"] for r in rows)})
    print(f"{digest:>18s} {len(rows):>5d} {rep['cd']:8.4f} "
          f"{str(rep['extents']):>24s} {rep['run']:>9d}")

print(f"\nLES 캠페인에 쓸 수 있는 서로 다른 차: {len(unique)}대")
if args.out:
    args.out.write_text(json.dumps({"shapes": unique}, indent=1) + "\n")
    print("wrote", args.out)
