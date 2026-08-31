"""Check every case landed inside the encoder grid, not just a sample.

Alignment anchors on the area-weighted centroid and the ground plane, which is
right for a car sitting on a road. A case that is scaled differently, or oriented
with a different axis up, can still fall outside afterwards - and training on
those is what produced the near-constant encodings in the first place.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv

# the fixed DrivAerML box the encoder normalizes against
BOX_MIN = np.array([-1.5, -1.4, -0.32])
BOX_MAX = np.array([5.0, 1.4, 1.4])

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--report", type=Path)
args = ap.parse_args()

rows = []
run_dirs = sorted((p for p in args.root.glob("run_*") if p.is_dir()),
                  key=lambda p: int(p.name.split("_")[1]))
for run_dir in run_dirs:
    tag = run_dir.name.split("_", 1)[1]
    try:
        points = np.asarray(pv.read(run_dir / f"boundary_{tag}.vtp").points, dtype=np.float64)
    except Exception as exc:
        rows.append({"run": tag, "inside": 0.0, "error": type(exc).__name__})
        continue
    norm = 2.0 * (points - BOX_MIN) / (BOX_MAX - BOX_MIN) - 1
    inside = float(np.all((norm >= -1) & (norm <= 1), axis=1).mean())
    extent = points.max(axis=0) - points.min(axis=0)
    rows.append({"run": tag, "inside": inside, "extent": extent.round(3).tolist()})

inside = np.array([r["inside"] for r in rows])
print(f"런 {len(rows)}개")
print(f"격자 안 정점 비율: 평균 {100*inside.mean():.1f}%  중앙값 {100*np.median(inside):.1f}%")
for bound in (0.0, 0.5, 0.9):
    print(f"  {int(100*bound)}% 이하인 런: {int((inside <= bound).sum())}개")

bad = sorted((r for r in rows if r["inside"] < 0.5), key=lambda r: r["inside"])
if bad:
    print(f"\n절반도 못 들어간 런 {len(bad)}개 (학습에서 빼야 할 후보)")
    for r in bad[:12]:
        print(f"  run_{r['run']}: {100*r['inside']:5.1f}%  치수 {r.get('extent')}")

if args.report:
    args.report.write_text(json.dumps(
        {r["run"]: r["inside"] for r in rows}, indent=1) + "\n")
    print("\nwrote", args.report)
