"""Sweep every case for a declared reference area that disagrees with its geometry.

Three of the fifteen standing-gate cases declare a `ref_area` 3.0-3.2x their own
frontal area. Since Cd = force/(q * ref_area), the label is wrong by exactly that
factor -- undoing it took one case from 65% error to 4%. Those three were found by
chance while chasing something else, so this checks the whole pool.

Frontal area is the bounding box's width x height, the same convention the upload
gate and the LES solver use. Cases are reported, never modified: a mismatch can mean
the label is wrong, or that the case legitimately uses a different reference (a
half-model, a wing planform), and that call needs a human.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv


def frontal_area(mesh):
    """Width x height from the bounding box, axis-order independent."""
    span = np.sort(np.diff(np.asarray(mesh.bounds).reshape(3, 2), axis=1).ravel())[::-1]
    return float(span[1] * span[2]), span


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--roots", type=Path, nargs="+", required=True)
ap.add_argument("--tolerance", type=float, default=0.15,
                help="relative disagreement treated as acceptable")
ap.add_argument("--out", type=Path)
args = ap.parse_args()

rows = []
for root in args.roots:
    run_dirs = sorted((p for p in root.glob("run_*") if p.is_dir()),
                      key=lambda p: int(p.name.split("_")[1]))
    for run_dir in run_dirs:
        tag = run_dir.name.split("_", 1)[1]
        cfg = run_dir / f"conditions_{tag}.json"
        vtp = run_dir / f"boundary_{tag}.vtp"
        if not (cfg.exists() and vtp.exists()):
            continue
        try:
            conditions = json.loads(cfg.read_text())
            declared = float(conditions.get("ref_area") or 0.0)
            measured, span = frontal_area(pv.read(vtp).extract_surface())
        except Exception:
            continue
        if declared <= 0 or measured <= 0:
            continue
        rows.append({
            "dataset": root.name, "run": tag,
            "declared": declared, "measured": measured,
            "ratio": declared / measured,
            "extents": [round(float(v), 3) for v in span],
            "su2_cd": conditions.get("su2_cd"),
        })

if not rows:
    raise SystemExit("검사할 케이스를 찾지 못했다")

ratios = np.array([r["ratio"] for r in rows])
print(f"검사 {len(rows)}건")
print(f"선언/실측 비: 중앙값 {np.median(ratios):.3f}  "
      f"사분위 {np.percentile(ratios, 25):.3f}/{np.percentile(ratios, 75):.3f}")

bad = [r for r in rows if abs(r["ratio"] - 1) > args.tolerance]
print(f"\n±{args.tolerance:.0%} 밖: {len(bad)}건 ({100*len(bad)/len(rows):.0f}%)")
for lo, hi, name in ((0, 0.6, "선언이 실측의 60% 미만"),
                     (1.4, 2.5, "1.4~2.5배 과대"),
                     (2.5, 3.5, "2.5~3.5배 과대 (3배 오류 의심)"),
                     (3.5, 1e9, "3.5배 초과")):
    group = [r for r in rows if lo <= r["ratio"] < hi]
    if group:
        print(f"  {name}: {len(group)}건")

print(f"\n{'데이터셋':>24s} {'run':>6s} {'선언':>8s} {'실측':>8s} {'비':>6s} {'치수':>22s}")
print("-" * 80)
for r in sorted(bad, key=lambda r: -abs(r["ratio"] - 1))[:20]:
    print(f"{r['dataset'][:24]:>24s} {r['run']:>6s} {r['declared']:8.3f} "
          f"{r['measured']:8.3f} {r['ratio']:6.2f} {str(r['extents']):>22s}")

if args.out:
    args.out.write_text(json.dumps({"checked": len(rows), "mismatched": len(bad),
                                    "tolerance": args.tolerance, "cases": bad}, indent=1) + "\n")
    print(f"\nwrote {args.out}")
