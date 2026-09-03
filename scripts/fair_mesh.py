"""Tessellate a STEP, stitch the tessellation seams, fill the small gaps smoothly.

    fair_mesh.py --in CAS-A.stp --heal-report heal_v17.json --out body.stl

The heal report tells this stage which openings the B-rep stage deliberately left
alone (overlays, undecided intent) so they stay open here too. Everything else
under the sealing size is closed with a patch that continues the neighbouring
surface, and the report shows the seam angle before and after fairing so the
"smoothly" is a number rather than a claim.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from aox_g3 import cad, fair, topology  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="src", type=Path, required=True)
ap.add_argument("--out", type=Path)
ap.add_argument("--report", type=Path)
ap.add_argument("--heal-report", type=Path,
                help="heal_step.py JSON; its left_open list is honoured")
ap.add_argument("--seal-below", type=float, default=900.0)
ap.add_argument("--curvature-angle", type=float, default=15.0)
ap.add_argument("--stitch", type=float, help="vertex merge distance")
ap.add_argument("--no-sew", action="store_true")
args = ap.parse_args()

shape, cad_report = cad.read_step(args.src)
if shape is None:
    raise SystemExit(f"읽기 실패: {cad_report.warnings}")
cad.diagnose(shape, cad_report)
if not args.no_sew:
    shape, cad_report = cad.sew_progressive(shape, cad_report)

t0 = time.time()
mesh = topology.tessellate_by_curvature(shape, args.curvature_angle)
print(f"삼각화 {time.time() - t0:.1f}s   삼각형 {len(mesh.faces):,}  "
      f"정점 {len(mesh.vertices):,}")

held = []
if args.heal_report and args.heal_report.exists():
    heal = json.loads(args.heal_report.read_text())["heal"]
    for b in heal.get("left_open", []):
        held.append((*b["centre"], b["size"]))
    print(f"B-rep 단계가 남긴 개구부 {len(held)}개는 그대로 둡니다")

t0 = time.time()
out, report = fair.fill_holes(mesh, sealing_size=args.seal_below, held=held,
                              stitch_tolerance=args.stitch)
print(f"\n메쉬 봉합 {time.time() - t0:.1f}s   정점 병합 거리 {report.stitch_tolerance:.2f}")
print(f"  경계 고리 {report.loops_before_stitch} → 병합 후 {report.loops_after_stitch}"
      f"  (단순 고리 아님 {report.loops_skipped_shape})")
print(f"  검토 {report.loops_considered}  메움 {report.loops_filled}  "
      f"봉합크기 초과 {report.loops_skipped_large}  B-rep 보류 {report.loops_skipped_held}")
print(f"  삼각형 추가 {report.triangles_added:,}   수밀 {out.is_watertight}")

filled = [h for h in report.holes if h.filled]
if filled:
    flat = np.array([h.seam_angle_flat_max for h in filled])
    faired = np.array([h.seam_angle_fair_max for h in filled])
    print(f"\n이음매 이면각 (최대, 도)  평평한 패치 → 페어링 패치")
    print(f"  중앙값 {np.median(flat):6.1f} → {np.median(faired):6.1f}")
    print(f"  최대   {flat.max():6.1f} → {faired.max():6.1f}")
    print(f"  {'크기':>8} {'정점':>5} {'삼각형':>6} {'평평 최대':>9} {'페어 최대':>9}")
    for h in sorted(filled, key=lambda x: -x.size)[:14]:
        print(f"  {h.size:8.1f} {h.vertices:5d} {h.triangles:6d} "
              f"{h.seam_angle_flat_max:9.1f} {h.seam_angle_fair_max:9.1f}")

if args.out:
    out.export(args.out)
    print(f"\n저장: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
if args.report:
    args.report.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    print(f"보고서: {args.report}")
