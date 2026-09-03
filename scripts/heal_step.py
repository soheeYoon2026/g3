"""STEP in, STEP out: sew, close the holes that are defects, report the ones that are not.

This is the pipeline a CAD engineer can actually use, because what comes back is
still CAD. The mesh cascade in aox_g3.seal stays where it belongs - for STL and OBJ
input, where there is no topology to preserve in the first place.

    heal_step.py --in CAS-A.stp --out CAS-A-healed.stp --report r.json

The one parameter that needs a human is --seal-below. Everything under it is closed
without asking; everything over it is listed with its size and position, because at
that scale the tool cannot tell a missing underbody from a cooling inlet and
guessing wrong changes the flow answer rather than fixing the model.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aox_g3 import brep, cad  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="src", type=Path, required=True)
ap.add_argument("--out", dest="dst", type=Path, help="healed STEP")
ap.add_argument("--report", type=Path, help="write the full report as JSON")
ap.add_argument("--seal-below", type=float,
                help="close holes smaller than this (default: diagonal/100)")
ap.add_argument("--fill-all", action="store_true",
                help="close every hole regardless of size — will seal real inlets")
ap.add_argument("--close-near", action="append", default=[],
                metavar="X,Y,Z,R",
                help="deliberate simplification: close boundaries whose centre is "
                     "within R of this point even when surface lies behind them "
                     "(e.g. a closed wheel rim); repeatable")
ap.add_argument("--sew-tolerance", type=float)
ap.add_argument("--pcurves", action="store_true",
                help="write parametric curves into the STEP (much larger file)")
ap.add_argument("--no-sew", action="store_true")
ap.add_argument("--list-only", action="store_true",
                help="report the boundaries and stop, writing nothing")
args = ap.parse_args()

if not args.src.exists():
    raise SystemExit(f"입력이 없습니다: {args.src}")

t0 = time.time()
shape, report = cad.read_step(args.src)
if shape is None:
    raise SystemExit(f"STEP 읽기 실패: {report.warnings}")
cad.diagnose(shape, report)
print(f"읽기 {time.time() - t0:5.1f}s   면 {report.faces:,}  셸 {report.shells:,}  "
      f"자유모서리 {report.free_edges:,}  단위 {report.units_hint}")
if report.invalid_faces:
    print(f"  OCC 가 무효로 보는 면 {report.invalid_faces:,}개")

for w in report.warnings:
    print(f"  경고: {w}")

if not args.no_sew:
    t0 = time.time()
    shape, report = (cad.sew(shape, report, args.sew_tolerance)
                     if args.sew_tolerance else cad.sew_progressive(shape, report))
    after = cad.CadReport()
    cad.diagnose(shape, after)
    stages = " → ".join(f"{t:g}" for t in report.sew_stages) or \
        f"{report.sew_tolerance:g}"
    print(f"꿰맴 {time.time() - t0:5.1f}s   허용오차 {stages}  "
          f"셸 {report.shells:,} → {after.shells:,}  "
          f"자유모서리 {report.free_edges:,} → {after.free_edges:,}  "
          f"무효면 {report.invalid_faces:,} → {after.invalid_faces:,}")

if args.list_only:
    t0 = time.time()
    holes, dangling = brep.free_boundaries(shape)
    print(f"\n자유경계 분석 {time.time() - t0:.1f}s   "
          f"구멍 {len(holes)}개, 고리 못 이룬 경계 {len(dangling)}개")
    print(f"{'크기':>10} {'둘레':>10} {'모서리':>7} {'평면':>5}   중심")
    for boundary, _ in holes:
        c = boundary.centre
        print(f"{boundary.size:10.1f} {boundary.length:10.1f} {boundary.edges:7d} "
              f"{'예' if boundary.planar else '  ':>5}   "
              f"({c[0]:8.0f},{c[1]:7.0f},{c[2]:7.0f})")
    if args.report:
        args.report.write_text(json.dumps(
            {"holes": [b.as_dict() for b, _ in holes],
             "dangling": [b.as_dict() for b, _ in dangling]},
            ensure_ascii=False, indent=2))
        print(f"\n보고서: {args.report}")
    raise SystemExit(0)

close_near = []
for spec in args.close_near:
    parts = [float(v) for v in spec.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise SystemExit(f"--close-near 는 X,Y,Z,R 이어야 합니다: {spec!r}")
    close_near.append(tuple(parts))

t0 = time.time()
shape, heal_report = brep.heal(shape, sealing_size=args.seal_below,
                               fill_all=args.fill_all,
                               sew_tolerance=args.sew_tolerance,
                               close_near=close_near)
print(f"\n봉합 {time.time() - t0:5.1f}s   봉합크기 {heal_report.sealing_size:.1f}")
print(f"  구멍 {heal_report.boundaries_found}개 중 "
      f"{heal_report.boundaries_filled}개 메움, {heal_report.boundaries_left}개 남김")
print(f"  면 {heal_report.faces_before:,} → {heal_report.faces_after:,}   "
      f"셸 {heal_report.shells_after}  고체 {heal_report.solids_after}")
print(f"  닫힘 {'예' if heal_report.closed else '아니오'}   "
      f"OCC 유효 {'예' if heal_report.valid else '아니오'}")
print(f"  표면적 {heal_report.area_before / 1e6:.2f} → {heal_report.area / 1e6:.2f} m2  "
      f"(패치 {heal_report.patch_area / 1e6:+.2f})", end="")
print(f"   체적 {heal_report.volume / 1e9:.3f} m3" if heal_report.closed else "")

if heal_report.left_open:
    print(f"\n엔지니어 판단이 필요한 열린 경계 {len(heal_report.left_open)}개 "
          f"(봉합크기 {heal_report.sealing_size:.0f} 초과):")
    print(f"{'크기':>10} {'둘레':>10}   중심")
    for boundary in heal_report.left_open[:20]:
        c = boundary.centre
        print(f"{boundary.size:10.1f} {boundary.length:10.1f}   "
              f"({c[0]:8.0f},{c[1]:7.0f},{c[2]:7.0f})   {boundary.note}")
    if len(heal_report.left_open) > 20:
        print(f"   … 그리고 {len(heal_report.left_open) - 20}개 더 (보고서 참조)")

for w in heal_report.warnings:
    print(f"  경고: {w}")

if args.dst:
    t0 = time.time()
    ok = brep.write_step(shape, args.dst, heal_report,
                         units="MM" if report.units_hint == "mm" else "M",
                         pcurves=args.pcurves)
    if ok:
        size = args.dst.stat().st_size / 1e6
        print(f"\nSTEP 저장 {time.time() - t0:.1f}s   {args.dst}  ({size:.1f} MB)")
    else:
        print("\nSTEP 저장 실패")

if args.report:
    payload = {"cad": report.as_dict(), "heal": heal_report.as_dict()}
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"보고서: {args.report}")

# A model still carrying holes is not watertight, and the caller should know from
# the exit code rather than by parsing the text
raise SystemExit(0 if heal_report.closed else 1)
