"""Run steps 7, 8 and 9 on a STEP file and report what each found.

    check_topology.py --in CAS-A.stp --report t.json

Step 7 is off by default because it costs about eight minutes on a car body; pass
--intersections to include it. Steps 8 and 9 take seconds.

Step 8 will refuse to classify a model whose surface leaks, and that refusal is the
point. Flooding from outside only distinguishes skin from baffle from buried if the
flood stays outside; through a 4.85 m underbody opening it fills the cabin, and
then every panel with the cabin behind it looks like a zero-thickness sheet. Better
to say the model is not ready for this question than to answer it wrongly.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aox_g3 import cad, topology  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="src", type=Path, required=True)
ap.add_argument("--report", type=Path)
ap.add_argument("--intersections", action="store_true",
                help="include step 7 — slow")
ap.add_argument("--fix-orientation", action="store_true")
ap.add_argument("--min-passage", type=float,
                help="step 8 voxel pitch (default: diagonal/300)")
ap.add_argument("--curvature-angle", type=float, default=15.0)
ap.add_argument("--max-edge", type=float)
ap.add_argument("--no-sew", action="store_true")
args = ap.parse_args()

shape, report = cad.read_step(args.src)
if shape is None:
    raise SystemExit(f"읽기 실패: {report.warnings}")
cad.diagnose(shape, report)
if not args.no_sew:
    shape, report = cad.sew_progressive(shape, report)
    after = cad.CadReport()
    cad.diagnose(shape, after)
    print(f"꿰맴: 셸 {report.shells:,} → {after.shells:,}  "
          f"자유모서리 {report.free_edges:,} → {after.free_edges:,}")

payload = {}

print("\n== 9. 법선 방향 ==")
orientation = topology.check_orientation(shape, fix=args.fix_orientation)
print(f"  면 {orientation.faces:,}  FORWARD {orientation.forward:,}  "
      f"REVERSED {orientation.reversed_:,}")
print(f"  이웃과 방향이 어긋난 셸 {orientation.shells_with_bad_edges}"
      + (f", {orientation.shells_fixed}개 수정" if args.fix_orientation else ""))
for w in orientation.warnings:
    print(f"  경고: {w}")
payload["orientation"] = orientation.as_dict()

print("\n== 9. 곡률 기반 표면 메쉬 ==")
print(f"  {'곡률각':>7} {'선형편차':>9} {'최대에지':>9} {'삼각형':>10} "
      f"{'표면적 m2':>10} {'초':>6}")
diag = float(sum((b - a) ** 2 for a, b in
                 zip(report.bbox[:3], report.bbox[3:]))) ** 0.5
for angle, sag, edge in ((30.0, diag / 2000, None),
                         (args.curvature_angle, diag / 2000, None),
                         (5.0, diag / 2000, None),
                         (args.curvature_angle, diag / 2000, diag / 100)):
    edge = args.max_edge or edge
    t0 = time.time()
    mesh = topology.tessellate_by_curvature(shape, angle, sag, edge)
    print(f"  {angle:7.1f} {sag:9.1f} "
          f"{(f'{edge:.1f}' if edge else '—'):>9} {len(mesh.faces):10,} "
          f"{mesh.area / 1e6:10.2f} {time.time() - t0:6.1f}")

print("\n== 8. 숨은 면과 baffle ==")
t0 = time.time()
classification, mesh, owner, faces = topology.classify_faces(
    shape, min_passage=args.min_passage)
print(f"  복셀 {classification.grid_shape} 피치 {classification.pitch:.1f}  "
      f"{time.time() - t0:.1f}s")
print(f"  갇힌 부피 {100 * classification.enclosed_fraction:.1f}% of box   "
      f"신뢰 가능 {'예' if classification.trustworthy else '아니오'}")
if classification.trustworthy:
    print(f"    {'외피':>7} {classification.skin_faces:5,}개  "
          f"{classification.skin_area / 1e6:7.2f} m2")
    print(f"    {'baffle':>7} {classification.baffle_faces:5,}개  "
          f"{classification.baffle_area / 1e6:7.2f} m2")
    print(f"    {'숨음':>7} {classification.hidden_faces:5,}개  "
          f"{classification.hidden_area / 1e6:7.2f} m2")
else:
    print(f"    분류를 보고하지 않습니다 (아래 이유)")
for w in classification.warnings:
    print(f"  경고: {w}")
payload["classification"] = classification.as_dict()

if args.intersections:
    print("\n== 7. 자기교차 ==")
    print("  검사 중… 차체 한 대에 약 8분 걸립니다", flush=True)
    intersections = topology.check_intersections(shape)
    print(f"  자기교차 {intersections.self_intersections:,}건  "
          f"기타 결함 {intersections.other_faults:,}건  "
          f"관련 형상 {intersections.faulty_faces:,}개  "
          f"{intersections.seconds:.0f}s")
    payload["intersections"] = intersections.as_dict()
else:
    print("\n== 7. 자기교차 ==\n  건너뜀 (--intersections 로 활성화)")

if args.report:
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n보고서: {args.report}")
