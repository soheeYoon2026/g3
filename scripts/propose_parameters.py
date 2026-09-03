"""Measure a STEP and propose the pipeline's parameters, with reasons.

    propose_parameters.py --in CAS-A.stp --out params.json

Prints the proposal and the reasoning, writes it as JSON that
prepare_geometry.py --params can consume. Nothing is changed; this only looks.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aox_g3 import autotune, cad  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="src", type=Path, required=True)
ap.add_argument("--out", type=Path, help="write the proposal as JSON")
ap.add_argument("--no-sweep", action="store_true",
                help="skip the sewing sweep (about two minutes on a car)")
args = ap.parse_args()

t0 = time.time()
shape, report = cad.read_step(args.src)
if shape is None:
    raise SystemExit(f"읽기 실패: {report.warnings}")
cad.diagnose(shape, report)
print(f"읽기 {time.time() - t0:.1f}s   면 {report.faces:,}")

t0 = time.time()
proposal, _ = autotune.propose(shape, report, do_sweep=not args.no_sweep)
print(f"측정 {time.time() - t0:.1f}s\n")

print("== 제안 ==")
print(f"  단위          {proposal.units}")
print(f"  절반 모델     {'예' if proposal.half_model else '아니오'}"
      + (f"  (대칭면 y={proposal.symmetry_plane_y:.0f})" if proposal.half_model else ""))
print(f"  꿰맴 사다리   {proposal.sew_stages}")
print(f"  봉합 크기     {proposal.seal_below}")
print(f"  close-near    {proposal.close_near or '없음'}")
print(f"  정점 병합     {proposal.stitch_tolerance}")
print("\n== 근거 ==")
for line in proposal.rationale:
    print(f"  - {line}")
sweeps = proposal.sweeps.get("sewing") or {}
for label, key in (("단일 통과", "single"), ("누적 (시작점부터)", "cumulative")):
    rows = sweeps.get(key) if isinstance(sweeps, dict) else None
    if not rows:
        continue
    print(f"\n== 꿰맴 쓸기 · {label} ==")
    print(f"  {'허용오차':>10} {'셸':>6} {'자유모서리':>10} {'무효면':>7}")
    for r in rows:
        if "error" in r:
            print(f"  {r['tolerance']:10.4g}  실패 {r['error']}")
        else:
            print(f"  {r['tolerance']:10.4g} {r['shells']:6,} {r['free_edges']:10,} "
                  f"{r['invalid_faces']:7,}")
if proposal.questions:
    print("\n== 사람이 답해야 할 것 ==")
    for q in proposal.questions:
        print(f"  ? {q}")

if args.out:
    args.out.write_text(json.dumps(proposal.as_dict(), ensure_ascii=False, indent=2))
    print(f"\n저장: {args.out}")
