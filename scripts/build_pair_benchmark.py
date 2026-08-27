"""Select a usable ΔCd pair benchmark from the stored G2 deformation pairs.

Three filters, each for a stated reason:
  * |ΔCd| above the solver's noise floor -- a label below it carries no direction.
  * both coefficients physically plausible for a road car -- the pool contains
    diverged jobs with negative or absurd Cd.
  * a cap per geometry -- one heavily optimised shape should not dominate the
    score with dozens of near-identical steps.

Writes the manifest and the list of surface meshes that have to be fetched, which
is fewer than two per pair because consecutive design steps share a mesh.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--manifest", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--noise-floor", type=float, default=0.0015)
ap.add_argument("--cd-min", type=float, default=0.10)
ap.add_argument("--cd-max", type=float, default=1.00)
ap.add_argument("--max-per-job", type=int, default=4)
args = ap.parse_args()

payload = json.loads(args.manifest.read_text())
pairs = payload["pairs"]
print(f"원본 쌍 {len(pairs)}개")

kept, reasons = [], {"noise": 0, "unphysical": 0, "job_cap": 0}
per_job = {}
for pair in sorted(pairs, key=lambda p: -abs(p["delta_cd"])):
    base, target, delta = pair["base_cd"], pair["target_cd"], pair["delta_cd"]
    if not all(np.isfinite([base, target, delta])):
        reasons["unphysical"] += 1
        continue
    if abs(delta) < args.noise_floor:
        reasons["noise"] += 1
        continue
    if not (args.cd_min <= base <= args.cd_max and args.cd_min <= target <= args.cd_max):
        reasons["unphysical"] += 1
        continue
    job = pair["job_uid"]
    if per_job.get(job, 0) >= args.max_per_job:
        reasons["job_cap"] += 1
        continue
    per_job[job] = per_job.get(job, 0) + 1
    kept.append(pair)

print(f"  노이즈 미달 제외 {reasons['noise']}  비물리 제외 {reasons['unphysical']}  "
      f"형상당 상한 초과 제외 {reasons['job_cap']}")
print(f"채택 {len(kept)}쌍 / 형상 {len(per_job)}종")

delta = np.array([p["delta_cd"] for p in kept])
up, down = int((delta > 0).sum()), int((delta < 0).sum())
print(f"  방향 증가 {up} / 감소 {down}  "
      f"— 한쪽으로만 답하면 {max(up,down)}/{len(kept)} = {100*max(up,down)/len(kept):.0f}%")
print(f"  |ΔCd| 중앙값 {np.median(np.abs(delta)):.5f}  최대 {np.abs(delta).max():.5f}")

meshes = set()
for pair in kept:
    for design in (pair["base_design"], pair["target_design"]):
        meshes.add((pair["job_uid"], design, pair["output_prefix"]))
print(f"내려받아야 할 표면 메시 {len(meshes)}개 (쌍당 2개 미만 — 연속 단계가 공유)")

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps({
    "name": "g2-deformation-pair-benchmark-v1",
    "filters": {"noise_floor": args.noise_floor, "cd_range": [args.cd_min, args.cd_max],
                "max_per_job": args.max_per_job},
    "counts": {"pairs": len(kept), "geometries": len(per_job),
               "up": up, "down": down, "meshes": len(meshes)},
    "pairs": kept,
    "meshes": [{"job_uid": j, "design": d, "output_prefix": p} for j, d, p in sorted(meshes)],
}, indent=1) + "\n")
print("wrote", args.out)
