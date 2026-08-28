"""Build a direction-balanced benchmark from the pairs already on hand.

The current car benchmark is 65% one direction, so a constant answer already
scores 65 and only 22 pairs carry the minority signal. Matching the drag-increasing
pairs with an equal number of drag-decreasing ones gives a 50% baseline, where
every pair discriminates.

This cannot be used to make a clean claim about the serving model - it trained on
some of these geometries - but it is the right benchmark for comparing candidate
models that are geometry-separated by cross-validation.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--manifest", type=Path, required=True)
ap.add_argument("--shape-class", type=Path, required=True)
ap.add_argument("--run-index", type=Path, required=True)
ap.add_argument("--noise-floor", type=float, default=0.0015)
ap.add_argument("--cd-min", type=float, default=0.10)
ap.add_argument("--cd-max", type=float, default=1.00)
ap.add_argument("--max-per-job", type=int, default=3)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

verdicts = json.loads(args.shape_class.read_text())
index = json.loads(args.run_index.read_text())
run_of = {(m["job_uid"], m["design"]): int(run) for run, m in index.items()}
class_of = {(m["job_uid"], m["design"]): verdicts.get(str(run), {}).get("verdict")
            for run, m in index.items()}

candidates = []
for pair in json.loads(args.manifest.read_text())["pairs"]:
    base_key = (pair["job_uid"], pair["base_design"])
    target_key = (pair["job_uid"], pair["target_design"])
    if base_key not in run_of or target_key not in run_of:
        continue  # mesh was never fetched
    if class_of.get(base_key) != "full_car" or class_of.get(target_key) != "full_car":
        continue
    base, target, delta = pair["base_cd"], pair["target_cd"], pair["delta_cd"]
    if not all(np.isfinite([base, target, delta])):
        continue
    if not (args.cd_min <= base <= args.cd_max and args.cd_min <= target <= args.cd_max):
        continue
    if abs(delta) < args.noise_floor:
        continue
    candidates.append({
        "baseline": run_of[base_key], "variant": run_of[target_key],
        "true_delta_cd": delta, "job_uid": pair["job_uid"],
        "note": f"{pair['job_uid'][:8]}:{pair['target_design']}",
    })

up = sorted([c for c in candidates if c["true_delta_cd"] > 0],
            key=lambda c: -abs(c["true_delta_cd"]))
down = sorted([c for c in candidates if c["true_delta_cd"] < 0],
              key=lambda c: -abs(c["true_delta_cd"]))
print(f"차량형 후보 {len(candidates)}쌍 — 증가 {len(up)} / 감소 {len(down)}")


def take(rows, limit):
    picked, per_job = [], Counter()
    for row in rows:
        if per_job[row["job_uid"]] >= args.max_per_job:
            continue
        per_job[row["job_uid"]] += 1
        picked.append(row)
        if len(picked) >= limit:
            break
    return picked


chosen_up = take(up, len(up))
chosen_down = take(down, len(chosen_up))
pairs = chosen_up + chosen_down
delta = np.array([p["true_delta_cd"] for p in pairs])
jobs = {p["job_uid"] for p in pairs}
n = len(pairs)
majority = max(int((delta > 0).sum()), int((delta < 0).sum()))
print(f"\n균형 벤치마크 {n}쌍 / 형상 {len(jobs)}종")
print(f"  증가 {int((delta>0).sum())} / 감소 {int((delta<0).sum())} "
      f"-> 다수기준 {majority}/{n} = {100*majority/n:.0f}%")
print(f"  |ΔCd| 중앙값 {np.median(np.abs(delta)):.5f}  "
      f"범위 {np.abs(delta).min():.5f}~{np.abs(delta).max():.5f}")

runs = sorted({r for p in pairs for r in (p["baseline"], p["variant"])})
args.out.write_text(json.dumps({"pairs": pairs}, indent=1) + "\n")
split = args.out.with_name(args.out.stem + "-split.json")
split.write_text(json.dumps({"test_cases": [{"run": r, "group_id": None} for r in runs]}) + "\n")
print(f"  런 {len(runs)}개")
print("wrote", args.out, "및", split)
