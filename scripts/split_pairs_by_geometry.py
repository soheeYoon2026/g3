"""Split the deformation-pair dataset by geometry, not by pair.

Every pair in a job is a step of the same optimisation on the same car, so
splitting pairs at random would put near-identical shapes on both sides and the
held-out score would measure memorisation. This assigns whole jobs, and balances
the two sides on how many pairs each carries and on the up/down mix.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--pairs", type=Path, required=True, help="fetched pairs.json")
ap.add_argument("--benchmark", type=Path, required=True, help="pair-benchmark.json with job_uid")
ap.add_argument("--out-dir", type=Path, required=True)
ap.add_argument("--test-fraction", type=float, default=0.35)
ap.add_argument("--seed", type=int, default=20260827)
args = ap.parse_args()

pairs = json.loads(args.pairs.read_text())["pairs"]
# the fetched manifest keeps the job in its note as "<job8>:<design>"
by_job = defaultdict(list)
for pair in pairs:
    by_job[pair["note"].split(":")[0]].append(pair)

jobs = sorted(by_job)
rng = np.random.default_rng(args.seed)
rng.shuffle(jobs)

target = args.test_fraction * len(pairs)
test_jobs, count = [], 0
for job in jobs:
    if count >= target:
        break
    test_jobs.append(job)
    count += len(by_job[job])
test_jobs = set(test_jobs)
train_jobs = [j for j in jobs if j not in test_jobs]


def describe(name, job_list):
    rows = [p for j in job_list for p in by_job[j]]
    delta = np.array([p["true_delta_cd"] for p in rows])
    up, down = int((delta > 0).sum()), int((delta < 0).sum())
    runs = sorted({r for p in rows for r in (p["baseline"], p["variant"])})
    print(f"{name}: 형상 {len(job_list)}종  쌍 {len(rows)}개  런 {len(runs)}개  "
          f"증가 {up}/감소 {down} (다수 {100*max(up,down)/max(len(rows),1):.0f}%)")
    return rows, runs


train_rows, train_runs = describe("train", train_jobs)
test_rows, test_runs = describe("test ", sorted(test_jobs))
overlap = set(train_runs) & set(test_runs)
print(f"런 중복: {len(overlap)}개" + (" — 누수!" if overlap else " (없음)"))

args.out_dir.mkdir(parents=True, exist_ok=True)
(args.out_dir / "split-pairs-train.json").write_text(json.dumps({
    "train_cases": [{"run": r, "group_id": None} for r in train_runs],
    "test_cases": [{"run": r, "group_id": None} for r in test_runs],
}, indent=1) + "\n")
(args.out_dir / "pairs-train.json").write_text(json.dumps({"pairs": train_rows}, indent=1) + "\n")
(args.out_dir / "pairs-test.json").write_text(json.dumps({"pairs": test_rows}, indent=1) + "\n")
(args.out_dir / "split-pairs-test.json").write_text(json.dumps({
    "test_cases": [{"run": r, "group_id": None} for r in test_runs],
}, indent=1) + "\n")
print("wrote", args.out_dir)
