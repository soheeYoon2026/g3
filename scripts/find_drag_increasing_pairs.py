"""Find the pairs the benchmark is actually short of.

The car subset is discriminating only because eight of its twenty-two geometries
carry the drag-increasing deformations - and those eight are exactly what the
serving model trained on. So the gap is not pair count: it is drag-increasing
deformations on geometry the serving model never saw. This searches the full
400-pair pool for them.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--manifest", type=Path, required=True)
ap.add_argument("--incumbent-log", type=Path, required=True)
ap.add_argument("--reaudit-root", type=Path, required=True)
ap.add_argument("--shape-class", type=Path, help="run -> verdict, for already-fetched runs")
ap.add_argument("--run-index", type=Path, help="run -> job/design, for already-fetched runs")
ap.add_argument("--noise-floor", type=float, default=0.0015)
ap.add_argument("--cd-min", type=float, default=0.10)
ap.add_argument("--cd-max", type=float, default=1.00)
ap.add_argument("--out", type=Path)
args = ap.parse_args()

# which geometries the serving model has already seen
log = args.incumbent_log.read_text()
runs, order = set(), []
for run in re.findall(r"load_case\(.run_([0-9]+).\)", log):
    if run not in runs:
        runs.add(run)
        order.append(run)
seen_jobs = set()
for run in order:
    path = args.reaudit_root / f"run_{run}" / f"conditions_{run}.json"
    if not path.exists():
        continue
    match = re.search(r"([A-Za-z0-9]{22})_CFD\.cfg",
                      json.loads(path.read_text()).get("source_cfg") or "")
    if match:
        seen_jobs.add(match.group(1))
print(f"현행이 학습에서 본 잡: {len(seen_jobs)}개")

known_class = {}
if args.shape_class and args.run_index:
    verdicts = json.loads(args.shape_class.read_text())
    index = json.loads(args.run_index.read_text())
    for run, meta in index.items():
        verdict = verdicts.get(str(run), {}).get("verdict")
        if verdict:
            known_class[(meta["job_uid"], meta["design"])] = verdict
    print(f"이미 분류된 (잡, 설계): {len(known_class)}개")

pairs = json.loads(args.manifest.read_text())["pairs"]
print(f"\n전체 쌍 {len(pairs)}개에서 걸러 나가기")

stages = Counter()
kept = []
for pair in pairs:
    stages["전체"] += 1
    base, target, delta = pair["base_cd"], pair["target_cd"], pair["delta_cd"]
    if not all(np.isfinite([base, target, delta])):
        continue
    if not (args.cd_min <= base <= args.cd_max and args.cd_min <= target <= args.cd_max):
        continue
    stages["물리적으로 타당"] += 1
    if delta <= 0:
        continue
    stages["항력 증가"] += 1
    if abs(delta) < args.noise_floor:
        continue
    stages["노이즈 기준 통과"] += 1
    if pair["job_uid"] in seen_jobs:
        continue
    stages["현행 미학습 형상"] += 1
    verdict = known_class.get((pair["job_uid"], pair["base_design"]))
    pair = dict(pair, shape_class=verdict or "미분류")
    if verdict in (None, "full_car", "미분류"):
        stages["차량형 또는 미분류"] += 1
        kept.append(pair)

for name, count in stages.items():
    print(f"  {name}: {count}")

jobs = sorted({p["job_uid"] for p in kept})
classes = Counter(p["shape_class"] for p in kept)
print(f"\n조건을 만족하는 쌍 {len(kept)}개 / 형상 {len(jobs)}종")
print(f"  형상 분류: {dict(classes)}")
if kept:
    delta = np.array([p["delta_cd"] for p in kept])
    print(f"  ΔCd {delta.min():+.4f} ~ {delta.max():+.4f} (중앙값 {np.median(delta):+.4f})")
    print(f"\n{'job':>10s} {'설계':>24s} {'ΔCd':>9s} {'분류':>12s}")
    for p in sorted(kept, key=lambda r: -r["delta_cd"])[:15]:
        print(f"{p['job_uid'][:8]:>10s} "
              f"{p['base_design']+'->'+p['target_design']:>24s} "
              f"{p['delta_cd']:+9.4f} {p['shape_class']:>12s}")

if args.out and kept:
    args.out.write_text(json.dumps({
        "purpose": "drag-increasing pairs on geometry unseen by the serving model",
        "counts": {"pairs": len(kept), "geometries": len(jobs)},
        "pairs": kept,
    }, indent=1) + "\n")
    print("\nwrote", args.out)
