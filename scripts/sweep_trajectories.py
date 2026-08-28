"""Test the model the way the product is used: push a control point step by step.

Each RBF job is exactly that experiment already run - DSN_001, 002, 003 ... are
successive deformations of one car with measured Cd at every step. Cross-validated
predictions come from a model that never trained on that car, so the trajectory is
a clean sweep test.

Three questions, in order of how much they matter to a user:
  * does the model point at the same best design the optimizer found?
  * does it order the steps the same way?
  * does it call each step's direction right?
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path):
    rows = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("model") == "fine_tuned" and "case" in row:
            rows[row["case"]] = row
    return rows


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--cv-dirs", type=Path, nargs="+", required=True)
ap.add_argument("--run-index", type=Path, required=True)
ap.add_argument("--shape-class", type=Path)
ap.add_argument("--min-steps", type=int, default=4)
ap.add_argument("--min-span", type=float, default=0.0,
                help="drop trajectories whose whole Cd range is below this; the "
                     "labels cannot say which design is best when it is")
ap.add_argument("--min-step-delta", type=float, default=0.0,
                help="score only steps this large; smaller ones are label noise")
args = ap.parse_args()

index = json.loads(args.run_index.read_text())
verdicts = json.loads(args.shape_class.read_text()) if args.shape_class else {}

predictions = {}
for cv_dir in args.cv_dirs:
    for fold in sorted(cv_dir.glob("fold*")):
        path = fold / "eval-test.jsonl"
        if path.exists():
            predictions.update(load(path))

trajectories = defaultdict(list)
for run, meta in index.items():
    case = f"run_{run}"
    if case not in predictions:
        continue
    if verdicts and verdicts.get(str(run), {}).get("verdict") != "full_car":
        continue
    trajectories[meta["job_uid"]].append({
        "design": meta["design"],
        "true": float(meta["su2_cd"]),
        "pred": float(predictions[case]["pred_cd"]),
    })

usable = {}
dropped_span = 0
for job, steps in trajectories.items():
    if len(steps) < args.min_steps:
        continue
    steps = sorted(steps, key=lambda s: s["design"])
    values = np.array([s["true"] for s in steps])
    if values.max() - values.min() < args.min_span:
        dropped_span += 1
        continue
    usable[job] = steps
if args.min_span:
    print(f"궤적 폭이 {args.min_span} 미만이라 제외: {dropped_span}종 "
          "(라벨이 최적 설계를 지목할 수 없음)")
print(f"학습에서 안 본 예측이 있는 형상 {len(trajectories)}종 중 "
      f"{args.min_steps}단계 이상 궤적 {len(usable)}종\n")

best_hits, near_hits, spearman, step_hits, step_total = 0, 0, [], 0, 0
print(f"{'job':>10s} {'단계':>4s} {'최적 설계 (정답/예측)':>22s} {'순위상관':>8s} "
      f"{'단계방향':>9s}")
print("-" * 62)
for job, steps in sorted(usable.items()):
    true = np.array([s["true"] for s in steps])
    pred = np.array([s["pred"] for s in steps])
    best_true, best_pred = int(np.argmin(true)), int(np.argmin(pred))
    best_hits += best_true == best_pred
    # a neighbouring design is a near miss, not a failure, when steps are small
    near_hits += abs(best_true - best_pred) <= 1
    rank = lambda v: np.argsort(np.argsort(v))
    rho = float(np.corrcoef(rank(true), rank(pred))[0, 1]) if len(true) > 2 else float("nan")
    spearman.append(rho)
    d_true, d_pred = np.diff(true), np.diff(pred)
    big = np.abs(d_true) >= args.min_step_delta
    hits = int((np.sign(d_true[big]) == np.sign(d_pred[big])).sum())
    step_hits += hits
    step_total += int(big.sum())
    print(f"{job[:8]:>10s} {len(steps):4d} "
          f"{steps[best_true]['design'][-3:]+' / '+steps[best_pred]['design'][-3:]:>22s} "
          f"{rho:+8.2f} {f'{hits}/{int(big.sum())}':>9s}")

n = len(usable)

# chance rates computed per trajectory, not assumed: a uniform random guess over
# that trajectory's designs, and for +/-1 the share of designs within one step
chance_exact, chance_near, down_steps = [], [], 0
for steps in usable.values():
    k = len(steps)
    true = np.array([s["true"] for s in steps])
    best = int(np.argmin(true))
    chance_exact.append(1.0 / k)
    chance_near.append(sum(1 for i in range(k) if abs(i - best) <= 1) / k)
    diffs = np.diff(true)
    down_steps += int((diffs[np.abs(diffs) >= args.min_step_delta] < 0).sum())

majority_steps = max(down_steps, step_total - down_steps)
print(f"\n최적 설계 정확히 일치: {best_hits}/{n} = {100*best_hits/n:.0f}%   "
      f"(우연 {100*np.mean(chance_exact):.0f}%)")
print(f"최적 설계 ±1단계 이내: {near_hits}/{n} = {100*near_hits/n:.0f}%   "
      f"(우연 {100*np.mean(chance_near):.0f}%)")
print(f"궤적 순위상관 중앙값: {np.nanmedian(spearman):+.2f}   (우연 0)")
print(f"단계별 방향(유의한 단계만): {step_hits}/{step_total} = {100*step_hits/step_total:.0f}%   "
      f"(다수기준 {majority_steps}/{step_total} = {100*majority_steps/step_total:.0f}%)")
print(f"\n정답 궤적에서 기준설계(001)가 최적인 경우: "
      f"{sum(1 for s in usable.values() if int(np.argmin([x['true'] for x in s])) == 0)}/{n}")
