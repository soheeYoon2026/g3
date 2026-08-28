"""Cross-validate with a large training pair set and a clean scoring set.

Training uses the ΔCp field loss, which never reads ΔCd, so the |ΔCd| noise floor
that the benchmark needs does not apply to training material: every physically
valid pair is usable. Scoring still uses only pairs above the floor, because a
label below it carries no direction to be right or wrong about.

Folds are drawn over the scoring geometries; each fold trains on every other
geometry's pairs, so no shape appears on both sides.
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", type=Path, required=True, help="dataset holding all runs")
ap.add_argument("--run-index", type=Path, required=True, help="run -> job/design map")
ap.add_argument("--score-manifest", type=Path, required=True,
                help="filtered benchmark (job_uid + designs) defining the scoring pairs")
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--trainer", type=Path, required=True)
ap.add_argument("--evaluator", type=Path, required=True)
ap.add_argument("--eval-cwd", type=Path, required=True)
ap.add_argument("--python", default=sys.executable)
ap.add_argument("--folds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--seed", type=int, default=20260827)
args = ap.parse_args()

index = json.loads(args.run_index.read_text())
run_of = {(v["job_uid"], v["design"]): int(run) for run, v in index.items()}
job_of_run = {int(run): v["job_uid"] for run, v in index.items()}

all_pairs = json.loads(Path(args.root / "pairs.json").read_text())["pairs"]
score_keys = {(p["job_uid"], p["base_design"], p["target_design"])
              for p in json.loads(args.score_manifest.read_text())["pairs"]}

# rebuild each fetched pair's identity so the scoring subset can be selected
pair_key = {}
for entry in json.loads(args.score_manifest.read_text())["pairs"]:
    base = run_of.get((entry["job_uid"], entry["base_design"]))
    target = run_of.get((entry["job_uid"], entry["target_design"]))
    if base is not None and target is not None:
        pair_key[(base, target)] = entry["job_uid"]

train_by_job, score_by_job = defaultdict(list), defaultdict(list)
for pair in all_pairs:
    job = job_of_run.get(pair["baseline"])
    if job is None:
        continue
    train_by_job[job].append(pair)
    if (pair["baseline"], pair["variant"]) in pair_key:
        score_by_job[job].append(pair)

score_jobs = sorted(score_by_job)
print(f"학습 가능 형상 {len(train_by_job)}종 / 쌍 {sum(len(v) for v in train_by_job.values())}개")
print(f"채점 형상 {len(score_jobs)}종 / 쌍 {sum(len(v) for v in score_by_job.values())}개")

rng = np.random.default_rng(args.seed)
shuffled = list(score_jobs)
rng.shuffle(shuffled)
folds = [shuffled[i::args.folds] for i in range(args.folds)]
args.work.mkdir(parents=True, exist_ok=True)


def load_deltas(path, subset):
    rows = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("model") == "fine_tuned" and "case" in row:
            rows[row["case"]] = row
    pred, true = [], []
    for pair in subset:
        base, var = f"run_{pair['baseline']}", f"run_{pair['variant']}"
        if base in rows and var in rows:
            pred.append(rows[var]["pred_cd"] - rows[base]["pred_cd"])
            true.append(pair["true_delta_cd"])
    return np.array(pred), np.array(true)


raw, debiased, truth = [], [], []
for k, held in enumerate(folds):
    held_set = set(held)
    train_rows = [p for job, rows in train_by_job.items() if job not in held_set for p in rows]
    test_rows = [p for job in held for p in score_by_job[job]]
    train_runs = sorted({r for p in train_rows for r in (p["baseline"], p["variant"])})
    test_runs = sorted({r for p in test_rows for r in (p["baseline"], p["variant"])})
    overlap = set(train_runs) & set(test_runs)
    if overlap:
        raise SystemExit(f"fold {k}: 런 {len(overlap)}개 중복 — 누수")

    base_dir = args.work / f"fold{k}"
    base_dir.mkdir(exist_ok=True)
    (base_dir / "split.json").write_text(json.dumps({
        "train_cases": [{"run": r, "group_id": None} for r in train_runs],
        "test_cases": [{"run": r, "group_id": None} for r in test_runs]}))
    (base_dir / "split-test.json").write_text(json.dumps({
        "test_cases": [{"run": r, "group_id": None} for r in test_runs]}))
    (base_dir / "pairs-train.json").write_text(json.dumps({"pairs": train_rows}))
    (base_dir / "pairs-test.json").write_text(json.dumps({"pairs": test_rows}))
    print(f"fold {k}: 학습 {len(train_rows)}쌍/{len(train_runs)}런  "
          f"채점 {len(test_rows)}쌍/{len(test_runs)}런", flush=True)

    checkpoint = base_dir / "model.pt"
    with open(base_dir / "train.log", "w") as log:
        rc = subprocess.run(
            [args.python, str(args.trainer), "--root", str(args.root),
             "--split", str(base_dir / "split.json"),
             "--pairs", str(base_dir / "pairs-train.json"),
             "--out", str(checkpoint), "--epochs", str(args.epochs)],
            stdout=log, stderr=subprocess.STDOUT, timeout=28800).returncode
    if rc != 0:
        print(f"  학습 실패 rc={rc}", flush=True)
        continue

    for tag, split, partition in (("train", "split.json", "train"),
                                  ("test", "split-test.json", "test")):
        with open(base_dir / f"eval-{tag}.jsonl", "w") as out:
            subprocess.run(
                [args.python, str(args.evaluator), "--root", str(args.root),
                 "--fine-tuned", str(checkpoint), "--split", str(base_dir / split),
                 "--partition", partition],
                stdout=out, stderr=subprocess.DEVNULL, cwd=str(args.eval_cwd), timeout=28800)

    pred_tr, true_tr = load_deltas(base_dir / "eval-train.jsonl", train_rows)
    pred_te, true_te = load_deltas(base_dir / "eval-test.jsonl", test_rows)
    bias = float(np.mean(pred_tr - true_tr)) if len(pred_tr) else 0.0
    print(f"  학습쌍 편향 {bias:+.5f} | 채점쌍 {len(pred_te)}개", flush=True)
    raw.append(pred_te)
    debiased.append(pred_te - bias)
    truth.append(true_te)

raw, debiased, truth = (np.concatenate(x) for x in (raw, debiased, truth))
n = len(truth)
majority = max(int((truth < 0).sum()), int((truth > 0).sum()))
p_major = majority / n
print(f"\n전체 채점 {n}쌍, 다수기준 {majority}/{n} = {100*p_major:.0f}%")
for name, pred in (("보정 없음", raw), ("평균편향 제거", debiased)):
    hits = int((np.sign(pred) == np.sign(truth)).sum())
    p_value = sum(comb(n, j) * p_major ** j * (1 - p_major) ** (n - j)
                  for j in range(hits, n + 1))
    says_up = pred > 0
    up_ok = int(((truth > 0) & says_up).sum())
    print(f"  {name}: {hits}/{n} = {100*hits/n:.0f}%  p {p_value:.4f}  "
          f"값상관 {float(np.corrcoef(pred, truth)[0,1]):+.2f}  "
          f"ΔCd MAE {np.abs(pred-truth).mean():.4f}  "
          f"'증가' {int(says_up.sum())}개 중 정답 {up_ok}개")
