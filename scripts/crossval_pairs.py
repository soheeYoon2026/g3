"""Cross-validate the pair training over geometries.

A single 31-pair holdout was not enough to separate a real effect from the
several calibration variants that were tried on it. Rotating three geometry folds
puts every one of the 83 pairs on the held-out side exactly once, so the direction
score is computed on all of them without any pair ever being trained on.
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
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--pairs", type=Path, required=True)
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--trainer", type=Path, required=True)
ap.add_argument("--evaluator", type=Path, required=True)
ap.add_argument("--eval-cwd", type=Path, required=True)
ap.add_argument("--python", default=sys.executable)
ap.add_argument("--folds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--seed", type=int, default=20260827)
args = ap.parse_args()

pairs = json.loads(args.pairs.read_text())["pairs"]
by_job = defaultdict(list)
for pair in pairs:
    by_job[pair["note"].split(":")[0]].append(pair)

jobs = sorted(by_job)
rng = np.random.default_rng(args.seed)
rng.shuffle(jobs)
folds = [jobs[i::args.folds] for i in range(args.folds)]
args.work.mkdir(parents=True, exist_ok=True)


def run(cmd, log):
    with open(log, "w") as stream:
        return subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=7200).returncode


def load_deltas(jsonl, subset):
    rows = {}
    for line in Path(jsonl).read_text().splitlines():
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


all_raw, all_debiased, all_true = [], [], []
for k, held in enumerate(folds):
    held_set = set(held)
    train_rows = [p for j in jobs if j not in held_set for p in by_job[j]]
    test_rows = [p for j in held for p in by_job[j]]
    train_runs = sorted({r for p in train_rows for r in (p["baseline"], p["variant"])})
    test_runs = sorted({r for p in test_rows for r in (p["baseline"], p["variant"])})
    if set(train_runs) & set(test_runs):
        raise SystemExit(f"fold {k}: 런 중복 — 분할 실패")

    base = args.work / f"fold{k}"
    base.mkdir(exist_ok=True)
    (base / "split.json").write_text(json.dumps({
        "train_cases": [{"run": r, "group_id": None} for r in train_runs],
        "test_cases": [{"run": r, "group_id": None} for r in test_runs]}))
    (base / "split-test.json").write_text(json.dumps({
        "test_cases": [{"run": r, "group_id": None} for r in test_runs]}))
    (base / "pairs-train.json").write_text(json.dumps({"pairs": train_rows}))
    (base / "pairs-test.json").write_text(json.dumps({"pairs": test_rows}))

    print(f"fold {k}: 학습 형상 {len(jobs)-len(held)}종/{len(train_rows)}쌍  "
          f"평가 형상 {len(held)}종/{len(test_rows)}쌍", flush=True)

    checkpoint = base / "model.pt"
    rc = run([args.python, str(args.trainer), "--root", str(args.root),
              "--split", str(base / "split.json"), "--pairs", str(base / "pairs-train.json"),
              "--out", str(checkpoint), "--epochs", str(args.epochs)], base / "train.log")
    if rc != 0:
        print(f"  학습 실패 rc={rc}", flush=True)
        continue

    for tag, split, subset in (("train", base / "split.json", train_rows),
                               ("test", base / "split-test.json", test_rows)):
        out = base / f"eval-{tag}.jsonl"
        cmd = [args.python, str(args.evaluator), "--root", str(args.root),
               "--fine-tuned", str(checkpoint), "--split", str(split),
               "--partition", "train" if tag == "train" else "test"]
        with open(out, "w") as stream:
            subprocess.run(cmd, stdout=stream, stderr=subprocess.DEVNULL,
                           cwd=str(args.eval_cwd), timeout=7200)

    pred_tr, true_tr = load_deltas(base / "eval-train.jsonl", train_rows)
    pred_te, true_te = load_deltas(base / "eval-test.jsonl", test_rows)
    bias = float(np.mean(pred_tr - true_tr))
    print(f"  학습쌍 편향 {bias:+.5f} | 평가쌍 {len(pred_te)}개", flush=True)
    all_raw.append(pred_te)
    all_debiased.append(pred_te - bias)
    all_true.append(true_te)

raw = np.concatenate(all_raw)
debiased = np.concatenate(all_debiased)
true = np.concatenate(all_true)
n = len(true)
majority = max(int((true < 0).sum()), int((true > 0).sum()))
p_major = majority / n
print(f"\n전체 held-out {n}쌍, 다수기준 {majority}/{n} = {100*p_major:.0f}%")
for name, pred in (("보정 없음", raw), ("평균편향 제거", debiased)):
    hits = int((np.sign(pred) == np.sign(true)).sum())
    p_value = sum(comb(n, j) * p_major ** j * (1 - p_major) ** (n - j)
                  for j in range(hits, n + 1))
    corr = float(np.corrcoef(pred, true)[0, 1])
    print(f"  {name}: {hits}/{n} = {100*hits/n:.0f}%  "
          f"다수기준 대비 p {p_value:.4f}  값상관 {corr:+.2f}  "
          f"ΔCd MAE {np.abs(pred-true).mean():.4f}")
