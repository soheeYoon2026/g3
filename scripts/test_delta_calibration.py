"""Does a bias correction fitted on training pairs fix the ΔCd sign errors?

A model can track ΔCd magnitude closely and still call the direction wrong if its
predictions carry a constant offset: the errors then concentrate near zero, where
the sign lives. The offset is fitted on the training pairs only and applied to the
held-out ones, so the held-out score stays honest.
"""

import argparse
import json
from math import comb
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


def deltas(rows, pairs):
    pred, true = [], []
    for pair in pairs:
        base, var = f"run_{pair['baseline']}", f"run_{pair['variant']}"
        if base not in rows or var not in rows:
            continue
        pred.append(rows[var]["pred_cd"] - rows[base]["pred_cd"])
        true.append(pair["true_delta_cd"])
    return np.array(pred), np.array(true)


def binomial_tail(k, n, p):
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--train-jsonl", type=Path, required=True)
ap.add_argument("--test-jsonl", type=Path, required=True)
ap.add_argument("--train-pairs", type=Path, required=True)
ap.add_argument("--test-pairs", type=Path, required=True)
ap.add_argument("--label", default="model")
args = ap.parse_args()

train_pairs = json.loads(args.train_pairs.read_text())["pairs"]
test_pairs = json.loads(args.test_pairs.read_text())["pairs"]
pred_tr, true_tr = deltas(load(args.train_jsonl), train_pairs)
pred_te, true_te = deltas(load(args.test_jsonl), test_pairs)
print(f"{args.label}: 학습쌍 {len(pred_tr)}개, 평가쌍 {len(pred_te)}개")

slope, intercept = np.polyfit(pred_tr, true_tr, 1)
print(f"학습쌍에서 적합: true ≈ {slope:.3f} * pred {intercept:+.5f}")
print(f"  학습쌍 예측 편향(pred-true 평균) {np.mean(pred_tr - true_tr):+.5f}")

n = len(true_te)
majority = max(int((true_te < 0).sum()), int((true_te > 0).sum()))
p_major = majority / n
print(f"평가쌍 다수기준 {majority}/{n} = {100*p_major:.0f}%\n")

variants = {
    "보정 없음": pred_te,
    "평균편향 제거": pred_te - np.mean(pred_tr - true_tr),
    "선형보정": slope * pred_te + intercept,
}
print(f"{'보정':>14s} {'방향':>10s} {'p':>8s} {'값상관':>7s} {'ΔCd MAE':>9s}")
print("-" * 52)
for name, adjusted in variants.items():
    hits = int((np.sign(adjusted) == np.sign(true_te)).sum())
    p_value = binomial_tail(hits, n, p_major)
    corr = float(np.corrcoef(adjusted, true_te)[0, 1])
    mae = float(np.abs(adjusted - true_te).mean())
    print(f"{name:>14s} {hits:3d}/{n:<4d} {100*hits/n:3.0f}% {p_value:8.4f} "
          f"{corr:+7.2f} {mae:9.4f}")

near = np.abs(true_te) < 0.005
print(f"\n|정답 ΔCd| < 0.005 인 쌍 {int(near.sum())}/{n} — 부호가 잘 뒤집히는 구간")
for name, adjusted in variants.items():
    if near.sum():
        hits = int((np.sign(adjusted[near]) == np.sign(true_te[near])).sum())
        big = int((np.sign(adjusted[~near]) == np.sign(true_te[~near])).sum())
        print(f"  {name}: 작은쪽 {hits}/{int(near.sum())}, 큰쪽 {big}/{int((~near).sum())}")
