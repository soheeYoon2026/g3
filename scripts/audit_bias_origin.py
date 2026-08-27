"""Is the ΔCd bias a model artefact, or the benchmark's prior in disguise?

Subtracting mean(pred - true) equals subtracting -mean(true) whenever the model
predicts near zero, and mean(true) is negative here because optimiser steps
mostly reduce drag. That would make the correction a dressed-up majority answer
rather than a fix. The test: after correcting, does the model still call any pair
"drag went up", and is it right when it does? A majority answer never says up.
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
        if base in rows and var in rows:
            pred.append(rows[var]["pred_cd"] - rows[base]["pred_cd"])
            true.append(pair["true_delta_cd"])
    return np.array(pred), np.array(true)


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--cv-dir", type=Path, required=True, help="crossval work dir with fold*/")
args = ap.parse_args()

raw, corrected, truth = [], [], []
for fold in sorted(args.cv_dir.glob("fold*")):
    train_pairs = json.loads((fold / "pairs-train.json").read_text())["pairs"]
    test_pairs = json.loads((fold / "pairs-test.json").read_text())["pairs"]
    pred_tr, true_tr = deltas(load(fold / "eval-train.jsonl"), train_pairs)
    pred_te, true_te = deltas(load(fold / "eval-test.jsonl"), test_pairs)
    bias = float(np.mean(pred_tr - true_tr))
    print(f"{fold.name}: 편향 {bias:+.5f} | 학습쌍 평균(정답) {np.mean(true_tr):+.5f} "
          f"| 학습쌍 평균(예측) {np.mean(pred_tr):+.5f}")
    raw.append(pred_te)
    corrected.append(pred_te - bias)
    truth.append(true_te)

raw = np.concatenate(raw)
corrected = np.concatenate(corrected)
truth = np.concatenate(truth)
n = len(truth)

print(f"\n전체 {n}쌍")
print(f"정답 평균 {truth.mean():+.5f}  |  예측 평균 {raw.mean():+.5f}  "
      f"|  차이 {raw.mean()-truth.mean():+.5f}")
print("→ 예측 평균이 0에 가깝고 정답 평균이 음수라면, 편향 제거는 사전확률 주입에 가깝다")

for name, pred in (("보정 없음", raw), ("편향 제거", corrected),
                   ("무조건 감소", -np.ones(n))):
    says_up = pred > 0
    up_correct = int(((truth > 0) & says_up).sum())
    down_correct = int(((truth < 0) & ~says_up).sum())
    hits = up_correct + down_correct
    print(f"\n{name}: 방향 {hits}/{n} = {100*hits/n:.0f}%")
    print(f"   '증가'라고 답한 쌍 {int(says_up.sum())}개 중 정답 {up_correct}개"
          + (f" (정밀도 {100*up_correct/says_up.sum():.0f}%)" if says_up.sum() else ""))
    print(f"   '감소'라고 답한 쌍 {int((~says_up).sum())}개 중 정답 {down_correct}개"
          + (f" (정밀도 {100*down_correct/(~says_up).sum():.0f}%)" if (~says_up).sum() else ""))

actual_up = int((truth > 0).sum())
print(f"\n실제로 항력이 증가한 쌍은 {actual_up}/{n}개다.")
print("편향 제거가 사전확률 주입에 불과하다면 '증가' 응답이 사라져야 하고,")
print("실제 신호가 있다면 '증가' 응답이 남으면서 정밀도가 우연(=%.0f%%)을 넘어야 한다."
      % (100 * actual_up / n))
