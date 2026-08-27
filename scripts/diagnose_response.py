"""Is the model wrong about the deformation, or blind to it?

A model that responds to a control-point move but mis-signs it is a different
problem from one whose output barely changes at all. This compares the size of
each model's predicted ΔCd against the true ΔCd, and against the size of the
deformation itself.
"""

import argparse
import json
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
ap.add_argument("--jsonl", type=Path, nargs="+", required=True)
ap.add_argument("--labels", nargs="+", required=True)
ap.add_argument("--pairs", type=Path, required=True)
args = ap.parse_args()

pairs = json.loads(args.pairs.read_text())["pairs"]
models = {lab: load(p) for p, lab in zip(args.jsonl, args.labels)}

truth, preds = [], {lab: [] for lab in models}
for pair in pairs:
    base, var = f"run_{pair['baseline']}", f"run_{pair['variant']}"
    if any(base not in m or var not in m for m in models.values()):
        continue
    truth.append(pair["true_delta_cd"])
    for lab, rows in models.items():
        preds[lab].append(rows[var]["pred_cd"] - rows[base]["pred_cd"])

truth = np.array(truth)
print(f"쌍 {len(truth)}개")
print(f"정답 |ΔCd|: 중앙값 {np.median(np.abs(truth)):.5f}  "
      f"사분위 {np.percentile(np.abs(truth),25):.5f}/{np.percentile(np.abs(truth),75):.5f}\n")

print(f"{'모델':>10s} {'예측|ΔCd| 중앙값':>16s} {'정답 대비':>9s} "
      f"{'예측이 더 작은 쌍':>16s} {'거의 무반응(<0.001)':>19s}")
print("-" * 78)
for lab in models:
    pred = np.abs(np.array(preds[lab]))
    ratio = np.median(pred) / np.median(np.abs(truth))
    smaller = int((pred < np.abs(truth)).sum())
    flat = int((pred < 0.001).sum())
    print(f"{lab:>10s} {np.median(pred):16.5f} {ratio:8.2f}x "
          f"{smaller:8d}/{len(truth):<7d} {flat:12d}/{len(truth)}")

print("\n예측 ΔCd 분포 (모델별)")
for lab in models:
    pred = np.array(preds[lab])
    print(f"  {lab}: 최소 {pred.min():+.4f}  중앙 {np.median(pred):+.4f}  "
          f"최대 {pred.max():+.4f}  표준편차 {pred.std():.4f}")
print(f"  정답:   최소 {truth.min():+.4f}  중앙 {np.median(truth):+.4f}  "
      f"최대 {truth.max():+.4f}  표준편차 {truth.std():.4f}")
