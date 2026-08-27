"""Score models on the product question over a large pair set.

With eighty-odd pairs the interesting comparison is no longer against a coin
flip but against the majority answer: this pool is 73% "drag went down", so a
model that always says so already scores 73%. Reports each model against that
baseline with a binomial test, plus the magnitude rank correlation, which the
majority answer cannot fake.
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


def binomial_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--jsonl", type=Path, nargs="+", required=True)
ap.add_argument("--labels", nargs="+", required=True)
ap.add_argument("--pairs", type=Path, required=True)
ap.add_argument("--min-delta", type=float, default=0.0,
                help="optionally restrict to pairs above this |ΔCd|")
args = ap.parse_args()

pairs = json.loads(args.pairs.read_text())["pairs"]
if args.min_delta:
    pairs = [p for p in pairs if abs(p["true_delta_cd"]) >= args.min_delta]
models = {lab: load(p) for p, lab in zip(args.jsonl, args.labels)}

truth, preds = [], {lab: [] for lab in models}
missing = 0
for pair in pairs:
    base, var = f"run_{pair['baseline']}", f"run_{pair['variant']}"
    if any(base not in m or var not in m for m in models.values()):
        missing += 1
        continue
    truth.append(pair["true_delta_cd"])
    for lab, rows in models.items():
        preds[lab].append(rows[var]["pred_cd"] - rows[base]["pred_cd"])

truth = np.array(truth)
n = len(truth)
down = int((truth < 0).sum())
majority = max(down, n - down)
p_major = majority / n
print(f"쌍 {n}개 (예측 누락 {missing}개 제외)")
print(f"다수 방향 기준선: {majority}/{n} = {100*p_major:.0f}%\n")

print(f"{'모델':>10s} {'방향':>10s} {'다수기준 대비 p':>15s} {'순위상관':>9s} {'값상관':>8s} {'ΔCd MAE':>9s}")
print("-" * 66)
rank = lambda v: np.argsort(np.argsort(v))
for lab in models:
    pred = np.array(preds[lab])
    hits = int((np.sign(pred) == np.sign(truth)).sum())
    p_value = binomial_tail(hits, n, p_major)
    spear = float(np.corrcoef(rank(pred), rank(truth))[0, 1])
    pear = float(np.corrcoef(pred, truth)[0, 1])
    mae = float(np.abs(pred - truth).mean())
    print(f"{lab:>10s} {hits:4d}/{n:<5d} {100*hits/n:3.0f}%  {p_value:13.4f} "
          f"{spear:+9.2f} {pear:+8.2f} {mae:9.4f}")

print("\n다수기준 대비 p = '항상 다수 방향으로만 답하는 모델'이 이 성적 이상을 낼 확률")
