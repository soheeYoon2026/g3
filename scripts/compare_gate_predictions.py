"""Show per-case gate predictions side by side, and split each model's error into
a systematic part (offset and scale) and a residual scatter.

A high rank correlation with a large MAE means the model orders the cases right
but reports them shifted, which is a different problem from noise and has a
different fix.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", type=Path, nargs="+", required=True,
                help="one evaluate_domino_v3 output per model")
ap.add_argument("--labels", nargs="+", required=True)
args = ap.parse_args()
if len(args.jsonl) != len(args.labels):
    raise SystemExit("need one label per file")

models = {}
for path, label in zip(args.jsonl, args.labels):
    rows = []
    for line in path.read_text().splitlines():  # the evaluator interleaves log lines
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    cases = [r for r in rows if "case" in r and r.get("true_cd") is not None]
    if not cases:
        raise SystemExit(f"{path}: no per-case rows found")
    models[label] = {r["case"]: r for r in cases}

shared = sorted(set.intersection(*(set(m) for m in models.values())),
                key=lambda c: int(c.split("_")[1]))
print(f"공통 케이스 {len(shared)}개\n")

header = f"{'case':>10s} {'true Cd':>9s}" + "".join(f"{lab:>12s}" for lab in models)
print(header)
print("-" * len(header))
for case in shared:
    true = models[args.labels[0]][case]["true_cd"]
    line = f"{case:>10s} {true:9.4f}"
    for label in models:
        line += f"{models[label][case]['pred_cd']:12.4f}"
    print(line)

print()
for label, rows in models.items():
    true = np.array([rows[c]["true_cd"] for c in shared])
    pred = np.array([rows[c]["pred_cd"] for c in shared])
    mae = np.abs(pred - true).mean()
    bias = (pred - true).mean()
    slope, intercept = np.polyfit(true, pred, 1)
    residual = np.abs(pred - (slope * true + intercept)).mean()
    print(f"{label}: MAE {mae:.4f} | 평균편향 {bias:+.4f} | "
          f"기울기 {slope:.3f} 절편 {intercept:+.4f} | 보정 후 잔차 MAE {residual:.4f}")
