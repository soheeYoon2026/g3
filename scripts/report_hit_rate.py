"""Per-case relative error and hit rate, which is what decides whether a predicted
coefficient can be reported to a user as a number."""

import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", type=Path, nargs="+", required=True)
ap.add_argument("--labels", nargs="+", required=True)
args = ap.parse_args()

models = {}
for path, label in zip(args.jsonl, args.labels):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    models[label] = {r["case"]: r for r in rows
                     if r.get("model") == "fine_tuned" and "case" in r
                     and r.get("true_cd") is not None}

shared = sorted(set.intersection(*(set(m) for m in models.values())),
                key=lambda c: int(c.split("_")[1]))

head = f"{'case':>9s} {'true':>7s}" + "".join(f"{lab:>10s}{'err%':>7s}" for lab in models)
print(head)
print("-" * len(head))
for case in shared:
    true = models[args.labels[0]][case]["true_cd"]
    row = f"{case:>9s} {true:7.4f}"
    for label in models:
        pred = models[label][case]["pred_cd"]
        row += f"{pred:10.4f}{100*abs(pred-true)/abs(true):6.0f}%"
    print(row)

print()
for label, rows in models.items():
    true = np.array([rows[c]["true_cd"] for c in shared])
    pred = np.array([rows[c]["pred_cd"] for c in shared])
    rel = np.abs(pred - true) / np.abs(true)
    within = {t: int((rel <= t).sum()) for t in (0.05, 0.10, 0.20, 0.30)}
    print(f"{label}: 상대오차 중앙값 {100*np.median(rel):.0f}%  최악 {100*rel.max():.0f}%")
    print("   " + "  ".join(f"±{int(100*t)}% 이내 {n}/{len(shared)}" for t, n in within.items()))
