"""Per-pair breakdown of the question the product actually asks: move a control
point, does drag go up or down?

Also reports how much statistical power six pairs give -- with a coin flip you
reach four correct out of six about a third of the time.
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
        if row.get("model") == "fine_tuned" and "case" in row and row.get("true_cd") is not None:
            rows[row["case"]] = row
    return rows


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--jsonl", type=Path, nargs="+", required=True)
ap.add_argument("--labels", nargs="+", required=True)
ap.add_argument("--pairs", type=Path, required=True)
args = ap.parse_args()

manifest = json.loads(args.pairs.read_text())
pairs = manifest["pairs"] if isinstance(manifest, dict) and "pairs" in manifest else manifest
models = {lab: load(p) for p, lab in zip(args.jsonl, args.labels)}

print(f"{'pair':>28s} {'true dCd':>9s}" + "".join(f"{lab:>14s}" for lab in models))
print("-" * (38 + 14 * len(models)))
hits = {lab: 0 for lab in models}
total = 0
for pair in pairs:
    base, var = f"run_{pair['baseline']}", f"run_{pair['variant']}"
    label = pair.get("name") or pair.get("label") or f"{base}->{var}"
    any_model = next(iter(models.values()))
    if base not in any_model or var not in any_model:
        print(f"{label:>28s}  (예측 없음)")
        continue
    true_delta = any_model[var]["true_cd"] - any_model[base]["true_cd"]
    total += 1
    line = f"{label:>28s} {true_delta:+9.4f}"
    for lab, rows in models.items():
        pred_delta = rows[var]["pred_cd"] - rows[base]["pred_cd"]
        ok = np.sign(pred_delta) == np.sign(true_delta)
        hits[lab] += int(ok)
        line += f" {pred_delta:+9.4f}{'O' if ok else 'X':>4s}"
    print(line)

print()
for lab in models:
    print(f"{lab}: 방향 {hits[lab]}/{total}")
if total:
    p_at_least = sum(comb(total, k) for k in range(max(hits.values()), total + 1)) / 2 ** total
    print(f"\n동전던지기로 {max(hits.values())}/{total} 이상 나올 확률: {100*p_at_least:.0f}%")
