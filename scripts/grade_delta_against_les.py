"""Grade the product question -- move a control point, does drag improve? --
against both available truths.

G2 supplies the absolute coefficients the models are trained on, but its ΔCd on
two of these deformations contradicts LES, and LES is the ΔCd truth in this
project's division of labour (G2 absolute / LES delta). Scoring only against G2
therefore penalises a model for agreeing with the better measurement.
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


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--jsonl", type=Path, nargs="+", required=True)
ap.add_argument("--labels", nargs="+", required=True)
ap.add_argument("--benchmark", type=Path, required=True)
args = ap.parse_args()

pairs = json.loads(args.benchmark.read_text())["pairs"]
models = {lab: load(p) for p, lab in zip(args.jsonl, args.labels)}

print(f"{'변형':>20s} {'G2 ΔCd':>9s} {'LES ΔCd':>9s} {'일치':>5s}" +
      "".join(f"{lab:>16s}" for lab in models))
print("-" * (46 + 16 * len(models)))

score = {lab: {"g2": 0, "les": 0} for lab in models}
counted = {"g2": 0, "les": 0}
for pair in pairs:
    base, var = f"run_{pair['baseline']}", f"run_{pair['variant']}"
    g2 = pair["true_delta_cd"]
    les = pair.get("les_delta_cd")
    agree = pair.get("direction_agree_g2_les")
    line = f"{pair['note']:>20s} {g2:+9.4f} " + \
           (f"{les:+9.4f}" if les is not None else f"{'-':>9s}") + \
           f" {('O' if agree else 'X'):>5s}"
    counted["g2"] += 1
    if les is not None:
        counted["les"] += 1
    for lab, rows in models.items():
        delta = rows[var]["pred_cd"] - rows[base]["pred_cd"]
        ok_g2 = np.sign(delta) == np.sign(g2)
        score[lab]["g2"] += int(ok_g2)
        mark = "O" if ok_g2 else "X"
        if les is not None:
            ok_les = np.sign(delta) == np.sign(les)
            score[lab]["les"] += int(ok_les)
            mark += "/" + ("O" if ok_les else "X")
        line += f" {delta:+9.4f}{mark:>6s}"
    print(line)

print(f"\n{'모델':>12s}  {'G2 기준':>10s}  {'LES 기준':>10s}")
for lab in models:
    print(f"{lab:>12s}  {score[lab]['g2']}/{counted['g2']:<8d}  "
          f"{score[lab]['les']}/{counted['les']}")

n = counted["les"]
les_deltas = np.array([p["les_delta_cd"] for p in pairs if p.get("les_delta_cd") is not None])
print(f"\nLES ΔCd 부호 분포: 음 {int((les_deltas<0).sum())} / 양 {int((les_deltas>0).sum())} "
      f"— 항상 '감소'라고만 답해도 {int((les_deltas<0).sum())}/{n}을 맞춘다")

for lab in models:
    k = score[lab]["les"]
    p = sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n
    preds = np.array([models[lab][f"run_{q['variant']}"]["pred_cd"]
                      - models[lab][f"run_{q['baseline']}"]["pred_cd"]
                      for q in pairs if q.get("les_delta_cd") is not None])
    rank = lambda v: np.argsort(np.argsort(v))
    spear = float(np.corrcoef(rank(preds), rank(les_deltas))[0, 1])
    pearson = float(np.corrcoef(preds, les_deltas)[0, 1])
    print(f"{lab}: 방향 {k}/{n} (우연 확률 {100*p:.1f}%) | "
          f"LES와 크기 순위상관 {spear:+.2f} | 값 상관 {pearson:+.2f}")
