"""Measure the ΔCl error the UI has been showing without one.

ΔCd has a measured error bar; ΔCl is displayed beside it with none, so the
interface implies the two are equally trustworthy. The balanced benchmark already
carries lift labels - integrated from the stored surface field, the same
integration that reproduced the manifest's drag to 0.7% - so the error can be
measured rather than guessed.

Reports the same way as ΔCd: MAE, direction against the majority answer within
this set, and correlation.
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
ap.add_argument("--cv-dir", type=Path, required=True)
ap.add_argument("--root", type=Path, required=True)
args = ap.parse_args()


def true_cl(run):
    path = args.root / f"run_{run}" / f"conditions_{run}.json"
    return float(json.loads(path.read_text())["su2_cl"])


records = []
for fold in sorted(args.cv_dir.glob("fold*")):
    rows = load(fold / "eval-test.jsonl")
    for pair in json.loads((fold / "pairs-test.json").read_text())["pairs"]:
        base, var = pair["baseline"], pair["variant"]
        if f"run_{base}" not in rows or f"run_{var}" not in rows:
            continue
        records.append({
            "pred_cd": rows[f"run_{var}"]["pred_cd"] - rows[f"run_{base}"]["pred_cd"],
            "true_cd": pair["true_delta_cd"],
            "pred_cl": rows[f"run_{var}"]["pred_cl"] - rows[f"run_{base}"]["pred_cl"],
            "true_cl": true_cl(var) - true_cl(base),
        })

n = len(records)
print(f"쌍 {n}개\n")
for name, pred_key, true_key in (("ΔCd", "pred_cd", "true_cd"),
                                 ("ΔCl", "pred_cl", "true_cl")):
    pred = np.array([r[pred_key] for r in records])
    true = np.array([r[true_key] for r in records])
    down, up = int((true < 0).sum()), int((true > 0).sum())
    majority = max(down, up)
    hits = int((np.sign(pred) == np.sign(true)).sum())
    p_value = sum(comb(n, j) * (majority / n) ** j * (1 - majority / n) ** (n - j)
                  for j in range(hits, n + 1))
    print(f"=== {name} ===")
    print(f"  정답 분포: 감소 {down} / 증가 {up}  -> 다수기준 {100*majority/n:.0f}%")
    print(f"  |정답| 중앙값 {np.median(np.abs(true)):.5f}  "
          f"범위 {np.abs(true).min():.5f}~{np.abs(true).max():.5f}")
    print(f"  방향 {hits}/{n} = {100*hits/n:.0f}%  다수기준 대비 p {p_value:.4f}")
    print(f"  **MAE {np.abs(pred-true).mean():.4f}**  "
          f"중앙값 오차 {np.median(np.abs(pred-true)):.4f}")
    print(f"  값상관 {float(np.corrcoef(pred, true)[0,1]):+.2f}\n")

cd_mae = np.abs(np.array([r["pred_cd"] for r in records])
                - np.array([r["true_cd"] for r in records])).mean()
cl_mae = np.abs(np.array([r["pred_cl"] for r in records])
                - np.array([r["true_cl"] for r in records])).mean()
print(f"UI 표기용: ΔCd ±{cd_mae:.3f}, ΔCl ±{cl_mae:.3f}")
