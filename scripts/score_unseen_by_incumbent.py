"""Compare the serving model and the cross-validated model on geometry neither saw.

The serving model was trained on the reaudit set, which shares 8 of the pair
benchmark's 22 car geometries, so its headline score on all 63 pairs is partly
recall. The cross-validated model is geometry-separated by construction. Scoring
both on the subset the serving model never saw is the only fair comparison.
"""

import argparse
import json
import re
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
ap.add_argument("--incumbent-jsonl", type=Path, required=True)
ap.add_argument("--cv-dir", type=Path, required=True)
ap.add_argument("--pairs", type=Path, required=True)
ap.add_argument("--run-index", type=Path, required=True)
ap.add_argument("--incumbent-log", type=Path, required=True)
ap.add_argument("--reaudit-root", type=Path, required=True)
args = ap.parse_args()

index = json.loads(args.run_index.read_text())
log = args.incumbent_log.read_text()
seen_runs, order = set(), []
for run in re.findall(r"load_case\(.run_([0-9]+).\)", log):
    if run not in seen_runs:
        seen_runs.add(run)
        order.append(run)
seen_jobs = set()
for run in order:
    path = args.reaudit_root / f"run_{run}" / f"conditions_{run}.json"
    if not path.exists():
        continue
    match = re.search(r"([A-Za-z0-9]{22})_CFD\.cfg",
                      json.loads(path.read_text()).get("source_cfg") or "")
    if match:
        seen_jobs.add(match.group(1))

pairs = json.loads(args.pairs.read_text())["pairs"]
unseen = [p for p in pairs
          if index[str(p["baseline"])]["job_uid"] not in seen_jobs]
print(f"차량형 채점 쌍 {len(pairs)}개 중 현행이 못 본 형상의 쌍 {len(unseen)}개")

cv_rows = {}
for fold in sorted(args.cv_dir.glob("fold*")):
    cv_rows.update(load(fold / "eval-test.jsonl"))
incumbent_rows = load(args.incumbent_jsonl)

for name, rows in (("현행 (일부 암기)", incumbent_rows), ("교차검증 모델", cv_rows)):
    for label, subset in (("전체 63쌍", pairs), ("미접촉 형상만", unseen)):
        pred, true = [], []
        for pair in subset:
            b, v = f"run_{pair['baseline']}", f"run_{pair['variant']}"
            if b in rows and v in rows:
                pred.append(rows[v]["pred_cd"] - rows[b]["pred_cd"])
                true.append(pair["true_delta_cd"])
        pred, true = np.array(pred), np.array(true)
        n = len(true)
        if n == 0:
            print(f"{name} / {label}: 예측 없음")
            continue
        majority = max(int((true < 0).sum()), int((true > 0).sum()))
        hits = int((np.sign(pred) == np.sign(true)).sum())
        p_value = sum(comb(n, j) * (majority / n) ** j * (1 - majority / n) ** (n - j)
                      for j in range(hits, n + 1))
        print(f"{name:>16s} / {label:>10s}: {hits}/{n} = {100*hits/n:3.0f}%  "
              f"다수기준 {100*majority/n:3.0f}%  p {p_value:.4f}  "
              f"값상관 {float(np.corrcoef(pred,true)[0,1]):+.2f}  "
              f"ΔCd MAE {np.abs(pred-true).mean():.4f}")
