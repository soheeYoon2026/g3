"""Score the product question separately on car-like and non-car geometry.

An 89% direction rate means nothing until it is compared with the majority answer
inside that same subset -- if the car-like pairs happen to be 89% one direction,
answering constantly would match it. This reports the baseline, the binomial test
against it, and whether the model still calls the minority direction and is right
when it does.
"""

import argparse
import json
import warnings
from math import comb
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv


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

cache = {}


def car_like(run):
    if run not in cache:
        mesh = pv.read(args.root / f"run_{run}" / f"boundary_{run}.vtp").extract_surface()
        span = np.sort(np.diff(np.asarray(mesh.bounds).reshape(3, 2), axis=1).ravel())[::-1]
        cache[run] = bool(1.2 <= span[2] <= 1.8 and 1.7 <= span[1] <= 2.3)
    return cache[run]


groups = {"차량형": {"pred": [], "true": []}, "비차량형": {"pred": [], "true": []}}
for fold in sorted(args.cv_dir.glob("fold*")):
    rows = load(fold / "eval-test.jsonl")
    for pair in json.loads((fold / "pairs-test.json").read_text())["pairs"]:
        base, var = pair["baseline"], pair["variant"]
        if f"run_{base}" not in rows or f"run_{var}" not in rows:
            continue
        key = "차량형" if car_like(base) else "비차량형"
        groups[key]["pred"].append(rows[f"run_{var}"]["pred_cd"] - rows[f"run_{base}"]["pred_cd"])
        groups[key]["true"].append(pair["true_delta_cd"])

for name, data in groups.items():
    pred = np.array(data["pred"])
    true = np.array(data["true"])
    n = len(true)
    if n == 0:
        continue
    down, up = int((true < 0).sum()), int((true > 0).sum())
    majority = max(down, up)
    p_major = majority / n
    hits = int((np.sign(pred) == np.sign(true)).sum())
    p_value = sum(comb(n, j) * p_major ** j * (1 - p_major) ** (n - j)
                  for j in range(hits, n + 1))
    says_up = pred > 0
    up_ok = int(((true > 0) & says_up).sum())
    minority_is_up = up < down
    print(f"\n=== {name} ({n}쌍) ===")
    print(f"  실제 방향: 감소 {down} / 증가 {up}  -> 다수기준 {majority}/{n} = {100*p_major:.0f}%")
    print(f"  모델 방향: {hits}/{n} = {100*hits/n:.0f}%   다수기준 대비 p = {p_value:.5f}")
    print(f"  '증가' 판정 {int(says_up.sum())}개 중 정답 {up_ok}개"
          + (f" (정밀도 {100*up_ok/says_up.sum():.0f}%, 우연 {100*up/n:.0f}%)"
             if says_up.sum() else ""))
    print(f"  값상관 {float(np.corrcoef(pred, true)[0,1]):+.2f}  "
          f"ΔCd MAE {np.abs(pred-true).mean():.4f}")
    if minority_is_up:
        print(f"  → 소수 방향(증가)을 {int(says_up.sum())}번 부르고 {up_ok}번 맞혔다"
              " — 상수 응답이 흉내낼 수 없는 부분")
