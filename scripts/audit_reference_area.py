"""Test whether the large gate errors are a reference-area bookkeeping problem.

Cd = force / (q * ref_area). If a case declares a ref_area that does not match its
own geometry, the predicted Cd is wrong by exactly that factor even when the
predicted pressure field is right. This rescales each prediction by
declared_ref_area / measured_frontal_area and reports what happens to the error.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv

ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", type=Path, required=True)
ap.add_argument("--root", type=Path, required=True)
args = ap.parse_args()

rows = []
for line in args.jsonl.read_text().splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if row.get("model") == "fine_tuned" and "case" in row and row.get("true_cd") is not None:
        rows.append(row)

print(f"{'case':>9s} {'true':>7s} {'pred':>7s} {'err%':>5s} "
      f"{'ratio':>6s} {'fixed':>7s} {'err%':>5s}")
before, after = [], []
for row in sorted(rows, key=lambda r: r["case"]):
    run = row["case"].split("_", 1)[1]
    run_dir = args.root / f"run_{run}"
    conditions = json.loads((run_dir / f"conditions_{run}.json").read_text())
    mesh = pv.read(run_dir / f"boundary_{run}.vtp").extract_surface()
    span = np.diff(np.asarray(mesh.bounds).reshape(3, 2), axis=1).ravel()
    order = np.argsort(span)
    frontal = float(span[order[0]] * span[order[1]])
    ratio = float(conditions["ref_area"]) / frontal

    true, pred = row["true_cd"], row["pred_cd"]
    fixed = pred * ratio
    e0 = 100 * abs(pred - true) / abs(true)
    e1 = 100 * abs(fixed - true) / abs(true)
    before.append(e0)
    after.append(e1)
    print(f"{row['case']:>9s} {true:7.4f} {pred:7.4f} {e0:5.0f}% "
          f"{ratio:6.2f} {fixed:7.4f} {e1:5.0f}%")

b, a = np.array(before), np.array(after)
print(f"\n보정 전: 중앙값 {np.median(b):.0f}%  평균 {b.mean():.0f}%  "
      f"±20% 이내 {(b<=20).sum()}/{len(b)}")
print(f"보정 후: 중앙값 {np.median(a):.0f}%  평균 {a.mean():.0f}%  "
      f"±20% 이내 {(a<=20).sum()}/{len(a)}")
