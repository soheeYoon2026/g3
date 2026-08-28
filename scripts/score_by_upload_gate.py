"""Re-score the shape split using the production upload gate, not a threshold
chosen after seeing the result.

aox_g3/upload_gate.py was written earlier for a different purpose, so its verdict
is independent of this analysis. If it separates the pairs the same way, the
89%/45% split is a property of the data; if it does not, the split is partly my
own threshold and has to be reported as such.
"""

import argparse
import json
import sys
import warnings
from math import comb
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv
import trimesh


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
ap.add_argument("--gate-module", type=Path, required=True, help="dir containing aox_g3/")
args = ap.parse_args()

sys.path.insert(0, str(args.gate_module))
from aox_g3.upload_gate import classify_mesh

cache = {}


def verdict(run):
    if run not in cache:
        surface = pv.read(args.root / f"run_{run}" / f"boundary_{run}.vtp") \
            .extract_surface().triangulate()
        mesh = trimesh.Trimesh(  # the gate is written against trimesh, not pyvista
            vertices=np.asarray(surface.points, dtype=np.float64),
            faces=np.asarray(surface.faces).reshape(-1, 4)[:, 1:], process=False)
        try:
            result = classify_mesh(mesh)
        except Exception as exc:
            cache[run] = ("오류", str(exc)[:60])
            return cache[run]
        label = result.get("verdict", "unknown")
        cache[run] = (label, "; ".join(result.get("reasons", []))[:70])
    return cache[run]


groups = {}
for fold in sorted(args.cv_dir.glob("fold*")):
    rows = load(fold / "eval-test.jsonl")
    for pair in json.loads((fold / "pairs-test.json").read_text())["pairs"]:
        base, var = pair["baseline"], pair["variant"]
        if f"run_{base}" not in rows or f"run_{var}" not in rows:
            continue
        label, reason = verdict(base)
        groups.setdefault(label, {"pred": [], "true": [], "reason": reason})
        groups[label]["pred"].append(rows[f"run_{var}"]["pred_cd"] - rows[f"run_{base}"]["pred_cd"])
        groups[label]["true"].append(pair["true_delta_cd"])

for label, data in sorted(groups.items(), key=lambda kv: -len(kv[1]["true"])):
    pred, true = np.array(data["pred"]), np.array(data["true"])
    n = len(true)
    down, up = int((true < 0).sum()), int((true > 0).sum())
    majority = max(down, up)
    hits = int((np.sign(pred) == np.sign(true)).sum())
    p_value = sum(comb(n, j) * (majority / n) ** j * (1 - majority / n) ** (n - j)
                  for j in range(hits, n + 1))
    says_up = pred > 0
    up_ok = int(((true > 0) & says_up).sum())
    print(f"\n=== {label} ({n}쌍) ===")
    if data["reason"]:
        print(f"  판정 사유 예: {data['reason']}")
    print(f"  실제: 감소 {down} / 증가 {up}  -> 다수기준 {100*majority/n:.0f}%")
    print(f"  모델: {hits}/{n} = {100*hits/n:.0f}%  p = {p_value:.5f}  "
          f"값상관 {float(np.corrcoef(pred, true)[0,1]):+.2f}")
    print(f"  '증가' {int(says_up.sum())}개 중 정답 {up_ok}개")
