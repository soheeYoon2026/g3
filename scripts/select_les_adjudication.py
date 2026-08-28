"""Pick the pairs worth spending 4-level LES on.

The open question is whether the 69% direction ceiling is the model's limit or the
label's. LES adjudicates, but only if the selection can tell those apart: taking
only disagreements would confound "G2 is wrong there" with "LES is noisy", so this
also picks agreement pairs as controls. If LES sides with G2 on the controls and
with the model on the disagreements, the labels are the ceiling.

Pairs are filtered to what LES can actually resolve. Measured run-to-run scatter
at 4-level is cd_std 0.0003-0.0007, so |ΔCd| below about 0.005 cannot be called.
"""

import argparse
import json
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
ap.add_argument("--run-index", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--min-delta", type=float, default=0.005,
                help="|ΔCd| LES can resolve at 4-level")
ap.add_argument("--disagree", type=int, default=8)
ap.add_argument("--agree", type=int, default=4)
args = ap.parse_args()

index = json.loads(args.run_index.read_text())

records = []
for fold in sorted(args.cv_dir.glob("fold*")):
    rows = load(fold / "eval-test.jsonl")
    for pair in json.loads((fold / "pairs-test.json").read_text())["pairs"]:
        base, var = pair["baseline"], pair["variant"]
        if f"run_{base}" not in rows or f"run_{var}" not in rows:
            continue
        pred = rows[f"run_{var}"]["pred_cd"] - rows[f"run_{base}"]["pred_cd"]
        true = pair["true_delta_cd"]
        meta_b = index[str(base)]
        meta_v = index[str(var)]
        records.append({
            "baseline_run": base, "variant_run": var,
            "job_uid": meta_v["job_uid"],
            "base_design": meta_b["design"], "variant_design": meta_v["design"],
            "g2_delta_cd": true, "model_delta_cd": float(pred),
            "agree": bool(np.sign(pred) == np.sign(true)),
            "resolvable": abs(true) >= args.min_delta,
        })

resolvable = [r for r in records if r["resolvable"]]
print(f"채점 쌍 {len(records)}개 중 LES가 판정 가능한(|ΔCd| >= {args.min_delta}) "
      f"쌍 {len(resolvable)}개")
disagree = [r for r in resolvable if not r["agree"]]
agree = [r for r in resolvable if r["agree"]]
print(f"  불일치 {len(disagree)}개 / 일치 {len(agree)}개")

# strongest evidence first: the model must be confident for its call to mean anything
disagree.sort(key=lambda r: -min(abs(r["model_delta_cd"]), abs(r["g2_delta_cd"])))
agree.sort(key=lambda r: -min(abs(r["model_delta_cd"]), abs(r["g2_delta_cd"])))
picked = disagree[:args.disagree] + agree[:args.agree]

print(f"\n선정 {len(picked)}개")
print(f"{'job':>10s} {'설계':>22s} {'G2 ΔCd':>9s} {'모델 ΔCd':>10s} {'':>6s}")
print("-" * 62)
for r in picked:
    print(f"{r['job_uid'][:8]:>10s} "
          f"{r['base_design']+'->'+r['variant_design']:>22s} "
          f"{r['g2_delta_cd']:+9.4f} {r['model_delta_cd']:+10.4f} "
          f"{'불일치' if not r['agree'] else '일치':>6s}")

runs = sorted({r for p in picked for r in (p["baseline_run"], p["variant_run"])})
print(f"\nLES 필요 형상 {len(runs)}개 (쌍당 2개 미만 — 기준 형상 공유)")
print(f"4단계 4시간 기준 약 {4*len(runs)} GPU-시간")

args.out.write_text(json.dumps({
    "purpose": "adjudicate G2 vs model on ΔCd direction with 4-level LES",
    "min_resolvable_delta_cd": args.min_delta,
    "pairs": picked,
    "runs": runs,
}, indent=1) + "\n")
print("wrote", args.out)
