#!/usr/bin/env python
"""Evaluate the PointNet-like G3 checkpoint on a DoMINO v3 group split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aox_g3.eval_metrics import coefficient_summary, load_pairs
from aox_g3.geometry.stl_sampler import load_mesh
from aox_g3.infer_fields import main as infer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--expert", default="g2_su2_clean")
    parser.add_argument("--pairs", type=Path, help="explicit baseline/variant pair manifest for ΔCd metrics")
    parser.add_argument("--direction-tolerance", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    pairs = load_pairs(args.pairs) if args.pairs else None
    split = json.loads(args.split.read_text())
    rows = []
    args.work_dir.mkdir(parents=True, exist_ok=False)
    for case in split["test_cases"]:
        run = str(case["run"])
        run_dir = args.root / f"run_{run}"
        conditions = json.loads((run_dir / f"conditions_{run}.json").read_text())
        stl = run_dir / f"drivaer_{run}.stl"
        mesh = load_mesh(str(stl))
        lo, hi = np.asarray(mesh.bounds, dtype=float)
        ref_length = float(np.max(hi - lo))
        output = infer([
            "--stl", str(stl), "--model", str(args.checkpoint),
            "--out-dir", str(args.work_dir / f"run_{run}"),
            "--u-x", str(conditions["speed"]), "--u-y", "0", "--u-z", "0",
            "--density", str(conditions["density"]),
            "--ref-length", str(ref_length), "--ref-area", str(conditions["ref_area"]),
            "--coefficients-only", "--coefficient-expert", args.expert,
            "--device", args.device,
        ])
        row = {
            "run": int(run), "group_id": case["group_id"],
            "true_cd": conditions["su2_cd"], "pred_cd": output["drag_coefficient"],
            "true_cl": conditions["su2_cl"], "pred_cl": output["lift_coefficient"],
            "elapsed_seconds": output["elapsed_seconds"],
        }
        rows.append(row)

    report = {"checkpoint": str(args.checkpoint)}
    report.update(coefficient_summary(
        rows, pairs=pairs, case_key="run",
        direction_tolerance=args.direction_tolerance,
    ))
    report["mean_inference_seconds"] = float(np.mean([r["elapsed_seconds"] for r in rows]))
    report["rows"] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
