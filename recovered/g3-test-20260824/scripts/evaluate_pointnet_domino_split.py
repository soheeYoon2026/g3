#!/usr/bin/env python
"""Evaluate the PointNet-like G3 checkpoint on a DoMINO v3 group split."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from aox_g3.geometry.stl_sampler import load_mesh
from aox_g3.infer_fields import main as infer


def rank(values):
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    result[order] = np.arange(len(values), dtype=float)
    return result


def correlation(actual, predicted):
    if len(actual) < 2:
        return float("nan")
    return float(np.corrcoef(rank(np.asarray(actual)), rank(np.asarray(predicted)))[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--expert", default="g2_su2_clean")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

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

    true_cd, pred_cd = [r["true_cd"] for r in rows], [r["pred_cd"] for r in rows]
    true_cl, pred_cl = [r["true_cl"] for r in rows], [r["pred_cl"] for r in rows]
    report = {
        "checkpoint": str(args.checkpoint), "cases": len(rows),
        "cd_mae": float(np.mean(np.abs(np.asarray(true_cd) - pred_cd))),
        "cl_mae": float(np.mean(np.abs(np.asarray(true_cl) - pred_cl))),
        "cd_spearman": correlation(true_cd, pred_cd),
        "cl_spearman": correlation(true_cl, pred_cl),
        "mean_inference_seconds": float(np.mean([r["elapsed_seconds"] for r in rows])),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
