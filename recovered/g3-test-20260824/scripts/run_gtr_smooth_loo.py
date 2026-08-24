#!/usr/bin/env python3
"""Run all GT-R smooth leave-one-variant-out folds and aggregate metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    folds = []
    for split_path in sorted(args.splits.glob("fold-*.json")):
        checkpoint = args.out / f"{split_path.stem}.pt"
        metrics_path = Path(str(checkpoint) + ".metrics.json")
        command = [
            sys.executable,
            str(args.root / "scripts" / "finetune_domino_v3.py"),
            "--root", str(args.root / "data" / "domino-gtr-smooth-v1"),
            "--split", str(split_path),
            "--out", str(checkpoint),
            "--base-checkpoint", str(args.base_checkpoint),
            "--epochs", str(args.epochs),
            "--learning-rate", str(args.learning_rate),
            "--train-mode", "decoder",
            "--device", args.device,
        ]
        if metrics_path.exists():
            print(f"resume: reusing {metrics_path}", flush=True)
        else:
            print("+ " + " ".join(command), flush=True)
            subprocess.run(command, cwd=args.root, check=True)
        metrics = json.loads(metrics_path.read_text())
        split = json.loads(split_path.read_text())
        folds.append({
            "fold": split_path.stem,
            "held_out_variant": split["test_cases"][0]["variant_name"],
            "pretrained": metrics["pretrained_test"],
            "finetuned": metrics["finetuned_test"],
        })

    def mean(model: str, metric: str) -> float:
        return sum(fold[model][metric] for fold in folds) / len(folds)

    summary = {
        "fold_count": len(folds),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "base_checkpoint": str(args.base_checkpoint),
        "folds": folds,
        "mean": {
            "pretrained": {
                "cd_mae": mean("pretrained", "cd_mae"),
                "cl_mae": mean("pretrained", "cl_mae"),
            },
            "finetuned": {
                "cd_mae": mean("finetuned", "cd_mae"),
                "cl_mae": mean("finetuned", "cl_mae"),
            },
        },
    }
    summary["improved_both"] = (
        summary["mean"]["finetuned"]["cd_mae"] < summary["mean"]["pretrained"]["cd_mae"]
        and summary["mean"]["finetuned"]["cl_mae"] < summary["mean"]["pretrained"]["cl_mae"]
    )
    output = args.out / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
