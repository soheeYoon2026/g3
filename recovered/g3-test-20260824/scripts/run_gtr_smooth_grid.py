#!/usr/bin/env python3
"""Compare conservative GT-R fine-tuning settings without promoting a model."""

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
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    settings = [(1, 3e-6), (3, 3e-6), (3, 1e-6)]
    results = []
    for epochs, learning_rate in settings:
        name = f"lr{learning_rate:.0e}-epoch{epochs}"
        output = args.out / name
        command = [
            sys.executable, str(args.root / "scripts" / "run_gtr_smooth_loo.py"),
            "--root", str(args.root), "--splits", str(args.splits),
            "--base-checkpoint", str(args.base_checkpoint), "--out", str(output),
            "--epochs", str(epochs), "--learning-rate", str(learning_rate),
            "--device", args.device,
        ]
        subprocess.run(command, cwd=args.root, check=True)
        summary = json.loads((output / "summary.json").read_text())
        results.append({"name": name, "path": str(output), **summary["mean"],
                        "improved_both": summary["improved_both"]})

    eligible = [row for row in results if row["improved_both"]]
    selected = min(
        eligible,
        key=lambda row: row["finetuned"]["cd_mae"] + row["finetuned"]["cl_mae"],
        default=None,
    )
    report = {"results": results, "selected": selected, "promoted": False}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "grid-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
