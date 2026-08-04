#!/usr/bin/env python3
"""Run the isolated G3-v2 collect, quality-gate, and optional training cycle."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(command: list[str], root: Path, *, stdout: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    if stdout:
        stdout.parent.mkdir(parents=True, exist_ok=True)
        with stdout.open("w", encoding="utf-8") as handle:
            subprocess.run(command, cwd=root, check=True, stdout=handle)
    else:
        subprocess.run(command, cwd=root, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def newest_collector_report(root: Path) -> dict:
    reports = sorted((root / "var" / "collector" / "runs").glob("*/report.json"))
    return read_json(reports[-1]) if reports else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--train", action="store_true",
                        help="train an isolated challenger after quality gates")
    parser.add_argument("--minimum-new-cases", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--expert-epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.root.resolve()
    python = sys.executable
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "var" / "nightly-v2" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    lock_path = root / "var" / "nightly-v2" / "nightly.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "training_requested": args.train,
    }
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

            if not args.skip_collection:
                run([python, "scripts/collect_s3_training_data.py", "--root", str(root)], root)
            collector = newest_collector_report(root)
            added = int(collector.get("added_cases", 0))
            report["collector"] = collector

            manifests = {
                "g2": root / "data/g2-v2/manifest.json",
                "g1_drag": root / "data/g1-drag-v2/manifest.json",
                "g1_lift": root / "data/g1-lift-v2/manifest.json",
                "g4": root / "data/g4-v2/manifest.json",
            }
            quality = {
                key: path.with_name("manifest.quality.json")
                for key, path in manifests.items()
            }

            run([python, "scripts/analyze_manifest_distributions.py",
                 *(str(path) for path in manifests.values())], root,
                stdout=run_dir / "distributions.json")
            run([python, "scripts/build_quality_gated_manifest.py",
                 "--manifest", str(manifests["g2"]),
                 "--out", str(quality["g2"]),
                 "--min-surface-points", "5000",
                 "--expert", "g2_su2_clean"], root)
            for key, coefficient in (
                ("g1_drag", "cd"), ("g1_lift", "cl"), ("g4", "cd")
            ):
                run([python, "scripts/filter_coefficient_manifest.py",
                     "--manifest", str(manifests[key]),
                     "--out", str(quality[key]),
                     "--coefficient", coefficient,
                     "--min-points", "4000"], root)

            quality_summary = {}
            for key, path in quality.items():
                payload = read_json(path)
                quality_summary[key] = payload.get("summary", {
                    "cases": len(payload.get("cases", []))
                })
            report["quality"] = quality_summary
            report["new_cases"] = added

            if not args.train:
                report["status"] = "quality-complete-training-disabled"
            elif added < args.minimum_new_cases:
                report["status"] = "quality-complete-insufficient-new-cases"
            else:
                field = run_dir / "g3-v2-field.pt"
                challenger = run_dir / "g3-v2-challenger.pt"
                run([python, "-m", "aox_g3.train_fields",
                     "--manifest", str(quality["g2"]),
                     "--g1-manifest", str(quality["g1_drag"]),
                     "--g1-cl-manifest", str(quality["g1_lift"]),
                     "--out", str(field),
                     "--epochs", str(args.epochs),
                     "--steps-per-epoch", str(args.steps_per_epoch),
                     "--group-balanced-sampling", "--device", args.device], root)
                run([python, "-m", "aox_g3.train_coefficient_experts",
                     "--base", str(field), "--out", str(challenger),
                     "--expert", "g1_openfoam", str(quality["g1_drag"]),
                     "--expert", "g1_openfoam", str(quality["g1_lift"]),
                     "--expert", "g4_lbm", str(quality["g4"]),
                     "--epochs", str(args.expert_epochs), "--device", args.device], root)
                report.update({
                    "status": "challenger-trained-not-promoted",
                    "field_checkpoint": str(field),
                    "challenger_checkpoint": str(challenger),
                })
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(run_dir / "report.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
