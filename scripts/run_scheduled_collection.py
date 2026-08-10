#!/usr/bin/env python3
"""Safely run the G3-v2 collection-only nightly pipeline."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/ubuntu/g3-v2"))
    parser.add_argument("--python", default="/home/ubuntu/venv_g3/bin/python")
    parser.add_argument("--minimum-free-gb", type=float, default=10.0)
    args = parser.parse_args()

    root = args.root.resolve()
    runtime = root / "var" / "scheduled-collection"
    runtime.mkdir(parents=True, exist_ok=True)
    lock_path = runtime / "collection.lock"

    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("collection already running; skipped")
            return 0

        started = datetime.now(timezone.utc)
        run_id = started.strftime("%Y%m%dT%H%M%SZ")
        run_dir = runtime / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        log_path = run_dir / "collection.log"
        report_path = run_dir / "report.json"
        free_before = shutil.disk_usage(root).free
        minimum = int(args.minimum_free_gb * 1024**3)
        report = {
            "run_id": run_id,
            "started_at": started.isoformat(),
            "root": str(root),
            "minimum_free_gb": args.minimum_free_gb,
            "free_gb_before": round(free_before / 1024**3, 3),
            "training_requested": False,
        }

        if free_before < minimum:
            report.update({"status": "skipped-low-disk", "finished_at": datetime.now(timezone.utc).isoformat()})
            write_json_atomic(report_path, report)
            print(json.dumps(report, indent=2))
            return 0

        command = [
            args.python,
            "scripts/run_g3_v2_nightly.py",
            "--root",
            str(root),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        free_after = shutil.disk_usage(root).free
        report.update({
            "status": "complete" if completed.returncode == 0 else "failed",
            "exit_code": completed.returncode,
            "free_gb_after": round(free_after / 1024**3, 3),
            "log": str(log_path),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        write_json_atomic(report_path, report)
        latest = runtime / "latest.json"
        write_json_atomic(latest, report)
        print(json.dumps(report, indent=2))
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
