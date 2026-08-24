#!/usr/bin/env python3
"""Collect quality-gated G2 cases and optionally train an isolated challenger."""

from __future__ import annotations
import argparse, fcntl, json, math, os, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

BUCKETS = ("aoxlabs-prod-static", "aoxlabs-stage-static")

def atomic(path, value):
    temp = path.with_name(f".{path.name}.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/home/ubuntu/g3-v2"))
    p.add_argument("--base", type=Path, default=Path("data/su2_labels_v3"))
    p.add_argument("--out", type=Path, default=Path("data/domino-g2-auto"))
    p.add_argument("--minimum-free-gb", type=float, default=10)
    p.add_argument("--train", action="store_true")
    p.add_argument("--min-new-cases", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args(); root = a.root.resolve()
    runtime = root / "var/domino-collector"; runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / "collection.lock").open("w") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: print("DoMINO collection already running; skipped"); return 0
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = runtime / "runs" / run_id; run_dir.mkdir(parents=True)
        report = {"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
                  "training_requested": a.train, "training_started": False}
        if shutil.disk_usage(root).free < a.minimum_free_gb * 1024**3:
            report["status"] = "skipped-low-disk"; atomic(run_dir / "report.json", report); return 0
        inventories = []
        try:
            for bucket in BUCKETS:
                target = run_dir / f"{bucket}.json"
                subprocess.run([sys.executable, "scripts/sync_training_events.py", "--bucket", bucket,
                                "--out-json", str(target)], cwd=root, check=True)
                inventories += [target]
            command = [sys.executable, "scripts/build_domino_s3_v4.py", "--base", str(root/a.base),
                       "--out", str(root/a.out)]
            for inventory in inventories: command += ["--event-inventory", str(inventory)]
            with (run_dir / "collection.log").open("w") as log:
                done = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            report["status"] = "complete" if done.returncode == 0 else "failed"
            report["exit_code"] = done.returncode
            manifest_path = root / a.out / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if done.returncode == 0 else {}
            summary = manifest.get("summary") or {}
            new_cases = int(summary.get("accepted_this_run") or 0)
            report["collection"] = summary
            report["new_accepted_cases"] = new_cases
            if a.train and done.returncode == 0 and new_cases >= a.min_new_cases:
                split_path = run_dir / "split.json"
                checkpoint = run_dir / "challenger.pt"
                split_command = [sys.executable, "scripts/split_domino_v3_groups.py",
                                 "--manifest", str(manifest_path), "--out", str(split_path)]
                with (run_dir / "split.log").open("w") as log:
                    subprocess.run(split_command, cwd=root, check=True,
                                   stdout=log, stderr=subprocess.STDOUT)
                train_command = [sys.executable, "scripts/finetune_domino_v3.py",
                                 "--root", str(root / a.out), "--split", str(split_path),
                                 "--out", str(checkpoint), "--epochs", str(a.epochs),
                                 "--train-mode", "decoder", "--device", a.device]
                report["training_started"] = True
                with (run_dir / "training.log").open("w") as log:
                    trained = subprocess.run(train_command, cwd=root, stdout=log,
                                             stderr=subprocess.STDOUT)
                report["training_exit_code"] = trained.returncode
                if trained.returncode != 0:
                    report["status"] = "training-failed"
                else:
                    metrics = json.loads(Path(str(checkpoint) + ".metrics.json").read_text())
                    before, after = metrics["pretrained_test"], metrics["finetuned_test"]
                    metric_values = [before["cd_mae"], before["cl_mae"],
                                     after["cd_mae"], after["cl_mae"]]
                    if not all(math.isfinite(float(value)) for value in metric_values):
                        raise ValueError("training produced non-finite validation metrics")
                    promotable = (after["cd_mae"] <= before["cd_mae"] and
                                  after["cl_mae"] <= before["cl_mae"])
                    report.update(status="challenger-trained", challenger=str(checkpoint),
                                  metrics=metrics, promotable=promotable,
                                  promotion_performed=False)
            elif a.train:
                report["training_skip_reason"] = (
                    f"new accepted cases {new_cases} < minimum {a.min_new_cases}"
                )
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic(run_dir / "report.json", report); atomic(runtime / "latest.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] in {"complete", "challenger-trained"} else 1

if __name__ == "__main__": raise SystemExit(main())
