#!/usr/bin/env python3
"""Collect new AOX training events, materialize v2 NPZs, and update manifests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EVENT_BUCKETS = ("aoxlabs-prod-static", "aoxlabs-stage-static")
DATASETS = {
    "g2": "g2-v2",
    "g1_drag": "g1-drag-v2",
    "g1_lift": "g1-lift-v2",
    "g4": "g4-v2",
}


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def known_job_uids(data_root: Path) -> set[str]:
    result: set[str] = set()
    for name in DATASETS.values():
        manifest = data_root / name / "manifest.json"
        if not manifest.exists():
            continue
        for row in read_json(manifest).get("cases", []):
            source = row.get("smoke_source") or {}
            job_uid = source.get("job_uid")
            if job_uid:
                result.add(str(job_uid))
            # G1 case IDs are their source job UID.
            if name.startswith("g1-") and row.get("case_id"):
                result.add(str(row["case_id"]))
    return result


def merge_csv(inputs: list[Path], output: Path, excluded_jobs: set[str]) -> int:
    rows, fieldnames, seen = [], None, set()
    for path in inputs:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = fieldnames or reader.fieldnames
            for row in reader:
                job_uid = str(row.get("job_uid") or "")
                if not job_uid or job_uid in excluded_jobs or job_uid in seen:
                    continue
                seen.add(job_uid)
                rows.append(row)
    if not fieldnames:
        raise ValueError("event CSV files have no header")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def append_cases(source_manifest: Path, target_dir: Path) -> tuple[int, int]:
    source = read_json(source_manifest)
    target_manifest = target_dir / "manifest.json"
    target = read_json(target_manifest)
    known = {str(row["case_id"]) for row in target.get("cases", [])}
    added = duplicates = 0
    for row in source.get("cases", []):
        case_id = str(row["case_id"])
        if case_id in known:
            duplicates += 1
            continue
        source_npz = Path(row["npz"])
        if not source_npz.is_absolute():
            source_npz = source_manifest.parent / source_npz
        destination = target_dir / "cases" / f"{case_id}.npz"
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copy2(source_npz, destination)
        merged = dict(row)
        merged["npz"] = f"cases/{destination.name}"
        target.setdefault("cases", []).append(merged)
        known.add(case_id)
        added += 1
    target["geometry_preprocessing_version"] = 2
    write_json_atomic(target_manifest, target)
    return added, duplicates


def open_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS collection_runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            discovered_jobs INTEGER NOT NULL DEFAULT 0,
            selected_jobs INTEGER NOT NULL DEFAULT 0,
            added_cases INTEGER NOT NULL DEFAULT 0,
            report_json TEXT
        )
    """)
    connection.commit()
    return connection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--event-bucket", action="append", default=[])
    parser.add_argument("--event-prefix", default="_private/g3/training-events")
    parser.add_argument("--state", type=Path, default=Path("var/collector/state.sqlite3"))
    args = parser.parse_args()

    root = args.root.resolve()
    data_root = root / "data"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "var" / "collector" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    state_path = args.state if args.state.is_absolute() else root / args.state
    state = open_state(state_path)
    state.execute(
        "INSERT INTO collection_runs(run_id, started_at, status) VALUES (?, ?, ?)",
        (run_id, datetime.now(timezone.utc).isoformat(), "running"),
    )
    state.commit()

    report: dict[str, object] = {"run_id": run_id, "datasets": {}}
    try:
        buckets = args.event_bucket or list(EVENT_BUCKETS)
        event_csvs = []
        discovered = 0
        for bucket in buckets:
            safe = bucket.replace("/", "_")
            inventory = run_dir / f"{safe}.events.json"
            event_csv = run_dir / f"{safe}.events.csv"
            run([
                sys.executable, "scripts/sync_training_events.py",
                "--bucket", bucket, "--prefix", args.event_prefix,
                "--out-json", str(inventory), "--out-csv", str(event_csv),
            ], root)
            discovered += len(read_json(inventory).get("cases", []))
            event_csvs.append(event_csv)

        new_events = run_dir / "new-events.csv"
        new_count = merge_csv(event_csvs, new_events, known_job_uids(data_root))
        report.update({"discovered_jobs": discovered, "new_event_jobs": new_count})
        if new_count == 0:
            report["status"] = "no-new-jobs"
        else:
            audit = run_dir / "audit.json"
            selected = run_dir / "selected.csv"
            run([sys.executable, "scripts/audit_training_inventory.py",
                 "--csv", str(new_events), "--out", str(audit)], root)
            run([sys.executable, "scripts/select_training_inventory.py",
                 "--csv", str(new_events), "--audit", str(audit),
                 "--out", str(selected)], root)

            staging = run_dir / "staging"
            jobs = [
                ("g1_drag", [sys.executable, "scripts/prepare_g1_surfaces.py",
                 "--csv", str(selected), "--out-dir", str(staging / "g1-drag"),
                 "--objective-function", "drag", "--solver", "G1"]),
                ("g1_lift", [sys.executable, "scripts/prepare_g1_surfaces.py",
                 "--csv", str(selected), "--out-dir", str(staging / "g1-lift"),
                 "--objective-function", "lift", "--solver", "G1"]),
                ("g4", [sys.executable, "scripts/prepare_smoke_g4_geometry.py",
                 "--csv", str(selected), "--out-dir", str(staging / "g4")]),
                ("g2", [sys.executable, "scripts/prepare_smoke_g2_s3.py",
                 "--csv", str(selected), "--out-dir", str(staging / "g2" / "cases"),
                 "--manifest", str(staging / "g2" / "manifest.json")]),
            ]
            added_total = 0
            for key, command in jobs:
                run(command, root)
                manifest = staging / ("g2" if key == "g2" else key.replace("_", "-")) / "manifest.json"
                added, duplicates = append_cases(manifest, data_root / DATASETS[key])
                source = read_json(manifest)
                report["datasets"][key] = {
                    "materialized": len(source.get("cases", [])),
                    "skipped": len(source.get("skipped", [])),
                    "added": added,
                    "duplicates": duplicates,
                }
                added_total += added
            report.update({"status": "complete", "added_cases": added_total})

        finished = datetime.now(timezone.utc).isoformat()
        state.execute("""
            UPDATE collection_runs SET finished_at=?, status=?, discovered_jobs=?,
            selected_jobs=?, added_cases=?, report_json=? WHERE run_id=?
        """, (finished, report["status"], int(report.get("discovered_jobs", 0)),
                int(report.get("new_event_jobs", 0)), int(report.get("added_cases", 0)),
                json.dumps(report), run_id))
        state.commit()
        write_json_atomic(run_dir / "report.json", report)
        print(json.dumps(report, indent=2))
    except Exception as exc:
        state.execute(
            "UPDATE collection_runs SET finished_at=?, status=?, report_json=? WHERE run_id=?",
            (datetime.now(timezone.utc).isoformat(), "failed",
             json.dumps({"error": f"{type(exc).__name__}: {exc}"}), run_id),
        )
        state.commit()
        raise
    finally:
        state.close()


if __name__ == "__main__":
    main()
