#!/usr/bin/env python3
"""Select materializable, labeled jobs for a manual G3 training run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def timestamp(row: dict[str, str]) -> str:
    return row.get("finished_at") or row.get("created_at") or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError("CSV has no header")

    audit = json.loads(args.audit.read_text())
    materializable = {
        row["job_uid"] for row in audit["jobs"] if row.get("materializable")
    }
    eligible = []
    skipped = []
    for row in rows:
        solver = row.get("solver")
        status = row.get("job_status") or row.get("status")
        prefix = row.get("output_s3_key") or row.get("s3_output_prefix")
        reason = None
        if solver not in {"G1", "G2", "G4"}:
            reason = "unsupported_solver"
        elif status != "succeeded":
            reason = "not_succeeded"
        elif not prefix:
            reason = "missing_output_prefix"
        elif row.get("job_uid") not in materializable:
            reason = "missing_required_artifacts"
        elif solver == "G1" and (
            row.get("objective_function", "").lower() not in {"drag", "lift"}
            or not row.get("cd_final", "").strip()
        ):
            reason = "missing_g1_coefficient_label"
        if reason:
            skipped.append({"job_uid": row.get("job_uid"), "reason": reason})
        else:
            eligible.append(row)

    eligible.sort(key=timestamp)
    selected = []
    latest = {}
    for row in eligible:
        solver = row["solver"]
        if solver == "G1":
            key = (solver, row.get("project_uid"), row["objective_function"].lower())
            latest[key] = row
        elif solver == "G4":
            key = (solver, row.get("project_uid"))
            latest[key] = row
        else:
            selected.append(row)
    selected.extend(latest.values())
    selected.sort(key=timestamp)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    counts = {
        solver: sum(row["solver"] == solver for row in selected)
        for solver in ("G1", "G2", "G4")
    }
    report = {
        "input_rows": len(rows),
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "selected_by_solver": counts,
        "skipped_rows": len(skipped),
        "policy": {
            "G1": "latest successful labeled job per project and objective",
            "G2": "all successful jobs; materializer removes exact field duplicates",
            "G4": "latest successful job per project",
        },
    }
    args.out.with_suffix(".selection.json").write_text(
        json.dumps({**report, "skipped": skipped}, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
