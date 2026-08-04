#!/usr/bin/env python3
"""Audit whether a multi-solver job inventory has materializable S3 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import boto3

from prepare_smoke_g2_s3 import case_files


def list_objects(client, bucket: str, prefix: str) -> list[dict]:
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        objects.extend(page.get("Contents", []))
    return objects


def classify(row: dict[str, str], objects: list[dict]) -> dict:
    prefix = (row.get("output_s3_key") or row.get("s3_output_prefix") or "").rstrip("/") + "/"
    keys = [item["Key"] for item in objects]
    solver = row.get("solver")
    result = {
        "job_uid": row.get("job_uid"),
        "project_uid": row.get("project_uid"),
        "solver": solver,
        "bucket": row.get("s3_bucket"),
        "prefix": prefix,
        "objects": len(objects),
        "bytes": sum(int(item.get("Size", 0)) for item in objects),
    }
    if solver == "G1":
        lower = [key.lower() for key in keys]
        result.update({
            "vtp_manifest": any(key.endswith("/vtp/manifest.json") for key in lower),
            "vtp_files": sum(key.endswith(".vtp") for key in lower),
        })
        result["materializable"] = result["vtp_files"] >= 3
    elif solver == "G2":
        available = case_files(prefix, keys)
        result.update({
            "field_cases": len(available),
            "field_case_names": sorted(available),
        })
        result["materializable"] = bool(available)
    elif solver == "G4":
        lower = [key.lower() for key in keys]
        result.update({
            "optimized_stl": sum(key.endswith("_optimized.stl") for key in lower),
            "results_json": sum(key.endswith("_results.json") for key in lower),
        })
        result["materializable"] = bool(result["optimized_stl"] and result["results_json"])
    else:
        result["materializable"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv.open(newline="", encoding="utf-8-sig")))
    eligible = [
        row for row in rows
        if row.get("solver") in {"G1", "G2", "G4"}
        and (row.get("job_status") or row.get("status")) == "succeeded"
        and row.get("s3_bucket")
        and (row.get("output_s3_key") or row.get("s3_output_prefix"))
    ]
    clients, audited = {}, []
    for number, row in enumerate(eligible, 1):
        bucket = row["s3_bucket"]
        client = clients.setdefault(bucket, boto3.client("s3"))
        prefix = row.get("output_s3_key") or row.get("s3_output_prefix")
        try:
            result = classify(row, list_objects(client, bucket, prefix))
        except Exception as exc:
            result = {
                "job_uid": row.get("job_uid"),
                "project_uid": row.get("project_uid"),
                "solver": row.get("solver"),
                "bucket": bucket,
                "prefix": prefix,
                "materializable": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        audited.append(result)
        print(
            f"[{number}/{len(eligible)}] {result['solver']} {result['job_uid']}: "
            f"{'OK' if result['materializable'] else 'SKIP'} "
            f"objects={result.get('objects', 0)} bytes={result.get('bytes', 0)}"
        )

    by_solver = {}
    for solver in ("G1", "G2", "G4"):
        selected = [row for row in audited if row["solver"] == solver]
        by_solver[solver] = {
            "eligible_jobs": len(selected),
            "materializable_jobs": sum(bool(row["materializable"]) for row in selected),
            "materializable_bytes": sum(
                int(row.get("bytes", 0)) for row in selected if row["materializable"]
            ),
            "projects": len({
                row.get("project_uid") for row in selected
                if row["materializable"] and row.get("project_uid")
            }),
        }
        if solver == "G2":
            by_solver[solver]["field_cases"] = sum(
                int(row.get("field_cases", 0)) for row in selected
            )
    payload = {
        "format": "g3-training-inventory-audit-v1",
        "source_csv": str(args.csv),
        "input_rows": len(rows),
        "eligible_rows": len(eligible),
        "status_counts": dict(Counter(
            row.get("job_status") or row.get("status") or "<blank>" for row in rows
        )),
        "summary": by_solver,
        "jobs": audited,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
