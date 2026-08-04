#!/usr/bin/env python3
"""Audit successful solver smoke-test artifacts recorded in the final report.

The script is intentionally read-only.  It lists S3 objects below every
successful output prefix and writes a compact JSON inventory that can be used
to decide which runs contain fields and trustworthy coefficient labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import boto3


def list_objects(client, bucket: str, prefix: str) -> list[dict]:
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            objects.append({"key": item["Key"], "size": int(item["Size"])})
    return objects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    clients: dict[str, object] = {}
    runs = []
    for row in rows:
        if row.get("job_status") != "succeeded":
            continue
        bucket = row["s3_bucket"]
        client = clients.setdefault(bucket, boto3.client("s3"))
        objects = list_objects(client, bucket, row["s3_output_prefix"])
        suffixes = Counter(Path(item["key"]).suffix.lower() or "<none>" for item in objects)
        run = {
            "test_case": row["test_case"],
            "solver": row["solver"],
            "job_id": row["job_id"],
            "job_uid": row["job_uid"],
            "bucket": bucket,
            "input_prefix": row["s3_input_prefix"],
            "output_prefix": row["s3_output_prefix"],
            "cd_final": float(row["cd_final"]) if row["cd_final"].strip() else None,
            "cl_final": float(row["cl_final"]) if row["cl_final"].strip() else None,
            "object_count": len(objects),
            "total_bytes": sum(item["size"] for item in objects),
            "suffixes": dict(sorted(suffixes.items())),
            "objects": objects,
        }
        runs.append(run)
        interesting = [
            Path(item["key"]).name for item in objects
            if Path(item["key"]).suffix.lower() in {
                ".json", ".csv", ".dat", ".npz", ".vtu", ".vtp", ".vti", ".stl"
            }
        ]
        print(
            f"{run['test_case']} {run['solver']} job={run['job_id']} "
            f"objects={len(objects)} bytes={run['total_bytes']} "
            f"suffixes={run['suffixes']}"
        )
        if interesting:
            print("  " + ", ".join(interesting[:30]))

    payload = {"source_csv": str(args.csv), "successful_runs": len(runs), "runs": runs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
