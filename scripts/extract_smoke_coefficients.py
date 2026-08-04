#!/usr/bin/env python3
"""Extract coefficient evidence from successful smoke-test S3 artifacts."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import boto3


COEFFICIENT_MARKERS = ("cd", "cl", "drag", "lift", "efficiency")


def relevant_mapping(value, prefix=""):
    found = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(marker in str(key).lower() for marker in COEFFICIENT_MARKERS):
                found[path] = item
            found.update(relevant_mapping(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(relevant_mapping(item, f"{prefix}[{index}]"))
    return found


def list_objects(client, bucket, prefix):
    result = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        result.extend(item["Key"] for item in page.get("Contents", []))
    return result


def get_text(client, bucket, key):
    return client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8", "replace")


def csv_evidence(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return {}
    row = rows[-1]
    return {
        key: value for key, value in row.items()
        if key and any(marker in key.lower() for marker in COEFFICIENT_MARKERS)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv.open(newline="", encoding="utf-8-sig")))
    clients, results = {}, []
    for row in rows:
        if row["job_status"] != "succeeded":
            continue
        bucket = row["s3_bucket"]
        client = clients.setdefault(bucket, boto3.client("s3"))
        keys = list_objects(client, bucket, row["s3_output_prefix"])
        evidence = []
        for key in keys:
            name = Path(key).name.lower()
            try:
                if name.endswith("_results.json"):
                    data = json.loads(get_text(client, bucket, key))
                    evidence.append({"key": key, "type": "json", "values": relevant_mapping(data)})
                elif name == "history.csv" or name == "rbf_optimization_history.csv":
                    values = csv_evidence(get_text(client, bucket, key))
                    if values:
                        evidence.append({"key": key, "type": "csv-last-row", "values": values})
            except Exception as exc:
                evidence.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
        result = {
            "test_case": row["test_case"], "solver": row["solver"],
            "job_id": row["job_id"], "job_uid": row["job_uid"],
            "report_cd_final": row["cd_final"] or None,
            "report_cl_final": row["cl_final"] or None,
            "evidence": evidence,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
    args.out.write_text(json.dumps({"runs": results}, indent=2))


if __name__ == "__main__":
    main()
