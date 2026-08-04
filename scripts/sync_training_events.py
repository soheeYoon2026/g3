#!/usr/bin/env python3
"""Sync private AOX solver-completion events into a deduplicated G3 inventory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import boto3


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _find_number(value: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(value, dict):
        normalized = {str(key).lower(): child for key, child in value.items()}
        for key in keys:
            if key in normalized:
                direct = _number(normalized[key])
                if direct is not None:
                    return direct
                nested = _find_number(normalized[key], ("final", "best", "value", "result"))
                if nested is not None:
                    return nested
        for child in value.values():
            nested = _find_number(child, keys)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for child in value:
            nested = _find_number(child, keys)
            if nested is not None:
                return nested
    return None


def list_event_keys(client, bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        keys.extend(
            row["Key"] for row in page.get("Contents", [])
            if row["Key"].lower().endswith(".json")
        )
    return sorted(keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket",
        default=os.environ.get("G3_TRAINING_BUCKET") or os.environ.get("AWS_S3_BUCKET"),
    )
    parser.add_argument(
        "--prefix",
        default=os.environ.get("G3_TRAINING_EVENT_PREFIX", "_private/g3/training-events"),
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()
    if not args.bucket:
        parser.error("--bucket or G3_TRAINING_BUCKET is required")

    client = boto3.client("s3")
    events, rejected = {}, []
    for key in list_event_keys(client, args.bucket, args.prefix):
        try:
            event = json.loads(
                client.get_object(Bucket=args.bucket, Key=key)["Body"].read()
            )
            event_id = str(event["event_id"])
            solver = str(event.get("solver") or "").upper()
            if solver not in {"G1", "G2", "G4"}:
                raise ValueError(f"unsupported solver {solver!r}")
            if not event.get("input_artifacts"):
                raise ValueError("missing source STL artifact")
            if not event.get("s3_output_prefix"):
                raise ValueError("missing output prefix")
            if not event.get("s3_output_bucket"):
                raise ValueError("missing output bucket")
            events[event_id] = event
        except Exception as exc:
            rejected.append({"key": key, "reason": f"{type(exc).__name__}: {exc}"})

    cases = [
        {
            "case_id": event_id,
            "group_id": event_id,
            "event": event,
        }
        for event_id, event in sorted(events.items())
    ]
    payload = {
        "schema_version": 1,
        "format": "g3-training-event-inventory-v1",
        "bucket": args.bucket,
        "prefix": args.prefix,
        "cases": cases,
        "rejected": rejected,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")

    fieldnames = [
        "solver",
        "job_status",
        "test_case",
        "job_id",
        "job_uid",
        "project_uid",
        "tenant_uid",
        "team_uid",
        "s3_bucket",
        "s3_output_prefix",
        "output_s3_key",
        "objective_function",
        "cd_initial",
        "cd_final",
        "cl_final",
        "run_config",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event_id, event in sorted(events.items()):
            summary = event.get("result_summary") or {}
            objective = str(event.get("objective") or "drag").lower()
            cd_final = _find_number(summary, ("cd_final", "cd_best", "best_cd", "cd"))
            cl_final = _find_number(summary, ("cl_final", "cl_best", "best_cl", "cl"))
            target_final = cl_final if objective == "lift" else cd_final
            writer.writerow({
                "solver": event["solver"],
                "job_status": "succeeded",
                "test_case": event_id.replace(":", "_"),
                "job_id": event.get("job_id"),
                "job_uid": event.get("job_uid"),
                "project_uid": event_id,
                "tenant_uid": event.get("tenant_schema"),
                "team_uid": "",
                "s3_bucket": event.get("s3_output_bucket"),
                "s3_output_prefix": str(event.get("s3_output_prefix") or "").rstrip("/"),
                "output_s3_key": str(event.get("s3_output_prefix") or "").rstrip("/") + "/",
                "objective_function": objective,
                "cd_initial": "",
                "cd_final": target_final if target_final is not None else "",
                "cl_final": cl_final if cl_final is not None else "",
                "run_config": json.dumps({
                    **(event.get("conditions") or {}),
                    "objective": objective,
                }),
            })
    print(json.dumps({"events": len(cases), "rejected": len(rejected)}, indent=2))


if __name__ == "__main__":
    main()
