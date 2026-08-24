#!/usr/bin/env python3
"""Annotate the dedicated GT-R smooth dataset and build leakage-free LOO splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    variants = {
        row["job_uid"]: row["variant_name"]
        for row in csv.DictReader(args.inventory.open(encoding="utf-8-sig", newline=""))
    }
    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"]
    for case in cases:
        source_id = case["source"]["source_id"]
        job_uid, design = source_id.split(":", 1)
        if job_uid not in variants or design != "RBF_DSN_001":
            raise ValueError(f"unexpected source: {source_id}")
        case["job_uid"] = job_uid
        case["variant_name"] = variants[job_uid]
        for coefficient in ("su2_cd", "su2_cl"):
            if not math.isfinite(float(case["conditions"][coefficient])):
                raise ValueError(f"non-finite {coefficient}: {source_id}")

    if len(cases) != 7 or len({case["variant_name"] for case in cases}) != 7:
        raise ValueError("expected exactly seven unique successful variants")
    baseline = next(case for case in cases if case["variant_name"] == "baseline_original")
    variants_only = [case for case in cases if case is not baseline]

    args.out.mkdir(parents=True, exist_ok=True)
    for index, test_case in enumerate(variants_only):
        validation_case = variants_only[(index + 1) % len(variants_only)]
        train_cases = [
            case for case in cases
            if case["run"] not in {test_case["run"], validation_case["run"]}
        ]
        split = {
            "schema_version": 1,
            "strategy": "leave-one-variant-out",
            "train_cases": train_cases,
            "validation_cases": [validation_case],
            "test_cases": [test_case],
        }
        name = f"fold-{index + 1:02d}-{test_case['variant_name']}.json"
        (args.out / name).write_text(json.dumps(split, indent=2) + "\n")

    final_split = {
        "schema_version": 1,
        "strategy": "all-seven-final-training-sanity-only",
        "train_cases": cases,
        "validation_cases": [baseline],
        "test_cases": [baseline],
    }
    (args.out / "final-training.json").write_text(json.dumps(final_split, indent=2) + "\n")

    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "cases": len(cases),
        "folds": len(variants_only),
        "variants": [case["variant_name"] for case in cases],
    }, indent=2))


if __name__ == "__main__":
    main()
