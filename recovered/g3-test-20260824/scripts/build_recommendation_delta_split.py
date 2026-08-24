#!/usr/bin/env python3
"""Build a split that compares each recommended shape with one baseline.

Only the FINAL geometry from each successful full-resolution G2 job is used.
The common baseline is included in every partition as the fixed reference;
candidate jobs themselves remain disjoint across train, validation, and test.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path


def job_uid(case: dict) -> str:
    return str(case.get("source", {}).get("source_id", "")).split(":", 1)[0]


def design(case: dict) -> str:
    source_id = str(case.get("source", {}).get("source_id", ""))
    return source_id.split(":", 1)[1] if ":" in source_id else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--test-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    final_by_job = {job_uid(case): case for case in manifest["cases"] if design(case) == "FINAL"}
    comparison = json.loads(args.comparison.read_text(encoding="utf-8-sig"))
    succeeded = [row for row in comparison["rows"] if row.get("status") == "succeeded" and row.get("g2_cd") is not None]
    baseline_meta = next(row for row in succeeded if row["label"] == "baseline")
    baseline = final_by_job[baseline_meta["job_uid"]]
    candidates = [row for row in succeeded if row["label"] != "baseline" and row["job_uid"] in final_by_job]
    random.Random(args.seed).shuffle(candidates)
    if args.validation_count + args.test_count >= len(candidates):
        raise SystemExit("validation and test counts leave no training candidates")

    partitions = {
        "validation": candidates[: args.validation_count],
        "test": candidates[args.validation_count : args.validation_count + args.test_count],
        "train": candidates[args.validation_count + args.test_count :],
    }

    def cases_for(name: str) -> list[dict]:
        anchor = copy.deepcopy(baseline)
        anchor["source"]["source_id"] = f"recommendation_{name}:FINAL"
        anchor["recommendation_label"] = "baseline"
        rows = [anchor]
        for meta in partitions[name]:
            case = copy.deepcopy(final_by_job[meta["job_uid"]])
            case["source"]["source_id"] = f"recommendation_{name}:{meta['label']}"
            case["recommendation_label"] = meta["label"]
            case["g2_delta_cd"] = meta["g2_delta_cd"]
            rows.append(case)
        return rows

    result = {
        "schema_version": 1,
        "seed": args.seed,
        "source_manifest": str(args.manifest.resolve()),
        "split_unit": "recommendation_job_with_shared_baseline",
        "train_candidates": len(partitions["train"]),
        "validation_candidates": len(partitions["validation"]),
        "test_candidates": len(partitions["test"]),
        "train_cases": cases_for("train"),
        "validation_cases": cases_for("validation"),
        "test_cases": cases_for("test"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "train_candidates", "validation_candidates", "test_candidates")}, indent=2))


if __name__ == "__main__":
    main()
