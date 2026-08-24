#!/usr/bin/env python3
"""Join the control-point Cd split to materialized DoMINO cases."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def source_job(case: dict) -> str:
    return str(case.get("source", {}).get("source_id", "")).split(":", 1)[0]


def source_design(case: dict) -> str:
    source_id = str(case.get("source", {}).get("source_id", ""))
    return source_id.split(":", 1)[1] if ":" in source_id else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    dataset = json.loads(args.dataset.read_text())
    final_by_job = {
        source_job(case): case
        for case in manifest["cases"]
        if source_design(case) == "FINAL"
    }
    baseline_meta = dataset["baseline"]
    baseline = final_by_job[baseline_meta["job_uid"]]

    def cases_for(split: str) -> list[dict]:
        anchor = copy.deepcopy(baseline)
        anchor["source"]["source_id"] = f"cd_{split}:FINAL"
        anchor["recommendation_label"] = "baseline"
        rows = [anchor]
        for sample in dataset["samples"]:
            if sample["split"] != split:
                continue
            case = copy.deepcopy(final_by_job[sample["job_uid"]])
            case["source"]["source_id"] = f"cd_{split}:{sample['label']}"
            case["recommendation_label"] = sample["label"]
            case["control_id"] = sample["control_id"]
            case["g2_delta_cd"] = sample["g2_delta_cd"]
            rows.append(case)
        return rows

    result = {
        "schema_version": 1,
        "objective": "delta_cd_only",
        "source_manifest": str(args.manifest.resolve()),
        "source_dataset": str(args.dataset.resolve()),
        "split_unit": "control_point_with_shared_baseline",
        "train_cases": cases_for("train"),
        "validation_cases": cases_for("validation"),
        "test_cases": cases_for("test"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "train_candidates": len(result["train_cases"]) - 1,
        "validation_candidates": len(result["validation_cases"]) - 1,
        "test_candidates": len(result["test_cases"]) - 1,
    }, indent=2))


if __name__ == "__main__":
    main()
