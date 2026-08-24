#!/usr/bin/env python3
"""Split a DoMINO manifest without leaking designs from one G2 job.

G2 optimization jobs may emit FINAL and multiple RBF_DSN designs.  Those
designs are closely related, so every design from one source job must remain
in the same train, validation, or test partition.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def source_job(case: dict) -> str:
    source_id = str(case.get("source", {}).get("source_id", ""))
    if source_id:
        return source_id.split(":", 1)[0]
    return str(case.get("group_id") or case.get("geometry_digest") or case["run"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        groups[source_job(case)].append(case)

    keys = sorted(groups)
    if len(keys) < 3:
        raise SystemExit("at least three independent source jobs are required")
    random.Random(args.seed).shuffle(keys)

    validation_count = max(1, round(len(keys) * args.validation_fraction))
    test_count = max(1, round(len(keys) * args.test_fraction))
    if validation_count + test_count >= len(keys):
        raise SystemExit("validation and test partitions leave no training jobs")

    validation_keys = set(keys[:validation_count])
    test_keys = set(keys[validation_count : validation_count + test_count])
    train_keys = set(keys) - validation_keys - test_keys

    def rows(selected: set[str]) -> list[dict]:
        return sorted(
            (case for key in selected for case in groups[key]),
            key=lambda case: int(case["run"]),
        )

    result = {
        "schema_version": 1,
        "seed": args.seed,
        "source_manifest": str(args.manifest.resolve()),
        "split_unit": "g2_source_job",
        "train_groups": len(train_keys),
        "validation_groups": len(validation_keys),
        "test_groups": len(test_keys),
        "train_cases": rows(train_keys),
        "validation_cases": rows(validation_keys),
        "test_cases": rows(test_keys),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "groups": len(keys),
                "train_groups": len(train_keys),
                "validation_groups": len(validation_keys),
                "test_groups": len(test_keys),
                "train_cases": len(result["train_cases"]),
                "validation_cases": len(result["validation_cases"]),
                "test_cases": len(result["test_cases"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
