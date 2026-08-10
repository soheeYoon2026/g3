#!/usr/bin/env python
"""Create a deterministic geometry-group split for a DoMINO v3 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def group_score(group_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-groups", type=int, default=8)
    parser.add_argument("--seed", default="g3-domino-v3-20260810")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cases = [row for row in manifest["cases"] if row["accepted"]]
    groups = sorted({row["group_id"] for row in cases}, key=lambda value: group_score(value, args.seed))
    if not 0 < args.test_groups < len(groups):
        raise SystemExit("test-groups must leave at least one train and one test group")
    test_groups = set(groups[: args.test_groups])
    train = [row for row in cases if row["group_id"] not in test_groups]
    test = [row for row in cases if row["group_id"] in test_groups]
    output = {
        "schema_version": 1,
        "seed": args.seed,
        "source_manifest": str(args.manifest.resolve()),
        "train_groups": len({row["group_id"] for row in train}),
        "test_groups": len({row["group_id"] for row in test}),
        "train_cases": train,
        "test_cases": test,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "train_groups": output["train_groups"], "train_cases": len(train),
        "test_groups": output["test_groups"], "test_cases": len(test),
        "group_overlap": len(
            {row["group_id"] for row in train} & {row["group_id"] for row in test}
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
