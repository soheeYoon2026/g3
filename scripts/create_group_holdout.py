#!/usr/bin/env python3
"""Create deterministic, group-isolated training and holdout manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return payload["cases"] if isinstance(payload, dict) else payload


def group_name(row: dict) -> str:
    value = row.get("group_id") or row.get("group")
    if value:
        return str(value)
    raise ValueError(f"case {row.get('case_id', '<unknown>')} has no group_id")


def stable_group_order(groups: dict[str, list[dict]], seed: int) -> list[str]:
    return sorted(
        groups,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
    )


def portable_rows(source: Path, output: Path, selected: list[dict]) -> list[dict]:
    result = []
    for original in selected:
        row = dict(original)
        npz = Path(row["npz"])
        if not npz.is_absolute():
            npz = (source.resolve().parent / npz).resolve()
        if not npz.is_file():
            raise FileNotFoundError(f"missing NPZ for {row['case_id']}: {npz}")
        row["npz"] = os.path.relpath(npz, output.resolve().parent)
        result.append(row)
    return result


def write_manifest(path: Path, source: Path, kind: str, cases: list[dict], seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "format": "g3-group-isolated-split-v1",
        "split": kind,
        "seed": seed,
        "source": str(source.resolve()),
        "groups": sorted({group_name(row) for row in cases}),
        "cases": portable_rows(source, path, cases),
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-out", type=Path, required=True)
    parser.add_argument("--holdout-out", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    if not 0.0 < args.holdout_fraction < 0.5:
        parser.error("--holdout-fraction must be between 0 and 0.5")

    cases = rows(args.manifest)
    groups: dict[str, list[dict]] = {}
    for row in cases:
        groups.setdefault(group_name(row), []).append(row)
    if len(groups) < 3:
        raise ValueError("at least three groups are required for an isolated holdout")

    target = max(1, round(len(cases) * args.holdout_fraction))
    selected_groups, selected_count = [], 0
    for name in stable_group_order(groups, args.seed):
        if selected_count >= target:
            break
        selected_groups.append(name)
        selected_count += len(groups[name])
    selected = set(selected_groups)
    holdout = [row for row in cases if group_name(row) in selected]
    training = [row for row in cases if group_name(row) not in selected]
    if not training or not holdout:
        raise ValueError("split produced an empty training or holdout set")

    write_manifest(args.training_out, args.manifest, "training", training, args.seed)
    write_manifest(args.holdout_out, args.manifest, "golden-holdout", holdout, args.seed)
    print(json.dumps({
        "input_cases": len(cases),
        "input_groups": len(groups),
        "training_cases": len(training),
        "training_groups": len(groups) - len(selected),
        "holdout_cases": len(holdout),
        "holdout_groups": len(selected),
        "target_holdout_cases": target,
    }, indent=2))


if __name__ == "__main__":
    main()
