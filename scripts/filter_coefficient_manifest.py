#!/usr/bin/env python3
"""Apply point-count and coefficient-range gates to an expert manifest."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coefficient", choices=("cd", "cl"), required=True)
    parser.add_argument("--min-points", type=int, default=4_000)
    parser.add_argument("--min-value", type=float)
    parser.add_argument("--max-value", type=float)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    rows = payload.get("cases", payload)
    accepted, rejected = [], []
    key = f"{args.coefficient}_final"
    for original in rows:
        row = copy.deepcopy(original)
        raw = row.get(key, row.get("coefficients", {}).get(args.coefficient))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = math.nan
        points = int(row.get("surface_points", row.get("counts", {}).get("surface", 0)))
        reasons = []
        if points < args.min_points:
            reasons.append(f"surface_points_{points}_below_{args.min_points}")
        if not math.isfinite(value):
            reasons.append(f"non_finite_{args.coefficient}")
        if args.min_value is not None and value < args.min_value:
            reasons.append(f"{args.coefficient}_{value}_below_{args.min_value}")
        if args.max_value is not None and value > args.max_value:
            reasons.append(f"{args.coefficient}_{value}_above_{args.max_value}")
        if reasons:
            rejected.append({
                "case_id": row.get("case_id"),
                "surface_points": points,
                args.coefficient: value,
                "reasons": reasons,
            })
        else:
            accepted.append(row)

    output = {
        "schema_version": 1,
        "format": "g3-quality-filtered-coefficient-manifest-v1",
        "source": str(args.manifest),
        "gate": {
            "coefficient": args.coefficient,
            "min_points": args.min_points,
            "min_value": args.min_value,
            "max_value": args.max_value,
        },
        "summary": {
            "input": len(rows),
            "accepted": len(accepted),
            "rejected": len(rejected),
        },
        "cases": accepted,
        "rejected": rejected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), **output["summary"]}, indent=2))
    for row in rejected:
        print(f"REJECT {row['case_id']}: {', '.join(row['reasons'])}")


if __name__ == "__main__":
    main()
