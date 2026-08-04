#!/usr/bin/env python3
"""Summarize coefficient and condition distributions in G3 JSON manifests."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def coefficient(row: dict, name: str):
    value = row.get(f"{name}_final")
    if value is None:
        value = row.get("coefficients", {}).get(name)
    return finite_number(value)


def summary(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": values[0],
        "p25": values[(len(values) - 1) // 4],
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p75": values[(3 * (len(values) - 1)) // 4],
        "max": values[-1],
        "std": statistics.pstdev(values),
    }


def analyze(path: Path):
    payload = json.loads(path.read_text())
    rows = payload["cases"] if isinstance(payload, dict) else payload
    groups = {str(row.get("group_id", row.get("case_id"))) for row in rows}
    return {
        "manifest": str(path.resolve()),
        "cases": len(rows),
        "groups": len(groups),
        "cd": summary([coefficient(row, "cd") for row in rows]),
        "cl": summary([coefficient(row, "cl") for row in rows]),
        "ref_length": summary(
            [finite_number(row.get("conditions", {}).get("ref_length")) for row in rows]
        ),
        "ref_area": summary(
            [finite_number(row.get("conditions", {}).get("ref_area")) for row in rows]
        ),
        "surface_points": summary(
            [
                finite_number(
                    row.get("surface_points", row.get("counts", {}).get("surface"))
                )
                for row in rows
            ]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([analyze(path) for path in args.manifests], indent=2))


if __name__ == "__main__":
    main()
