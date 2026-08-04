#!/usr/bin/env python3
"""Audit G2 manifests for missing metadata and contradictory labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CONDITIONS = (
    "u_x", "u_y", "u_z", "density", "viscosity", "temperature",
    "ref_length", "ref_area",
)


def rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return payload["cases"] if isinstance(payload, dict) else payload


def resolved_npz(manifest: Path, row: dict) -> Path:
    path = Path(row["npz"])
    return path if path.is_absolute() else (manifest.resolve().parent / path).resolve()


def geometry_digest(data) -> str:
    digest = hashlib.sha256()
    for key in ("geometry_points", "geometry_normals"):
        value = np.ascontiguousarray(np.asarray(data[key], dtype=np.float32))
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cd-conflict-threshold", type=float, default=0.02)
    parser.add_argument("--cl-conflict-threshold", type=float, default=0.02)
    args = parser.parse_args()

    cases = rows(args.manifest)
    audited, invalid = [], []
    clusters: dict[tuple, list[dict]] = defaultdict(list)
    condition_values = defaultdict(list)
    for row in cases:
        case_id = str(row.get("case_id", "<missing>"))
        reasons = []
        conditions = row.get("conditions") or {}
        missing = [name for name in CONDITIONS if name not in conditions]
        if missing:
            reasons.append(f"missing conditions: {', '.join(missing)}")
        values = []
        for name in CONDITIONS:
            value = conditions.get(name, np.nan)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = np.nan
            values.append(value)
            if np.isfinite(value):
                condition_values[name].append(value)
        if not np.isfinite(values).all():
            reasons.append("non-finite condition")
        for name in ("density", "viscosity", "temperature", "ref_length", "ref_area"):
            value = conditions.get(name)
            if value is not None and np.isfinite(float(value)) and float(value) <= 0:
                reasons.append(f"non-positive {name}")

        coefficients = row.get("coefficients") or {}
        cd, cl = float(coefficients.get("cd", np.nan)), float(coefficients.get("cl", np.nan))
        if not np.isfinite(cd) or cd <= 0:
            reasons.append("invalid cd")
        if not np.isfinite(cl):
            reasons.append("invalid cl")
        path = resolved_npz(args.manifest, row)
        fingerprint = None
        try:
            with np.load(path) as data:
                required = {"geometry_points", "geometry_normals"}
                if not required.issubset(data.files):
                    reasons.append("NPZ missing geometry arrays")
                else:
                    points = np.asarray(data["geometry_points"])
                    normals = np.asarray(data["geometry_normals"])
                    if points.ndim != 2 or points.shape[1] != 3 or points.shape != normals.shape:
                        reasons.append("invalid geometry array shapes")
                    elif not np.isfinite(points).all() or not np.isfinite(normals).all():
                        reasons.append("non-finite geometry")
                    else:
                        fingerprint = geometry_digest(data)
        except Exception as exc:
            reasons.append(f"unreadable NPZ: {exc}")

        item = {
            "case_id": case_id,
            "group_id": row.get("group_id"),
            "npz": str(path),
            "cd": cd,
            "cl": cl,
            "conditions": dict(zip(CONDITIONS, values)),
            "geometry_digest": fingerprint,
            "reasons": reasons,
        }
        audited.append(item)
        if reasons:
            invalid.append(item)
        elif fingerprint:
            signature = tuple(round(value, 9) for value in values)
            clusters[(fingerprint, signature)].append(item)

    conflicts = []
    for (_, _), members in clusters.items():
        if len(members) < 2:
            continue
        cd_values = [item["cd"] for item in members]
        cl_values = [item["cl"] for item in members]
        cd_span, cl_span = max(cd_values) - min(cd_values), max(cl_values) - min(cl_values)
        if cd_span > args.cd_conflict_threshold or cl_span > args.cl_conflict_threshold:
            conflicts.append({
                "case_ids": [item["case_id"] for item in members],
                "group_ids": sorted({str(item["group_id"]) for item in members}),
                "conditions": members[0]["conditions"],
                "cd_values": cd_values,
                "cd_span": cd_span,
                "cl_values": cl_values,
                "cl_span": cl_span,
            })

    ranges = {}
    for name, values in condition_values.items():
        ranges[name] = {"min": min(values), "max": max(values)} if values else None
    report = {
        "manifest": str(args.manifest.resolve()),
        "cases": len(cases),
        "invalid_cases": len(invalid),
        "duplicate_condition_geometry_clusters": sum(len(v) > 1 for v in clusters.values()),
        "conflicting_clusters": len(conflicts),
        "condition_ranges": ranges,
        "invalid": invalid,
        "conflicts": conflicts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "cases", "invalid_cases", "duplicate_condition_geometry_clusters",
        "conflicting_clusters", "condition_ranges",
    )}, indent=2))


if __name__ == "__main__":
    main()
