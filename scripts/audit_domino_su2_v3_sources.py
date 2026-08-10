#!/usr/bin/env python
"""Audit legacy DoMINO source cases before rebuilding a flow-aligned v3 set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_domino_su2_v3 import read_flow_conditions, read_su2_coefficients


def find_cfd_config(case_dir: Path) -> Path | None:
    candidates = sorted(case_dir.glob("*_CFD.cfg")) or sorted(case_dir.glob("*.cfg"))
    for path in candidates:
        name = path.name.lower()
        if "adjoint" not in name and "ffd" not in name:
            return path
    return None


def geometry_info(vtu_path: Path) -> tuple[str, bool]:
    import pyvista as pv

    surface = pv.read(vtu_path).extract_surface().triangulate()
    digest = hashlib.sha256()
    points = np.ascontiguousarray(np.round(np.asarray(surface.points), 7), dtype=np.float64)
    faces = np.ascontiguousarray(np.asarray(surface.faces), dtype=np.int64)
    digest.update(points.tobytes())
    digest.update(faces.tobytes())
    return digest.hexdigest(), surface.n_open_edges == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    entries = json.loads(args.manifest.read_text())
    rows = []
    for entry in entries:
        source = Path(entry["src"])
        case_dir = source if source.is_absolute() else args.source_root / source
        vtu = case_dir / "surface_flow.vtu"
        cfg = find_cfd_config(case_dir)
        row = {"legacy_run": entry.get("run"), "case_dir": str(case_dir), "ready": False}
        try:
            if not vtu.is_file():
                raise ValueError("missing surface_flow.vtu")
            if cfg is None:
                raise ValueError("missing CFD config")
            flow = read_flow_conditions(cfg)
            cd, cl = read_su2_coefficients(case_dir)
            if not np.isfinite([cd, cl]).all():
                raise ValueError("missing finite Cd/Cl")
            if flow["ref_area"] is None or float(flow["ref_area"]) <= 0:
                raise ValueError("missing positive REF_AREA")
            digest, watertight = geometry_info(vtu)
            if not watertight:
                raise ValueError("surface is not watertight")
            row.update({
                "ready": True,
                "vtu": str(vtu),
                "cfg": str(cfg),
                "cd": cd,
                "cl": cl,
                "speed": flow["speed"],
                "mach": flow["mach"],
                "aoa": flow["aoa"],
                "sideslip": flow["sideslip"],
                "ref_area": flow["ref_area"],
                "geometry_digest": digest,
                "watertight": watertight,
            })
        except Exception as exc:
            row["reason"] = str(exc)
        rows.append(row)

    accepted = [row for row in rows if row["ready"]]
    digest_counts = Counter(row["geometry_digest"] for row in accepted)
    aoa_counts = Counter(round(float(row["aoa"]), 3) for row in accepted)
    report = {
        "input": len(rows),
        "accepted": len(accepted),
        "rejected": len(rows) - len(accepted),
        "unique_geometries": len(digest_counts),
        "duplicate_cases": sum(count - 1 for count in digest_counts.values()),
        "aoa_counts": dict(sorted(aoa_counts.items())),
        "rejection_reasons": dict(Counter(row.get("reason", "") for row in rows if not row["ready"])),
        "cases": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
