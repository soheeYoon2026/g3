#!/usr/bin/env python
"""Verify that an AI-input STL and a G2/SU2 case describe the same problem.

Produces a JSON identity report covering the handover checklist: STL hash and
v3 geometry digest, units and axis extents, flow conditions from the SU2
config, and per-step CD/CL from every history file — including locating which
step (if any) produced a given coefficient value. Reuses the exact digest and
config parsing that built the v3 manifests, so digests are comparable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_domino_su2_v3_sources import find_cfd_config, geometry_info
from prepare_domino_su2_v3 import read_flow_conditions

RELATIVE_TOLERANCE = 1e-4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_geometry(path: Path) -> dict:
    import pyvista as pv

    surface = pv.read(path).extract_surface().triangulate()
    digest, watertight = geometry_info(path)
    bounds = np.asarray(surface.bounds, dtype=float).reshape(3, 2)
    extents = bounds[:, 1] - bounds[:, 0]
    normals = surface.face_normals
    areas = surface.compute_cell_sizes(length=False, volume=False).cell_data["Area"]
    # Closed surfaces are counted front and back, hence the factor of one half.
    projected = 0.5 * np.abs(normals * np.asarray(areas)[:, None]).sum(axis=0)
    max_extent = float(extents.max())
    if max_extent > 100.0:
        units_note = "max extent > 100: likely millimeters, not meters"
    elif max_extent < 0.5:
        units_note = "max extent < 0.5: check units and scale"
    else:
        units_note = "meter-scale extents"
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "geometry_digest": digest,
        "watertight": bool(watertight),
        "points": int(surface.n_points),
        "faces": int(surface.n_cells),
        "bounds": bounds.tolist(),
        "extents": extents.tolist(),
        "longest_axis": "xyz"[int(np.argmax(extents))],
        "projected_area": {axis: float(projected[i]) for i, axis in enumerate("xyz")},
        "units_note": units_note,
    }


def summarize_history(path: Path, step=None, find_cd=None, find_tolerance=5e-5, tail=100) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    header = [value.strip().strip('"') for value in rows[0]]
    body = [row for row in rows[1:] if len(row) == len(header)]
    summary = {"path": str(path), "columns": header, "steps": len(body)}
    if "CD" not in header or not body:
        summary["note"] = "no CD column or no data rows"
        return summary
    cd_index = header.index("CD")
    cl_index = header.index("CL") if "CL" in header else None
    iter_name = next(
        (name for name in ("Inner_Iter", "Outer_Iter", "Time_Iter") if name in header), None
    )

    def step_of(row_number, row):
        if iter_name is None:
            return row_number
        return int(float(row[header.index(iter_name)]))

    cd = np.asarray([float(row[cd_index]) for row in body])
    window = cd[-tail:]
    summary.update({
        "iteration_column": iter_name,
        "first_step": step_of(0, body[0]),
        "last_step": step_of(len(body) - 1, body[-1]),
        "cd_last": float(cd[-1]),
        "cl_last": float(body[-1][cl_index]) if cl_index is not None else None,
        "cd_tail_mean": float(window.mean()),
        "cd_tail_std": float(window.std()),
        "cd_tail_steps": int(len(window)),
        "cd_min": float(cd.min()),
        "cd_max": float(cd.max()),
    })
    if step is not None:
        matches = [row for number, row in enumerate(body) if step_of(number, row) == step]
        summary["cd_at_step"] = float(matches[0][cd_index]) if matches else None
    if find_cd is not None:
        hits = [
            {"step": step_of(number, row), "cd": float(row[cd_index])}
            for number, row in enumerate(body)
            if abs(float(row[cd_index]) - find_cd) <= find_tolerance
        ]
        summary["find_cd"] = {
            "target": find_cd,
            "tolerance": find_tolerance,
            "matches": len(hits),
            "hits": hits[:20],
        }
    return summary


def check(name, status, detail):
    return {"name": name, "status": status, "detail": detail}


def relative_check(name, expected, actual, tolerance=RELATIVE_TOLERANCE):
    if expected is None or actual is None:
        return check(name, "skipped", f"expected={expected} actual={actual}")
    expected, actual = float(expected), float(actual)
    scale = max(abs(expected), abs(actual), 1e-12)
    status = "ok" if abs(expected - actual) / scale <= tolerance else "mismatch"
    return check(name, status, f"expected={expected:.9g} actual={actual:.9g}")


def build_checks(stl, vtu, flow, conditions, histories, expect_digest):
    checks = []
    if stl and expect_digest:
        status = "ok" if stl["geometry_digest"] == expect_digest else "mismatch"
        checks.append(check("stl_vs_expected_digest", status, stl["geometry_digest"]))
    if stl and vtu:
        status = "ok" if stl["geometry_digest"] == vtu["geometry_digest"] else "mismatch"
        checks.append(check(
            "stl_vs_vtu_digest", status,
            "identical tessellation" if status == "ok"
            else "different tessellation (expected when the STL was re-exported)",
        ))
        difference = float(np.abs(np.asarray(stl["bounds"]) - np.asarray(vtu["bounds"])).max())
        scale = max(max(stl["extents"]), 1e-12)
        checks.append(check(
            "stl_vs_vtu_bounds",
            "ok" if difference / scale <= 1e-3 else "mismatch",
            f"max bound difference {difference:.6g} over extent {scale:.6g}",
        ))
    if flow and conditions:
        for key in ("speed", "aoa", "sideslip", "density", "ref_area"):
            checks.append(relative_check(f"cfg_vs_conditions_{key}", flow.get(key), conditions.get(key)))
    if conditions and histories:
        target = conditions.get("su2_cd")
        located = []
        for history in histories:
            hits = history.get("find_cd", {}).get("hits", []) if target is not None else []
            located.extend(f"{Path(history['path']).name}@{hit['step']}" for hit in hits)
        if target is None:
            checks.append(check("conditions_cd_in_history", "skipped", "no su2_cd in conditions"))
        else:
            checks.append(check(
                "conditions_cd_in_history",
                "ok" if located else "mismatch",
                f"su2_cd={target} found at: {', '.join(located[:8]) or 'no step in any history file'}",
            ))
    if stl and flow and flow.get("ref_area"):
        ratio = stl["projected_area"]["x"] / float(flow["ref_area"])
        checks.append(check(
            "ref_area_vs_projected_x", "info",
            f"projected +X area {stl['projected_area']['x']:.6g} / REF_AREA {float(flow['ref_area']):.6g} = {ratio:.3f}",
        ))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", type=Path, help="AI-input STL to fingerprint")
    parser.add_argument("--case-dir", type=Path, help="G2 run directory (auto-detects vtu/cfg/history)")
    parser.add_argument("--vtu", type=Path, help="override surface VTU path")
    parser.add_argument("--cfg", type=Path, help="override SU2 config path")
    parser.add_argument("--conditions", type=Path, help="v3 conditions_<run>.json to cross-check")
    parser.add_argument("--expect-digest", help="geometry_digest from a manifest to compare with the STL")
    parser.add_argument("--step", type=int, help="report CD/CL at this iteration")
    parser.add_argument("--find-cd", type=float, help="locate steps where CD matches this value")
    parser.add_argument("--find-tolerance", type=float, default=5e-5)
    parser.add_argument("--tail", type=int, default=100)
    parser.add_argument("--out", type=Path, help="write the JSON report here as well")
    parser.add_argument("--strict", action="store_true", help="exit 1 when any check mismatches")
    args = parser.parse_args()

    vtu = args.vtu
    cfg = args.cfg
    histories = []
    if args.case_dir:
        vtu = vtu or (args.case_dir / "surface_flow.vtu")
        cfg = cfg or find_cfd_config(args.case_dir)
        history_paths = sorted(args.case_dir.glob("history*.csv"))
    else:
        history_paths = []

    conditions = json.loads(args.conditions.read_text()) if args.conditions else None
    find_cd = args.find_cd
    if find_cd is None and conditions is not None:
        find_cd = conditions.get("su2_cd")

    report = {"checks": []}
    if args.stl:
        report["stl"] = describe_geometry(args.stl)
    if vtu and Path(vtu).is_file():
        report["surface_vtu"] = describe_geometry(Path(vtu))
    flow = None
    if cfg and Path(cfg).is_file():
        flow = read_flow_conditions(cfg)
        report["config"] = {
            "path": str(cfg),
            **{key: (value.tolist() if isinstance(value, np.ndarray) else value)
               for key, value in flow.items() if key != "rotation"},
        }
    for path in history_paths:
        histories.append(summarize_history(
            path, step=args.step, find_cd=find_cd,
            find_tolerance=args.find_tolerance, tail=args.tail,
        ))
    if histories:
        report["history"] = histories
    if conditions is not None:
        report["conditions"] = {"path": str(args.conditions), **conditions}

    report["checks"] = build_checks(
        report.get("stl"), report.get("surface_vtu"), flow, conditions, histories,
        args.expect_digest,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    if args.strict and any(item["status"] == "mismatch" for item in report["checks"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
