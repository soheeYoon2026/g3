#!/usr/bin/env python
"""Judge a checkpoint on surface-field deltas instead of a single ΔCd.

Integrating Cp to one drag number throws away the local evidence of a small
deformation: on the gtr-smooth pairs the deformed patch carries 1.3-2.3x the
pressure change of the rest of the body (2026-08-26), while ΔCd itself sits
near the CFD noise floor. This evaluator compares the model's predicted ΔCp
field against the CFD ΔCp field, so one pair yields tens of thousands of
observations rather than one.

Reported per pair: correlation and sign agreement of ΔCp inside the deformed
patch, the same outside it (a control — should be weaker), and the patch/far
signal ratio for both CFD and model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_pairs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    pairs = payload["pairs"] if isinstance(payload, dict) else payload
    for pair in pairs:
        if "baseline" not in pair or "variant" not in pair:
            raise ValueError(f"pair needs baseline/variant: {pair}")
    return pairs


def surface_cells(mesh):
    """Cell centres plus the cell-centred pressure field, whatever it is named."""
    import numpy as np

    for name in ("pMeanTrim", "Pressure_Coefficient", "cpavg"):
        if name in mesh.cell_data:
            return (np.asarray(mesh.cell_centers().points, dtype=float),
                    np.asarray(mesh.cell_data[name], dtype=float).reshape(-1))
    raise ValueError(f"no pressure field among {list(mesh.cell_data)}")


def patch_mask(base_centres, moved_points, radius):
    from scipy.spatial import cKDTree

    if moved_points is None or len(moved_points) == 0:
        return np.zeros(len(base_centres), dtype=bool)
    distance, _ = cKDTree(moved_points).query(base_centres, distance_upper_bound=radius)
    return distance < radius


def moved_vertices(stl_dir: Path, base_name: str, variant_name: str, tol=1e-5):
    """Index-matched displacement — the Workbench exports share topology."""
    import trimesh

    base = trimesh.load(stl_dir / base_name, force="mesh", process=False)
    variant = trimesh.load(stl_dir / variant_name, force="mesh", process=False)
    base_v = np.asarray(base.vertices, dtype=float)
    var_v = np.asarray(variant.vertices, dtype=float)
    if base_v.shape != var_v.shape:
        return None
    scale = 1e-3 if float(base.extents.max()) > 100 else 1.0
    moved = np.linalg.norm(var_v - base_v, axis=1) * scale > tol
    return base_v[moved] * scale


def summarize(cfd_delta, model_delta, mask):
    def stats(sel):
        if sel.sum() < 8:
            return None
        cfd, model = cfd_delta[sel], model_delta[sel]
        corr = float(np.corrcoef(cfd, model)[0, 1]) if cfd.std() and model.std() else float("nan")
        return {
            "cells": int(sel.sum()),
            "correlation": corr,
            "sign_agreement": float(np.mean((cfd > 0) == (model > 0))),
            "cfd_rms": float(np.sqrt(np.mean(cfd ** 2))),
            "model_rms": float(np.sqrt(np.mean(model ** 2))),
        }

    inside, outside = stats(mask), stats(~mask)
    result = {"patch": inside, "far": outside}
    if inside and outside and outside["cfd_rms"] and outside["model_rms"]:
        result["cfd_patch_far_ratio"] = inside["cfd_rms"] / outside["cfd_rms"]
        result["model_patch_far_ratio"] = inside["model_rms"] / outside["model_rms"]
    return result


def main() -> None:
    import pyvista as pv
    from scipy.spatial import cKDTree

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="v3 run-dir dataset root")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="dir of per-run predicted surface fields (npy: cell-centred Cp)")
    parser.add_argument("--stl-dir", type=Path, help="upload STLs, for locating the deformed patch")
    parser.add_argument("--stl-map", type=Path, help="json {run: stl filename}")
    parser.add_argument("--patch-radius", type=float, default=0.15)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    stl_map = json.loads(args.stl_map.read_text()) if args.stl_map else {}

    def cfd(run):
        mesh = pv.read(args.root / f"run_{run}" / f"boundary_{run}.vtp").extract_surface()
        return surface_cells(mesh)

    def predicted(run):
        path = args.predictions / f"run_{run}.npy"
        if not path.is_file():
            raise SystemExit(f"missing prediction {path}")
        return np.asarray(np.load(path), dtype=float).reshape(-1)

    rows = []
    for pair in load_pairs(args.pairs):
        base_run, var_run = pair["baseline"], pair["variant"]
        base_centres, base_cfd = cfd(base_run)
        var_centres, var_cfd = cfd(var_run)
        _, idx = cKDTree(var_centres).query(base_centres)
        cfd_delta = var_cfd[idx] - base_cfd
        model_delta = predicted(var_run)[idx] - predicted(base_run)

        mask = np.zeros(len(base_centres), dtype=bool)
        if args.stl_dir and str(base_run) in stl_map and str(var_run) in stl_map:
            moved = moved_vertices(args.stl_dir, stl_map[str(base_run)], stl_map[str(var_run)])
            mask = patch_mask(base_centres, moved, args.patch_radius)

        row = {"baseline": base_run, "variant": var_run, "note": pair.get("note", "")}
        row.update(summarize(cfd_delta, model_delta, mask))
        rows.append(row)
        patch = row.get("patch") or {}
        print(f"{row['note'] or var_run:20s} patch corr {patch.get('correlation', float('nan')):+.2f} "
              f"sign {patch.get('sign_agreement', float('nan')):.0%} "
              f"cells {patch.get('cells', 0):,} | "
              f"cfd ratio {row.get('cfd_patch_far_ratio', float('nan')):.2f} "
              f"model ratio {row.get('model_patch_far_ratio', float('nan')):.2f}", flush=True)

    patch_corrs = [r["patch"]["correlation"] for r in rows if r.get("patch")]
    if patch_corrs:
        print(f"\n변형부 ΔCp 상관 중앙값 {np.median(patch_corrs):+.2f} (n={len(patch_corrs)})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
