#!/usr/bin/env python
"""Stream a DrivAerML/WindsorML surface subset into v3 run-dir format.

Per run: download boundary_i.vtp (+ force/geo CSVs) from the HuggingFace
dataset, decimate the surface to a target cell count while re-sampling the
mean-field point data (pMeanTrim, wallShearStressMeanTrim), write
run_i/{boundary_i.vtp, conditions_i.json}, then delete the raw download.
Keeps disk use bounded so a 500MB-per-case dataset fits a 13GB margin.

CC-BY-SA 4.0 datasets (commercial-clean); DrivAerNet/DrivAerStar must NOT be
ingested here (non-commercial licenses).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def read_csv_row(path: Path) -> dict:
    with path.open(newline="") as fp:
        rows = list(csv.reader(fp))
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    header = [c.strip().lstrip("#").strip() for c in rows[0]]
    values = rows[1] if len(rows) > 1 else []
    out = {}
    for key, value in zip(header, values):
        try:
            out[key.lower()] = float(value)
        except ValueError:
            out[key.lower()] = value.strip()
    return out


def pick(row: dict, *names, default=None):
    for name in names:
        for key, value in row.items():
            if name in key and isinstance(value, float):
                return value
    return default


def main() -> None:
    import pyvista as pv
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="neashton/drivaerml")
    ap.add_argument("--prefix", default="drivaer", help="file stem prefix inside runs")
    ap.add_argument("--runs", required=True, help="e.g. 1-80 or 1,5,9")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-cells", type=int, default=200_000)
    ap.add_argument("--speed", type=float, default=38.889)
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--cache", type=Path, default=Path("/tmp/hf-ingest-cache"))
    args = ap.parse_args()

    if "drivaernet" in args.repo.lower() or "drivaerstar" in args.repo.lower():
        raise SystemExit("non-commercial dataset — refusing to ingest")

    runs = []
    for part in args.runs.split(","):
        if "-" in part:
            a, b = part.split("-")
            runs.extend(range(int(a), int(b) + 1))
        else:
            runs.append(int(part))

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "ingest_manifest.jsonl"
    done = set()
    if manifest_path.exists():
        for line in manifest_path.open():
            done.add(json.loads(line)["run"])

    for run in runs:
        if run in done:
            print(f"[skip] run_{run} already ingested", flush=True)
            continue
        free_gb = shutil.disk_usage(args.out).free / 1e9
        if free_gb < args.min_free_gb:
            print(f"[stop] free disk {free_gb:.1f}GB < {args.min_free_gb}GB", flush=True)
            break
        record = {"run": run, "repo": args.repo}
        try:
            def fetch(candidates):
                last = None
                for name in candidates:
                    try:
                        return name, Path(hf_hub_download(
                            args.repo, f"run_{run}/{name}",
                            repo_type="dataset", cache_dir=str(args.cache)))
                    except Exception as exc:
                        last = exc
                raise last

            force = read_csv_row(fetch([f"force_mom_{run}.csv"])[1])
            try:
                geo = read_csv_row(
                    fetch([f"geo_ref_{run}.csv", f"geo_parameters_{run}.csv"])[1])
            except Exception:
                geo = {}
            surf_name, vtp_path = fetch([f"boundary_{run}.vtp", f"boundary_{run}.vtu"])
            ext = surf_name[surf_name.rfind("."):]
            # HF cache may hand back an extension-less blob; give VTK a real extension.
            readable = args.cache / f"b_{run}{ext}"
            readable.unlink(missing_ok=True)
            try:
                readable.hardlink_to(vtp_path.resolve())
            except OSError:
                shutil.copy(vtp_path, readable)

            mesh = pv.read(readable).extract_surface().triangulate()
            n_cells = mesh.n_cells
            if n_cells > args.target_cells:
                reduction = 1.0 - args.target_cells / n_cells
                deci = mesh.decimate(reduction)
                deci = deci.sample(mesh, snap_to_closest_point=True)
            else:
                deci = mesh
            # Decimation leaves zero-area slivers; DoMINO's solver head divides
            # by cell area, so they poison predictions with NaN (2026-08-26).
            import numpy as np
            sized = deci.compute_cell_sizes(length=False, volume=False)
            keep = np.asarray(sized.cell_data["Area"]) > 1e-12
            if not keep.all():
                deci = deci.extract_cells(np.where(keep)[0]).extract_surface().triangulate()
            import numpy as np
            dyn = 0.5 * args.speed ** 2 * args.density   # kinematic when density=1
            if "pMeanTrim" not in deci.point_data:
                if "pMean" in deci.point_data:
                    deci.point_data["pMeanTrim"] = deci.point_data["pMean"]
                elif "cpavg" in deci.point_data:   # WindsorML stores coefficients
                    deci.point_data["pMeanTrim"] = (
                        np.asarray(deci.point_data["cpavg"], dtype=float) * dyn)
                else:
                    raise ValueError(f"no pressure field; fields={list(deci.point_data)}")
            if "wallShearStressMeanTrim" not in deci.point_data:
                if "wallShearStressMean" in deci.point_data:
                    deci.point_data["wallShearStressMeanTrim"] = deci.point_data["wallShearStressMean"]
                elif all(k in deci.point_data for k in ("cfxavg", "cfyavg", "cfzavg")):
                    deci.point_data["wallShearStressMeanTrim"] = np.stack(
                        [np.asarray(deci.point_data[k], dtype=float) for k in ("cfxavg", "cfyavg", "cfzavg")],
                        axis=1) * dyn
                else:
                    raise ValueError(f"no shear field; fields={list(deci.point_data)}")

            run_dir = args.out / f"run_{run}"
            run_dir.mkdir(exist_ok=True)
            deci.save(run_dir / f"boundary_{run}.vtp")
            cd = pick(force, "cd", "drag")
            cl = pick(force, "cl", "lift")
            ref_area = pick(geo, "aref", "area", default=None) or pick(force, "aref", "area", default=2.17)
            conditions = {
                "speed": args.speed,
                "velocity_world": [args.speed, 0.0, 0.0],
                "mach": None, "aoa": 0.0, "sideslip": 0.0,
                "density": args.density,
                "ref_area": float(ref_area),
                "su2_cd": cd, "su2_cl": cl,
                "flow_aligned": True,
                "source": f"{args.repo}/run_{run} (CC-BY-SA 4.0)",
            }
            (run_dir / f"conditions_{run}.json").write_text(json.dumps(conditions, indent=1))
            record.update({"ok": True, "cells": int(deci.n_cells), "cd": cd, "cl": cl,
                           "ref_area": float(ref_area)})
            print(f"[ok] run_{run}: {n_cells:,} -> {deci.n_cells:,} cells, cd={cd}", flush=True)
        except Exception as exc:
            record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
            print(f"[fail] run_{run}: {record['error']}", flush=True)
        finally:
            shutil.rmtree(args.cache, ignore_errors=True)
        with manifest_path.open("a") as fp:
            fp.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
