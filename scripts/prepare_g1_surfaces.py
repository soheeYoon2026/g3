#!/usr/bin/env python3
"""Build G1 surface Cp/velocity/Cd samples directly from S3 VTP outputs."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import boto3
import numpy as np

from build_g1_dataset import region_from_bucket


def read_polydata(path: Path):
    import vtk

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def point_array(poly, name: str) -> np.ndarray:
    from vtk.util.numpy_support import vtk_to_numpy

    array = poly.GetPointData().GetArray(name)
    if array is None:
        raise ValueError(f"VTP is missing point array {name}")
    return vtk_to_numpy(array)


def merge_body(polys):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    append = vtk.vtkAppendPolyData()
    for poly in polys:
        append.AddInputData(poly)
    append.Update()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(append.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.Update()
    poly = normals.GetOutput()
    points = vtk_to_numpy(poly.GetPoints().GetData())
    normal = point_array(poly, "Normals")
    pressure = point_array(poly, "pMean")
    velocity = point_array(poly, "UMean")
    return points, normal, pressure, velocity


def download(client, bucket: str, key: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(target))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("--objective-function", choices=("drag", "lift"), default="drag",
                        help="G1 objective rows to materialize")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--solver", default=None,
                        help="optional solver filter for multi-solver smoke reports")
    parser.add_argument("--dedupe-key", default=None,
                        help="keep the last successful row for this column, e.g. test_case")
    parser.add_argument("--coefficient-expert", default="g1_openfoam")
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for source in raw_rows:
        row = dict(source)
        # Accept both the historical G1 inventory and the multi-solver smoke
        # report without maintaining a second downloader.
        if "job_status" in row:
            row["status"] = row.get("job_status", "")
            row["output_s3_key"] = row.get("s3_output_prefix", "").rstrip("/") + "/"
            try:
                run_config = json.loads(row.get("run_config") or "{}")
            except json.JSONDecodeError:
                run_config = {}
            row["objective_function"] = str(
                run_config.get("objectiveFunction") or run_config.get("objective") or "drag"
            ).lower()
        rows.append(row)
    if args.solver:
        rows = [row for row in rows if row.get("solver") == args.solver]
    rows = [row for row in rows if row.get("status") == "succeeded"
            and row.get("objective_function") == args.objective_function
            and row.get("cd_final", "").strip()]
    if args.dedupe_key:
        deduplicated = {}
        for row in rows:
            deduplicated[row.get(args.dedupe_key) or row["job_uid"]] = row
        rows = list(deduplicated.values())
    coefficient_name = "cd" if args.objective_function == "drag" else "cl"
    if args.limit is not None:
        rows = rows[:args.limit]
    cases_dir = args.out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    clients = {}
    cases, skipped = [], []

    for number, row in enumerate(rows, 1):
        job = row["job_uid"]
        target = cases_dir / f"{job}.npz"
        try:
            bucket = row["s3_bucket"]
            region = region_from_bucket(bucket)
            client = clients.setdefault(region, boto3.client("s3", region_name=region))
            prefix = row["output_s3_key"]
            manifest_key = prefix + "vtp/manifest.json"
            inlet_key = prefix + "vtp/inlet.vtp"
            outlet_key = prefix + "vtp/outlet.vtp"
            try:
                manifest = json.loads(client.get_object(
                    Bucket=bucket, Key=manifest_key
                )["Body"].read())
                manifest_entries = [item for item in manifest["files"] if item.get("vtp_file")]
                # New G1 manifests contain all optimisation cycles. Select the
                # last exported boundary directory so inlet/outlet and body
                # fields describe the same final state.
                directories = {}
                for item in manifest_entries:
                    directory = str(Path(item["vtp_file"]).parent)
                    directories.setdefault(directory, []).append(item)
                selected_directory = max(
                    directories,
                    key=lambda directory: max(
                        int(item.get("artifact_id") or 0) for item in directories[directory]
                    ),
                )
                selected_entries = directories[selected_directory]
                inlet_item = next(
                    item for item in selected_entries
                    if (item.get("file_name") or Path(item["vtp_file"]).name).lower() == "inlet.vtp"
                )
                outlet_item = next(
                    item for item in selected_entries
                    if (item.get("file_name") or Path(item["vtp_file"]).name).lower() == "outlet.vtp"
                )
                inlet_key = inlet_item.get("s3_key") or prefix + "vtp/" + inlet_item["vtp_file"]
                outlet_key = outlet_item.get("s3_key") or prefix + "vtp/" + outlet_item["vtp_file"]
                try:
                    config = json.loads(row.get("run_config") or "{}")
                except json.JSONDecodeError:
                    config = {}
                objective_patches = set(config.get("objectivePatches") or [])
                body_entries = [
                    item for item in selected_entries
                    if (
                        item.get("patch_id") is not None
                        or (item.get("file_name") or Path(item["vtp_file"]).name).replace(".vtp", ".stl")
                        in objective_patches
                    )
                ]
                if not body_entries:
                    body_entries = [
                        item for item in selected_entries
                        if (item.get("file_name") or Path(item["vtp_file"]).name)
                        .lower().startswith("source-mesh-")
                    ]
            except client.exceptions.NoSuchKey:
                excluded = {
                    "inlet.vtp", "outlet.vtp", "lowerwall.vtp", "upperwall.vtp",
                    "sidewall.vtp", "sym.vtp", "frontandback.vtp", "walls.vtp",
                }
                body_entries = []
                paginator = client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "vtp/"):
                    for obj in page.get("Contents", []):
                        name = Path(obj["Key"]).name
                        if name.lower().endswith(".vtp") and name.lower() not in excluded:
                            body_entries.append({"vtp_file": name, "s3_key": obj["Key"]})
            if not body_entries:
                raise ValueError("manifest has no source body patches")

            if not target.exists() or args.overwrite:
                with tempfile.TemporaryDirectory(prefix=f"g1_{job}_") as temp:
                    temp = Path(temp)
                    inlet = read_polydata(download(client, bucket, inlet_key, temp / "inlet.vtp"))
                    outlet = read_polydata(download(client, bucket, outlet_key, temp / "outlet.vtp"))
                    polys = []
                    for item in body_entries:
                        key = item.get("s3_key") or prefix + "vtp/" + item["vtp_file"]
                        filename = item.get("file_name") or Path(item["vtp_file"]).name
                        polys.append(read_polydata(download(client, bucket, key, temp / filename)))

                    inlet_velocity = np.median(point_array(inlet, "UMean"), axis=0)
                    outlet_velocity = np.median(point_array(outlet, "UMean"), axis=0)
                    inlet_speed = float(np.linalg.norm(inlet_velocity))
                    outlet_speed = float(np.linalg.norm(outlet_velocity))
                    u_ref_vector = inlet_velocity if inlet_speed >= outlet_speed else outlet_velocity
                    u_ref = max(inlet_speed, outlet_speed)
                    if u_ref < 1e-6:
                        raise ValueError("inlet and outlet UMean are zero")
                    p_ref = float(np.median(point_array(outlet, "pMean")))
                    points, normals, pressure, velocity = merge_body(polys)
                    finite = (np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
                              & np.isfinite(pressure) & np.isfinite(velocity).all(axis=1))
                    points, normals = points[finite], normals[finite]
                    pressure, velocity = pressure[finite], velocity[finite]
                    lo, hi = points.min(axis=0), points.max(axis=0)
                    center = (lo + hi) / 2.0
                    scale = float(np.max(hi - lo))
                    if scale <= 0:
                        raise ValueError("body bounding box has zero scale")
                    rng = np.random.default_rng(number)
                    count = min(len(points), args.max_points)
                    select = rng.choice(len(points), count, replace=False)
                    points = ((points[select] - center) / scale).astype(np.float32)
                    normals = normals[select].astype(np.float32)
                    cp = ((pressure[select] - p_ref) / (0.5 * u_ref**2)).astype(np.float32)
                    normalized_velocity = (velocity[select] / u_ref).astype(np.float32)
                    np.savez_compressed(
                        target,
                        geometry_points=points,
                        geometry_normals=normals,
                        surface_points=points,
                        surface_cp=cp,
                        surface_velocity=normalized_velocity,
                        **{
                            f"{coefficient_name}_initial": (
                                np.float32(float(row["cd_initial"])) if row["cd_initial"] else np.float32(np.nan)
                            ),
                            f"{coefficient_name}_final": np.float32(float(row["cd_final"])),
                        },
                        center=center.astype(np.float32),
                        scale=np.float32(scale),
                        u_ref_vector=u_ref_vector.astype(np.float32),
                        p_ref=np.float32(p_ref),
                    )
            with np.load(target) as data:
                u_vec = data["u_ref_vector"].astype(float)
                scale = float(data["scale"])
                n_points = len(data["surface_points"])
            cases.append({
                "case_id": job,
                "group_id": row["project_uid"],
                "npz": str(target.relative_to(args.out_dir)),
                f"{coefficient_name}_initial": float(row["cd_initial"]) if row["cd_initial"] else None,
                f"{coefficient_name}_final": float(row["cd_final"]),
                "source_patches": [
                    item.get("file_name") or Path(item["vtp_file"]).name for item in body_entries
                ],
                "surface_points": n_points,
                "coefficient_expert": args.coefficient_expert,
                "coefficient_quality": "successful_solver_surface_fields",
                "smoke_test_case": row.get("test_case"),
                "conditions": {
                    "u_x": float(u_vec[0]), "u_y": float(u_vec[1]), "u_z": float(u_vec[2]),
                    "density": 1.225, "viscosity": 1.7894e-5, "temperature": 288.15,
                    "ref_length": scale, "ref_area": 1.0,
                },
            })
            print(f"[{number}/{len(rows)}] {job}: {n_points} points {coefficient_name.upper()}={row['cd_final']}")
        except Exception as exc:
            skipped.append({"job_uid": job, "error": str(exc)})
            print(f"[{number}/{len(rows)}] SKIP {job}: {exc}")

    payload = {
        "format": "g1-surface-v1",
        "source_csv": str(args.csv),
        "coefficient": coefficient_name,
        "selection": ("status=succeeded, objective_function=" + args.objective_function
                      + ", finite cd_final"),
        "cases": cases,
        "skipped": skipped,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"cases": len(cases), "skipped": len(skipped)}, indent=2))


if __name__ == "__main__":
    main()
