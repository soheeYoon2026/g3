#!/usr/bin/env python3
"""Incrementally materialize quality-gated G2 S3 results for DoMINO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import boto3

from aox_g3.upload_gate import classify_case, classify_stl
from prepare_domino_su2_v3 import convert_case
from prepare_smoke_g2_s3 import case_files, list_keys, normalize_inventory_row
from training_event_inventory import load_event_rows
from validate_domino_su2_v3 import integrate_case


def parse_rbf_objectives(text: str) -> dict[str, float]:
    """Map G2 optimization rows to the matching RBF_DSN result folders."""
    rows = csv.DictReader(text.splitlines())
    objectives = {}
    for row in rows:
        try:
            iteration = int(row["iteration"])
            objective = float(row["objective"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(objective):
            objectives[f"RBF_DSN_{iteration:03d}"] = objective
    return objectives


def load_rbf_objectives(client, bucket: str, prefix: str, keys: list[str]) -> dict[str, float]:
    key = prefix.rstrip("/") + "/rbf_optimization_history.csv"
    if key not in keys:
        return {}
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return parse_rbf_objectives(body.decode("utf-8-sig", "replace"))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_finite_coefficients(case_dir: Path, run: str | int, integrated: dict) -> dict:
    """Persist the quality-gated coefficient labels consumed by the trainer."""
    path = case_dir / f"conditions_{run}.json"
    conditions = json.loads(path.read_text())
    for key in ("su2_cd", "su2_cl"):
        value = float(integrated[key])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {key}")
        conditions[key] = value
    atomic_json(path, conditions)
    return conditions


def seed_dataset(base: Path, out: Path, max_cd_error: float, max_cl_error: float) -> dict:
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads((base / "manifest.json").read_text())
    cases, attempts = [], []
    for row in source.get("cases", []):
        if not row.get("accepted"):
            continue
        old = str(row["run"])
        source_id = f"accepted_seed:{old}"
        try:
            integrated = integrate_case(base / f"run_{old}")
            if not all(math.isfinite(float(integrated[key])) for key in ("cd", "cl", "su2_cd", "su2_cl")):
                raise ValueError("non-finite integrated or SU2 coefficient")
            cd_scale = max(abs(float(integrated["su2_cd"])), 1e-3)
            cd_error = abs(integrated["cd"] - integrated["su2_cd"]) / cd_scale
            cl_error = abs(integrated["cl"] - integrated["su2_cl"])
            if cd_error > max_cd_error:
                raise ValueError(f"Cd integration relative error {cd_error:.6g}")
            if cl_error > max_cl_error:
                raise ValueError(f"Cl integration absolute error {cl_error:.6g}")
        except Exception as exc:
            attempts.append({"source_id": source_id, "status": "rejected",
                             "reason": f"{type(exc).__name__}: {exc}"})
            continue
        run = len(cases) + 1
        destination = out / f"run_{run}"
        destination.mkdir()
        for prefix, suffix in (("boundary", "vtp"), ("drivaer", "stl"), ("conditions", "json")):
            shutil.copy2(base / f"run_{old}" / f"{prefix}_{old}.{suffix}", destination / f"{prefix}_{run}.{suffix}")
        conditions = write_finite_coefficients(destination, run, integrated)
        copied = dict(row)
        copied.update({"run": run, "source_kind": "reaudited_seed", "source_run": old,
                       "conditions": conditions,
                       "integrated": integrated, "cd_relative_error": cd_error,
                       "cl_absolute_error": cl_error})
        cases.append(copied)
        attempts.append({"source_id": source_id, "status": "accepted", "run": run})
    payload = {"schema_version": 2, "format": "domino-g2-incremental-v1", "base": str(base),
               "cases": cases, "attempts": attempts, "summary": {}}
    atomic_json(manifest_path, payload)
    return payload


def input_rows(args) -> list[dict]:
    raw = []
    if args.event_inventory:
        raw.extend(load_event_rows(args.event_inventory))
    if args.csv:
        for path in args.csv:
            raw.extend(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))
    return [normalize_inventory_row(row) for row in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--event-inventory", action="append", type=Path)
    parser.add_argument("--csv", action="append", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cd-relative-error", type=float, default=0.03)
    parser.add_argument("--require-car-case", action="store_true",
                        help="reject cases the upload gate does not call a road-car case")
    parser.add_argument("--max-cl-absolute-error", type=float, default=0.03)
    parser.add_argument("--no-seed", action="store_true",
                        help="build a dedicated dataset without copying the historical seed")
    parser.add_argument("--include-design", action="append",
                        help="only materialize matching design tags (repeatable)")
    args = parser.parse_args()
    if not args.event_inventory and not args.csv:
        parser.error("provide --event-inventory and/or --csv")

    if args.no_seed:
        args.out.mkdir(parents=True, exist_ok=True)
        manifest_path = args.out / "manifest.json"
        payload = (json.loads(manifest_path.read_text()) if manifest_path.exists() else
                   {"schema_version": 2, "format": "domino-g2-dedicated-v1",
                    "base": None, "cases": [], "attempts": [], "summary": {}})
    else:
        payload = seed_dataset(args.base, args.out, args.max_cd_relative_error, args.max_cl_absolute_error)
    cases, attempts = payload["cases"], payload.setdefault("attempts", [])
    attempted = {row["source_id"] for row in attempts}
    geometry_digests = {row.get("geometry_digest") for row in cases if row.get("geometry_digest")}
    rows = [row for row in input_rows(args) if row.get("solver") == "G2" and row.get("job_status") == "succeeded"]
    clients, added = {}, 0

    for inventory in rows:
        bucket, prefix = inventory["s3_bucket"], inventory["s3_output_prefix"]
        client = clients.setdefault(bucket, boto3.client("s3"))
        keys = list_keys(client, bucket, prefix)
        available = case_files(prefix, keys)
        rbf_objectives = load_rbf_objectives(client, bucket, prefix, keys)
        for design, files in sorted(available.items()):
            if args.include_design and design not in set(args.include_design):
                continue
            source_id = f"{inventory['job_uid']}:{design}"
            source_attempts = [row for row in attempts if row.get("source_id") == source_id]
            retry_with_objective = (
                design in rbf_objectives
                and not any(row.get("status") == "accepted" for row in source_attempts)
                and any(
                    row.get("reason") == "ValueError: no finite converged SU2 Cd"
                    for row in source_attempts
                )
            )
            if source_id in attempted and not retry_with_objective:
                continue
            result = {"source_id": source_id, "bucket": bucket, "output_prefix": prefix}
            try:
                with tempfile.TemporaryDirectory(prefix="domino-g2-") as raw_temp:
                    temp = Path(raw_temp)
                    for name, key in files.items():
                        if name == "surface_flow.vtu" or name == "history.csv" or name.endswith("_CFD.cfg"):
                            client.download_file(bucket, key, str(temp / name))
                    vtu = temp / "surface_flow.vtu"
                    cfg = next(temp.glob("*_CFD.cfg"))
                    digest = file_digest(vtu)
                    if digest in geometry_digests:
                        raise ValueError("duplicate geometry/result artifact")
                    run = len(cases) + 1
                    with tempfile.TemporaryDirectory(prefix="domino-convert-") as converted:
                        converted_root = Path(converted)
                        conditions = convert_case(vtu, cfg, converted_root, str(run))
                        integrated = integrate_case(converted_root / f"run_{run}")
                        if design in rbf_objectives:
                            integrated["su2_cd"] = rbf_objectives[design]
                            integrated["su2_cd_source"] = "rbf_optimization_history.csv"
                            # G2 does not persist a primal coefficient history for each
                            # RBF candidate. Cl is still physically integrated from its
                            # saved surface field, but has no independent history value.
                            if not math.isfinite(integrated["su2_cl"]):
                                integrated["su2_cl"] = integrated["cl"]
                                integrated["su2_cl_source"] = "surface_field_only"
                        if not math.isfinite(integrated["su2_cd"]):
                            raise ValueError("no finite converged SU2 Cd")
                        scale = max(abs(integrated["su2_cd"]), 1e-3)
                        relative_error = abs(integrated["cd"] - integrated["su2_cd"]) / scale
                        if not math.isfinite(relative_error) or relative_error > args.max_cd_relative_error:
                            raise ValueError(f"Cd integration relative error {relative_error:.6g}")
                        if not math.isfinite(integrated["su2_cl"]):
                            raise ValueError("no finite converged SU2 Cl")
                        cl_error = abs(integrated["cl"] - integrated["su2_cl"])
                        if not math.isfinite(cl_error) or cl_error > args.max_cl_absolute_error:
                            raise ValueError(f"Cl integration absolute error {cl_error:.6g}")
                        conditions = write_finite_coefficients(converted_root / f"run_{run}", run, integrated)
                        shutil.move(str(converted_root / f"run_{run}"), str(args.out / f"run_{run}"))
                    # Tag the shape class so split builders can keep components and
                    # off-regime cases out of car training/eval (2026-08-26 reaudit:
                    # transonic wings had polluted the "unseen car" gate).
                    try:
                        geometry_class = classify_stl(str(args.out / f"run_{run}" / f"drivaer_{run}.stl"))
                        geometry_class.pop("path", None)
                    except Exception as exc:
                        geometry_class = {"verdict": "unsure", "reasons": [f"gate error: {exc}"]}
                    case_class = classify_case(conditions, geometry_class)
                    if args.require_car_case and case_class["case_class"] != "car_case":
                        shutil.rmtree(args.out / f"run_{run}", ignore_errors=True)
                        raise ValueError(
                            f"not a road-car case: {case_class['case_class']} "
                            f"({'; '.join(case_class['flow_reasons'] + case_class['geometry_reasons'])[:120]})")
                    row = {"run": run, "group_id": f"artifact_{digest[:16]}", "geometry_digest": digest,
                           "conditions": conditions, "integrated": integrated, "cd_relative_error": relative_error,
                           "cl_absolute_error": cl_error, "shape_class": case_class,
                           "accepted": True, "source_kind": "s3_g2", "source": result}
                    cases.append(row)
                    geometry_digests.add(digest)
                    result.update({"status": "accepted", "run": run})
                    added += 1
            except Exception as exc:
                result.update({"status": "rejected", "reason": f"{type(exc).__name__}: {exc}"})
            attempts.append(result)
            attempted.add(source_id)
            atomic_json(args.out / "manifest.json", payload)

    class_counts = {}
    for row in cases:
        key = (row.get("shape_class") or {}).get("case_class", "untagged")
        class_counts[key] = class_counts.get(key, 0) + 1
    payload["summary"] = {"accepted": len(cases), "attempted_sources": len(attempts),
                          "shape_classes": class_counts,
                          "accepted_this_run": added,
                          "rejected_sources": sum(row["status"] == "rejected" for row in attempts),
                          "unique_groups": len({row["group_id"] for row in cases})}
    atomic_json(args.out / "manifest.json", payload)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
