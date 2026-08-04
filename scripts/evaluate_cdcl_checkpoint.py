#!/usr/bin/env python3
"""Evaluate a G3 Cd/Cl checkpoint on its recorded train/validation splits.

Coefficient predictions are averaged over repeated geometry samples because the
PointNet encoder receives a random subset of surface points in production.  The
report includes per-case predictions, regression metrics, a train-median
baseline, and G2 validation-field errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from aox_g3.data.field_dataset import CONDITION_NAMES, FieldCase, load_field_manifest
from aox_g3.models.implicit_field import DragHead, ImplicitFieldNet, LiftHead


def _target(case: FieldCase, source: str) -> float:
    if source == "g2_cd":
        value = case.metadata.get("coefficients", {}).get("cd")
        return float(value) if value is not None else math.nan
    if source == "g2_cl":
        value = case.metadata.get("coefficients", {}).get("cl")
        return float(value) if value is not None else math.nan
    key = "cd_final" if source == "g1_cd" else "cl_final"
    with np.load(case.path) as data:
        return float(data[key])


def _sample_geometry(case: FieldCase, count: int, repeats: int, seed: int) -> np.ndarray:
    with np.load(case.path) as data:
        points = np.asarray(data["geometry_points"], dtype=np.float32)
        normals = np.asarray(data["geometry_normals"], dtype=np.float32)
    geometry = np.concatenate([points, normals], axis=1)
    rng = np.random.default_rng(seed)
    indices = np.stack([
        rng.choice(len(geometry), count, replace=len(geometry) < count)
        for _ in range(repeats)
    ])
    return geometry[indices]


def _predict_coefficient(
    model,
    head,
    case: FieldCase,
    condition_mean,
    condition_std,
    geometry_points: int,
    repeats: int,
    seed: int,
    device,
    transform,
) -> tuple[float, float]:
    geometry = torch.from_numpy(
        _sample_geometry(case, geometry_points, repeats, seed)
    ).to(device)
    condition = ((case.conditions - condition_mean) / condition_std).astype(np.float32)
    conditions = torch.from_numpy(np.repeat(condition[None], repeats, axis=0)).to(device)
    with torch.no_grad():
        latent, condition_latent = model.encode(geometry, conditions)
        normalized = head(latent, condition_latent).squeeze(1)
        values = transform(normalized).detach().cpu().numpy()
    return float(values.mean()), float(values.std())


def _metrics(rows: list[dict], train_targets: np.ndarray) -> dict:
    if not rows:
        return {"count": 0, "mae": math.nan, "rmse": math.nan, "r2": math.nan}
    target = np.asarray([row["target"] for row in rows], dtype=float)
    prediction = np.asarray([row["prediction"] for row in rows], dtype=float)
    error = prediction - target
    absolute = np.abs(error)
    mae = float(absolute.mean())
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = float(np.mean(np.abs(target)))
    relative_mae = mae / denominator if denominator > 1e-12 else math.nan
    sst = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / sst if sst > 1e-12 else math.nan
    if len(target) >= 2 and np.std(target) > 1e-12 and np.std(prediction) > 1e-12:
        correlation = float(np.corrcoef(target, prediction)[0, 1])
    else:
        correlation = math.nan
    baseline_value = float(np.median(train_targets))
    baseline_mae = float(np.mean(np.abs(target - baseline_value)))
    skill = 1.0 - mae / baseline_mae if baseline_mae > 1e-12 else math.nan
    return {
        "count": len(rows),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "mae": mae,
        "rmse": rmse,
        "relative_mae": relative_mae,
        "r2": r2,
        "pearson_r": correlation,
        "max_absolute_error": float(absolute.max()),
        "mean_sampling_std": float(np.mean([row["prediction_std"] for row in rows])),
        "within_0.01": float(np.mean(absolute <= 0.01)),
        "within_0.02": float(np.mean(absolute <= 0.02)),
        "within_0.05": float(np.mean(absolute <= 0.05)),
        "baseline": "training-target median",
        "baseline_value": baseline_value,
        "baseline_mae": baseline_mae,
        "skill_vs_baseline": skill,
    }


def _field_metrics(
    model,
    cases: list[FieldCase],
    condition_mean,
    condition_std,
    geometry_points: int,
    query_points: int,
    repeats: int,
    seed: int,
    device,
) -> dict:
    rows = []
    with torch.no_grad():
        for index, case in enumerate(cases):
            local_seed = seed + 900_001 + index * 10_007
            geometry = torch.from_numpy(
                _sample_geometry(case, geometry_points, repeats, local_seed)
            ).to(device)
            condition = ((case.conditions - condition_mean) / condition_std).astype(np.float32)
            conditions = torch.from_numpy(np.repeat(condition[None], repeats, axis=0)).to(device)
            with np.load(case.path) as data:
                rng = np.random.default_rng(local_seed + 1)
                indices = rng.choice(
                    len(data["volume_points"]), query_points,
                    replace=len(data["volume_points"]) < query_points,
                )
                points = np.asarray(data["volume_points"][indices], dtype=np.float32)
                target = np.concatenate([
                    data["volume_cp"][indices, None], data["volume_velocity"][indices]
                ], axis=1).astype(np.float32)
            query = torch.from_numpy(np.repeat(points[None], repeats, axis=0)).to(device)
            latent, condition_latent = model.encode(geometry, conditions)
            prediction = model.decode(latent, condition_latent, query).mean(dim=0).cpu().numpy()
            cp_mae = float(np.mean(np.abs(prediction[:, 0] - target[:, 0])))
            velocity_rmse = float(np.sqrt(np.mean((prediction[:, 1:] - target[:, 1:]) ** 2)))
            rows.append({
                "case_id": case.case_id,
                "cp_mae": cp_mae,
                "velocity_rmse_u_ref": velocity_rmse,
            })
    return {
        "count": len(rows),
        "cp_mae": float(np.mean([row["cp_mae"] for row in rows])),
        "velocity_rmse_u_ref": float(np.mean([row["velocity_rmse_u_ref"] for row in rows])),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--g2-manifest", type=Path, required=True)
    parser.add_argument("--g1-cd-manifest", type=Path)
    parser.add_argument("--g1-cl-manifest", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--geometry-points", type=int)
    parser.add_argument("--field-query-points", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device")
    parser.add_argument("--expert", default=None,
                        help="coefficient expert to evaluate; defaults to checkpoint primary")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    geometry_points = args.geometry_points or int(
        checkpoint.get("training", {}).get("geometry_points", 4096)
    )
    condition_mean = np.asarray(checkpoint["condition_mean"], dtype=np.float32)
    condition_std = np.asarray(checkpoint["condition_std"], dtype=np.float32)

    model = ImplicitFieldNet(**checkpoint["model_config"]).to(device)
    expert_name = args.expert or checkpoint.get("default_coefficient_expert")
    expert_specs = checkpoint.get("coefficient_experts", {})
    expert = expert_specs.get(expert_name) if expert_name else None
    if expert is None:
        expert_name = "legacy"
        expert = {
            "drag_head_state": checkpoint.get("drag_head_state"),
            "lift_head_state": checkpoint.get("lift_head_state"),
            "log_cd_mean": checkpoint.get("log_cd_mean", 0.0),
            "log_cd_std": checkpoint.get("log_cd_std", 1.0),
            "cl_mean": checkpoint.get("cl_mean", 0.0),
            "cl_std": checkpoint.get("cl_std", 1.0),
        }
    drag_head = DragHead().to(device) if expert.get("drag_head_state") else None
    lift_head = LiftHead().to(device) if expert.get("lift_head_state") else None
    model.load_state_dict(checkpoint["model_state"])
    if drag_head is not None:
        drag_head.load_state_dict(expert["drag_head_state"])
    if lift_head is not None:
        lift_head.load_state_dict(expert["lift_head_state"])
    model.eval()
    if drag_head is not None:
        drag_head.eval()
    if lift_head is not None:
        lift_head.eval()

    manifests = {"g2": load_field_manifest(args.g2_manifest)}
    if args.g1_cd_manifest:
        manifests["g1_cd"] = load_field_manifest(args.g1_cd_manifest)
    if args.g1_cl_manifest:
        manifests["g1_cl"] = load_field_manifest(args.g1_cl_manifest)
    cases_by_source = {key: {case.case_id: case for case in value} for key, value in manifests.items()}
    definitions = {}
    if drag_head is not None:
        definitions["g2_cd"] = (
            "g2", expert.get("train_cases", checkpoint["train_cases"]),
            expert.get("val_cases", checkpoint["val_cases"]), drag_head,
            lambda x: torch.exp(x * expert["log_cd_std"] + expert["log_cd_mean"]),
        )
    if lift_head is not None:
        definitions["g2_cl"] = (
            "g2", expert.get("train_cases", checkpoint["train_cases"]),
            expert.get("val_cases", checkpoint["val_cases"]), lift_head,
            lambda x: x * expert["cl_std"] + expert["cl_mean"],
        )
    if "g1_cd" in manifests and checkpoint.get("g1_train_cases"):
        definitions["g1_cd"] = (
            "g1_cd", checkpoint["g1_train_cases"], checkpoint["g1_val_cases"], drag_head,
            lambda x: torch.exp(x * checkpoint["log_cd_std"] + checkpoint["log_cd_mean"]),
        )
    if "g1_cl" in manifests and checkpoint.get("g1_cl_train_cases"):
        definitions["g1_cl"] = (
            "g1_cl", checkpoint["g1_cl_train_cases"], checkpoint["g1_cl_val_cases"], lift_head,
            lambda x: x * checkpoint["cl_std"] + checkpoint["cl_mean"],
        )

    all_rows: list[dict] = []
    summary = {}
    for source_index, (source, definition) in enumerate(definitions.items()):
        manifest_key, train_ids, val_ids, head, transform = definition
        source_cases = cases_by_source[manifest_key]
        train_targets = np.asarray([
            _target(source_cases[cid], source) for cid in train_ids
            if np.isfinite(_target(source_cases[cid], source))
        ])
        summary[source] = {}
        for split, ids in (("train", train_ids), ("validation", val_ids)):
            rows = []
            for case_index, case_id in enumerate(ids):
                case = source_cases[case_id]
                target = _target(case, source)
                if not np.isfinite(target):
                    continue
                prediction, prediction_std = _predict_coefficient(
                    model, head, case, condition_mean, condition_std,
                    geometry_points, args.repeats,
                    args.seed + source_index * 1_000_003 + case_index * 10_007,
                    device, transform,
                )
                error = prediction - target
                row = {
                    "dataset": source,
                    "split": split,
                    "case_id": case_id,
                    "target": target,
                    "prediction": prediction,
                    "error": error,
                    "absolute_error": abs(error),
                    "relative_error": abs(error) / abs(target) if abs(target) > 1e-12 else math.nan,
                    "prediction_std": prediction_std,
                }
                rows.append(row)
                all_rows.append(row)
            summary[source][split] = _metrics(rows, train_targets)

    g2_val = [cases_by_source["g2"][case_id] for case_id in checkpoint["val_cases"]]
    field_metrics = _field_metrics(
        model, g2_val, condition_mean, condition_std, geometry_points,
        args.field_query_points, args.repeats, args.seed, device,
    )
    rejected_rows = []
    if drag_head is not None:
        for case_index, case in enumerate(manifests["g2"]):
            raw = case.metadata.get("raw_coefficients_rejected", {}).get("cd")
            try:
                raw = float(raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(raw):
                continue
            prediction, prediction_std = _predict_coefficient(
                model, drag_head, case, condition_mean, condition_std,
                geometry_points, args.repeats, args.seed + 8_000_000 + case_index,
                device,
                lambda x: torch.exp(x * expert["log_cd_std"] + expert["log_cd_mean"]),
            )
            rejected_rows.append({
                "case_id": case.case_id,
                "group_id": case.group_id,
                "rejected_raw_cd": raw,
                "prediction": prediction,
                "prediction_std": prediction_std,
                "rejection_reason": case.metadata.get("coefficient_quality"),
            })
    report = {
        "checkpoint": str(args.checkpoint),
        "best_epoch": checkpoint.get("best_epoch"),
        "coefficient_expert": expert_name,
        "device": str(device),
        "geometry_points": geometry_points,
        "geometry_repeats": args.repeats,
        "summary": summary,
        "g2_validation_fields": field_metrics,
        "rejected_coefficient_audit": rejected_rows,
        "predictions": all_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=True))
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    print(json.dumps({"summary": summary, "g2_validation_fields": field_metrics}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
