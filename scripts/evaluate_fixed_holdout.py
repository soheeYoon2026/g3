#!/usr/bin/env python3
"""Evaluate G3 checkpoints on frozen manifests shared by all candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aox_g3.train_coefficient_experts import load_cases


def metrics(targets: list[float], predictions: list[float]) -> dict:
    target = np.asarray(targets, dtype=float)
    prediction = np.asarray(predictions, dtype=float)
    error = prediction - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = (
        1.0 - float(np.sum(error**2)) / denominator
        if len(target) >= 2 and denominator > 1e-12
        else math.nan
    )
    return {
        "count": len(target),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "max_absolute_error": float(np.max(np.abs(error))),
    }


def main():
    import torch

    from aox_g3.models.implicit_field import DragHead, ImplicitFieldNet, LiftHead

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        nargs=3,
        action="append",
        metavar=("LABEL", "EXPERT", "MANIFEST"),
        required=True,
    )
    parser.add_argument("--geometry-points", type=int, default=4096)
    parser.add_argument(
        "--field-query-points",
        type=int,
        default=0,
        help="also evaluate Cp/velocity fields using this many volume points per case",
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device")
    parser.add_argument(
        "--include-predictions",
        action="store_true",
        help="include per-case targets, predictions, and signed errors in the report",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ImplicitFieldNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    cond_mean = np.asarray(checkpoint["condition_mean"], np.float32)
    cond_std = np.asarray(checkpoint["condition_std"], np.float32)
    expert_specs = checkpoint.get("coefficient_experts", {})
    report = {"checkpoint": str(args.checkpoint.resolve()), "metrics": {"datasets": {}}}

    for dataset_index, (label, expert_name, manifest) in enumerate(args.dataset):
        if expert_name not in expert_specs:
            raise KeyError(f"checkpoint has no coefficient expert {expert_name!r}")
        expert = expert_specs[expert_name]
        drag = DragHead().to(device) if expert.get("drag_head_state") else None
        lift = LiftHead().to(device) if expert.get("lift_head_state") else None
        if drag is not None:
            drag.load_state_dict(expert["drag_head_state"])
            drag.eval()
        if lift is not None:
            lift.load_state_dict(expert["lift_head_state"])
            lift.eval()

        cd_targets, cd_predictions, cl_targets, cl_predictions = [], [], [], []
        field_cp_mae, field_velocity_rmse = [], []
        prediction_rows = []
        cases = load_cases(Path(manifest))
        with torch.no_grad():
            for case_index, case in enumerate(cases):
                row = {"case_id": case.case_id, "group_id": case.group_id}
                with np.load(case.path) as data:
                    points = np.asarray(data["geometry_points"], np.float32)
                    normals = np.asarray(data["geometry_normals"], np.float32)
                geometry = np.concatenate([points, normals], axis=1)
                rng = np.random.default_rng(
                    args.seed + dataset_index * 1_000_003 + case_index * 10_007
                )
                indices = np.stack([
                    rng.choice(
                        len(geometry),
                        args.geometry_points,
                        replace=len(geometry) < args.geometry_points,
                    )
                    for _ in range(args.repeats)
                ])
                geometry_tensor = torch.from_numpy(geometry[indices]).to(device)
                condition = ((case.conditions - cond_mean) / cond_std).astype(np.float32)
                condition_tensor = torch.from_numpy(
                    np.repeat(condition[None], args.repeats, axis=0)
                ).to(device)
                latent, condition_latent = model.encode(
                    geometry_tensor, condition_tensor
                )
                if args.field_query_points:
                    with np.load(case.path) as data:
                        field_keys = {
                            "volume_points", "volume_cp", "volume_velocity"
                        }
                        if field_keys.issubset(data.files):
                            field_indices = rng.choice(
                                len(data["volume_points"]),
                                args.field_query_points,
                                replace=len(data["volume_points"])
                                < args.field_query_points,
                            )
                            query_points = np.asarray(
                                data["volume_points"][field_indices], np.float32
                            )
                            field_target = np.concatenate(
                                [
                                    np.asarray(
                                        data["volume_cp"][field_indices, None],
                                        np.float32,
                                    ),
                                    np.asarray(
                                        data["volume_velocity"][field_indices],
                                        np.float32,
                                    ),
                                ],
                                axis=1,
                            )
                        else:
                            query_points = field_target = None
                    if query_points is not None:
                        query_tensor = torch.from_numpy(
                            np.repeat(
                                query_points[None], args.repeats, axis=0
                            )
                        ).to(device)
                        field_prediction = (
                            model.decode(
                                latent, condition_latent, query_tensor
                            )
                            .mean(dim=0)
                            .cpu()
                            .numpy()
                        )
                        cp_mae = float(
                            np.mean(
                                np.abs(
                                    field_prediction[:, 0]
                                    - field_target[:, 0]
                                )
                            )
                        )
                        velocity_rmse = float(
                            np.sqrt(
                                np.mean(
                                    (
                                        field_prediction[:, 1:]
                                        - field_target[:, 1:]
                                    )
                                    ** 2
                                )
                            )
                        )
                        field_cp_mae.append(cp_mae)
                        field_velocity_rmse.append(velocity_rmse)
                        row["fields"] = {
                            "cp_mae": cp_mae,
                            "velocity_rmse_u_ref": velocity_rmse,
                        }
                if drag is not None and np.isfinite(case.cd) and case.cd > 0:
                    prediction = torch.exp(
                        drag(latent, condition_latent)
                        * expert["log_cd_std"]
                        + expert["log_cd_mean"]
                    )
                    cd_targets.append(case.cd)
                    cd_prediction = float(prediction.mean().cpu())
                    cd_predictions.append(cd_prediction)
                    row["cd"] = {
                        "target": case.cd,
                        "prediction": cd_prediction,
                        "error": cd_prediction - case.cd,
                    }
                if lift is not None and np.isfinite(case.cl):
                    prediction = (
                        lift(latent, condition_latent) * expert["cl_std"]
                        + expert["cl_mean"]
                    )
                    cl_targets.append(case.cl)
                    cl_prediction = float(prediction.mean().cpu())
                    cl_predictions.append(cl_prediction)
                    row["cl"] = {
                        "target": case.cl,
                        "prediction": cl_prediction,
                        "error": cl_prediction - case.cl,
                    }
                if args.include_predictions:
                    prediction_rows.append(row)

        dataset_report = {
            "expert": expert_name,
            "manifest": str(Path(manifest).resolve()),
            "cases": len(cases),
        }
        if cd_targets:
            dataset_report["cd"] = metrics(cd_targets, cd_predictions)
        if cl_targets:
            dataset_report["cl"] = metrics(cl_targets, cl_predictions)
        if field_cp_mae:
            dataset_report["fields"] = {
                "count": len(field_cp_mae),
                "cp_mae": float(np.mean(field_cp_mae)),
                "velocity_rmse_u_ref": float(
                    np.mean(field_velocity_rmse)
                ),
            }
        if args.include_predictions:
            dataset_report["predictions"] = prediction_rows
        report["metrics"]["datasets"][label] = dataset_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report["metrics"], indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
