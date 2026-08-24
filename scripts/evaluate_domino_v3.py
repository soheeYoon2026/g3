#!/usr/bin/env python
"""Evaluate pretrained and fine-tuned DoMINO checkpoints on v3 cases."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aox_g3.eval_metrics import coefficient_summary, load_pairs


def forward_surface(model, data):
    smin, smax = data["surface_min_max"][:, 0], data["surface_min_max"][:, 1]
    geometry = 2.0 * (data["geometry_coordinates"] - smin) / (smax - smin) - 1
    encoding = model.geo_rep_surface(geometry, data["surf_grid"], data["sdf_surf_grid"])
    local = model.surface_local_geo_encodings(
        0.5 * encoding, data["surface_mesh_centers"], data["surf_grid"]
    )
    position = model.fc_p_surf(data["pos_surface_center_of_mass"])
    return model.solution_calculator_surf(
        data["surface_mesh_centers"], local, position,
        data["surface_mesh_neighbors"], data["surface_normals"],
        data["surface_neighbors_normals"], data["surface_areas"].unsqueeze(-1),
        data["surface_neighbors_areas"].unsqueeze(-1),
        data["global_params_values"], data["global_params_reference"],
    )


def integrate_prediction(prediction, data, factors, conditions):
    from physicsnemo.models.domino.utils import unnormalize

    fields = unnormalize(prediction, factors[0], factors[1]) * (2.0 * conditions["density"])
    centers = data["surface_mesh_centers"][0]
    normals = data["surface_normals"][0]
    areas = data["surface_areas"][0]
    volume_sign = torch.sum(torch.sum(centers * normals, dim=1) * areas)
    if volume_sign < 0:
        normals = -normals
    cp, cf = fields[:, 0], fields[:, 1:4]
    pressure = torch.sum((-cp[:, None] * normals) * areas[:, None], dim=0)
    friction = -torch.sum(cf * areas[:, None], dim=0)
    force = (pressure + friction) / float(conditions["ref_area"])
    return float(force[0]), float(force[2])


def main():
    from huggingface_hub import snapshot_download
    from physicsnemo.cfd.evaluation.datasets.adapters.drivaerml import DrivAerMLAdapter
    from physicsnemo.cfd.evaluation.models.wrappers.domino.wrapper import DominoWrapper

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fine-tuned", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--partition", choices=("train", "test"), default="test")
    parser.add_argument("--pairs", type=Path, help="explicit baseline/variant pair manifest for ΔCd metrics")
    parser.add_argument("--direction-tolerance", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    pairs = load_pairs(args.pairs) if args.pairs else None

    snapshot = Path(snapshot_download("nvidia/domino_drivaerml"))
    surface = snapshot / "domino_drivaerml_surface_checkpoint"
    wrapper = DominoWrapper().load(
        checkpoint_path=str(surface / "DoMINO.0.501.mdlus"),
        stats_path=str(surface / "scaling_factors.pkl"),
        device=args.device,
        domino_config=str(surface / "config.yaml"),
    )
    model, factors = wrapper._model, wrapper._surf_factors
    pretrained = {key: value.detach().clone() for key, value in model.state_dict().items()}
    adapter = DrivAerMLAdapter(root=str(args.root))
    cases = adapter.list_cases()
    groups = {}
    if args.split:
        split = json.loads(args.split.read_text())
        groups = {
            f"run_{row['run']}": row.get("group_id") for row in split[f"{args.partition}_cases"]
        }
        cases = [case_id for case_id in cases if case_id in groups]

    def evaluate(label, state):
        model.load_state_dict(state)
        model.eval()
        rows = []
        for case_id in cases:
            suffix = case_id.removeprefix("run_")
            conditions = json.loads(
                (args.root / case_id / f"conditions_{suffix}.json").read_text()
            )
            data = wrapper.prepare_inputs(adapter.load_case(case_id))["data_dict"]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                prediction = forward_surface(model, data)[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            cd, cl = integrate_prediction(prediction, data, factors, conditions)
            row = {
                "model": label, "case": case_id, "run": suffix,
                "group_id": groups.get(case_id),
                "pred_cd": cd, "true_cd": conditions["su2_cd"],
                "pred_cl": cl, "true_cl": conditions["su2_cl"],
                "cd_abs_error": abs(cd - conditions["su2_cd"]),
                "cl_abs_error": abs(cl - conditions["su2_cl"]),
                "inference_seconds": elapsed,
            }
            rows.append(row)
            print(json.dumps(row))
        summary = {"model": label}
        summary.update(coefficient_summary(
            rows, pairs=pairs, case_key="run",
            direction_tolerance=args.direction_tolerance,
        ))
        summary["mean_inference_seconds"] = float(
            np.mean([row["inference_seconds"] for row in rows])
        )
        print(json.dumps(summary))

    evaluate("pretrained", pretrained)
    evaluate("fine_tuned", torch.load(args.fine_tuned, map_location=args.device))


if __name__ == "__main__":
    main()
