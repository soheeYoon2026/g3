#!/usr/bin/env python3
"""Build a Cd-only active-learning table from G3 recommendations and G2 results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import boto3


SUPPLEMENTAL = {
    "dxBrBDidEIwok2yGtlktAf": "point_18_in_scale_0.5",
    "sVXw08RsNaJPNJDKj2fwPf": "point_18_out_scale_0.5",
    "me6LuM82ubCbuEuXDiLA2I": "point_26_in_scale_0.5",
    "nDnaYIkTM8hV6dy63CwDOT": "point_26_out_scale_0.5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/ubuntu/g3-v2"))
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def split_label(label: str) -> tuple[int, str, float]:
    parts = label.split("_")
    control_id = int(parts[1])
    direction = parts[2]
    scale = float(parts[4]) if len(parts) >= 5 and parts[3] == "scale" else 1.0
    return control_id, direction, scale


def read_converged_cd(s3, bucket: str, key: str) -> float:
    text = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode(errors="replace")
    rows = list(csv.DictReader(text.splitlines(), skipinitialspace=True))
    if not rows:
        raise ValueError(f"No CFD history rows: s3://{bucket}/{key}")
    cd_key = next(name for name in rows[-1] if name.strip() == "CD")
    return float(rows[-1][cd_key])


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = (args.out or root / "var/active-learning/20260821-cd-only").resolve()
    out.mkdir(parents=True, exist_ok=True)

    comparison_path = root / "var/active-learning/fullres-comparison-all.json"
    recommendation_dir = root / "var/active-learning/20260820-workbench-recommendations"
    recommendation_path = recommendation_dir / "recommendations.submit.json"
    comparison = json.loads(comparison_path.read_text())
    recommendations = json.loads(recommendation_path.read_text())
    recommendation_by_label = {row["label"]: row for row in recommendations["recommendations"]}
    recommendation_by_control = {row["control_id"]: row for row in recommendations["recommendations"]}
    baseline_row = next(row for row in comparison["rows"] if row["label"] == "baseline")
    baseline_cd = float(baseline_row["g2_cd"])

    samples: list[dict] = []
    rejected: list[dict] = []
    for result in comparison["rows"]:
        label = result["label"]
        if label == "baseline":
            continue
        if result.get("status") != "succeeded" or result.get("g2_cd") is None:
            rejected.append({"label": label, "job_uid": result.get("job_uid"), "reason": "missing_g2_cd"})
            continue
        recommendation = recommendation_by_label[label]
        control_id, direction, scale = split_label(label)
        g2_cd = float(result["g2_cd"])
        samples.append({
            "label": label,
            "control_id": control_id,
            "direction": direction,
            "amplitude_scale": scale,
            "position": recommendation["position"],
            "displacement": recommendation["displacement"],
            "influence_radius": recommendation["influence_radius"],
            "symmetric": bool(recommendation.get("symmetric", False)),
            "baseline_g2_cd": baseline_cd,
            "g2_cd": g2_cd,
            "g2_delta_cd": g2_cd - baseline_cd,
            "predicted_cd": result.get("predicted_cd"),
            "predicted_delta_cd": result.get("predicted_delta_cd"),
            "job_uid": result["job_uid"],
            "candidate_sha256": result["candidate_sha256"],
            "geometry_kind": "merged_stl",
            "geometry_uri": str((recommendation_dir / recommendation["stl"]).resolve()),
            "source": "full_amplitude_g2",
        })

    s3 = boto3.client("s3")
    bucket = baseline_row["s3_bucket"]
    project_prefix = baseline_row["s3_output_prefix"].split("/job/", 1)[0]
    supplemental_metadata = {
        "point_18_in_scale_0.5": {
            "position": [2.2120931, 0.8495653, 0.9636704],
            "displacement": [-0.0049337, 0.00056525, -0.00267175],
        },
        "point_18_out_scale_0.5": {
            "position": [2.2120931, 0.8495653, 0.9636704],
            "displacement": [0.0049337, -0.00056525, 0.00267175],
        },
        "point_26_in_scale_0.5": {
            "position": [-1.8588037, 0.4978723, 0.4298106],
            "displacement": [-0.00337715, -0.00446015, 0.00070785],
        },
        "point_26_out_scale_0.5": {
            "position": [-1.8588037, 0.4978723, 0.4298106],
            "displacement": [0.00337715, 0.00446015, -0.00070785],
        },
    }
    for job_uid, label in SUPPLEMENTAL.items():
        control_id, direction, scale = split_label(label)
        job_prefix = f"{project_prefix}/job/{job_uid}"
        g2_cd = read_converged_cd(s3, bucket, f"{job_prefix}/output/history.csv")
        metadata = supplemental_metadata[label]
        samples.append({
            "label": label,
            "control_id": control_id,
            "direction": direction,
            "amplitude_scale": scale,
            "position": metadata["position"],
            "displacement": metadata["displacement"],
            "influence_radius": recommendation_by_control[control_id]["influence_radius"],
            "symmetric": True,
            "baseline_g2_cd": baseline_cd,
            "g2_cd": g2_cd,
            "g2_delta_cd": g2_cd - baseline_cd,
            "predicted_cd": None,
            "predicted_delta_cd": None,
            "job_uid": job_uid,
            "candidate_sha256": None,
            "geometry_kind": "multipart_s3",
            "geometry_uri": f"s3://{bucket}/{job_prefix}/input/stl/",
            "source": "half_amplitude_g2",
        })

    samples = [row for row in samples if math.isfinite(row["g2_cd"]) and 0 < row["g2_cd"] < 2]
    controls = sorted({row["control_id"] for row in samples}, key=lambda value: hashlib.sha256(f"g3-cd-v1:{value}".encode()).hexdigest())
    train_end = max(1, int(len(controls) * 0.70))
    validation_end = min(len(controls) - 1, train_end + max(1, int(len(controls) * 0.15)))
    split_by_control = {
        control_id: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, control_id in enumerate(controls)
    }
    for row in samples:
        row["split"] = split_by_control[row["control_id"]]
    samples.sort(key=lambda row: (row["split"], row["control_id"], row["direction"], row["amplitude_scale"]))

    fields = [
        "split", "label", "control_id", "direction", "amplitude_scale", "baseline_g2_cd", "g2_cd",
        "g2_delta_cd", "predicted_cd", "predicted_delta_cd", "job_uid", "candidate_sha256",
        "geometry_kind", "geometry_uri", "position", "displacement", "influence_radius", "symmetric", "source",
    ]
    with (out / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)
    inventory_fields = [
        "solver", "job_status", "job_uid", "project_uid", "s3_bucket",
        "s3_output_prefix", "output_s3_key", "variant_name",
    ]
    project_uid = project_prefix.rsplit("/project/", 1)[1]
    inventory_rows = [{
        "solver": "G2",
        "job_status": "succeeded",
        "job_uid": baseline_row["job_uid"],
        "project_uid": project_uid,
        "s3_bucket": bucket,
        "s3_output_prefix": baseline_row["s3_output_prefix"],
        "output_s3_key": baseline_row["s3_output_prefix"],
        "variant_name": "baseline",
    }]
    inventory_rows.extend({
        "solver": "G2",
        "job_status": "succeeded",
        "job_uid": row["job_uid"],
        "project_uid": project_uid,
        "s3_bucket": bucket,
        "s3_output_prefix": f"{project_prefix}/job/{row['job_uid']}/output",
        "output_s3_key": f"{project_prefix}/job/{row['job_uid']}/output",
        "variant_name": row["label"],
    } for row in samples)
    with (out / "inventory.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=inventory_fields)
        writer.writeheader()
        writer.writerows(inventory_rows)
    (out / "dataset.json").write_text(json.dumps({
        "schema_version": 1,
        "objective": "delta_cd_only",
        "baseline": {
            "label": "baseline",
            "job_uid": baseline_row["job_uid"],
            "g2_cd": baseline_cd,
            "geometry_sha256": baseline_row["candidate_sha256"],
        },
        "samples": samples,
        "rejected": rejected,
    }, indent=2))
    for split in ("train", "validation", "test"):
        rows = [row for row in samples if row["split"] == split]
        (out / f"{split}.json").write_text(json.dumps(rows, indent=2))
    summary = {
        "baseline_g2_cd": baseline_cd,
        "valid_transform_samples": len(samples),
        "rejected_original_jobs": rejected,
        "unique_control_points": len(controls),
        "splits": {split: sum(row["split"] == split for row in samples) for split in ("train", "validation", "test")},
        "split_control_points": {
            split: sorted({row["control_id"] for row in samples if row["split"] == split})
            for split in ("train", "validation", "test")
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
