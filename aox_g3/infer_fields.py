"""Infer surface pressure and a 3-D velocity field from an STL."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .data.field_dataset import CONDITION_NAMES
from .geometry.stl_sampler import load_mesh, sample_surface
from .geometry.surface_sampling import GEOMETRY_PREPROCESSING_VERSION


def _conditions(args) -> tuple[np.ndarray, float, float]:
    velocity = np.asarray([args.u_x, args.u_y, args.u_z], dtype=np.float32)
    u_ref = float(np.linalg.norm(velocity)) or 1.0
    values = {
        "u_x": args.u_x, "u_y": args.u_y, "u_z": args.u_z,
        "density": args.density, "viscosity": args.viscosity,
        "temperature": args.temperature, "ref_length": args.ref_length,
        "ref_area": args.ref_area,
    }
    return np.asarray([values[name] for name in CONDITION_NAMES], np.float32), u_ref, 0.5 * args.density * u_ref**2


def _predict_chunks(model, latent, cond_latent, points, device, chunk_size):
    import torch
    outputs = []
    with torch.no_grad():
        for start in range(0, len(points), chunk_size):
            query = torch.from_numpy(points[start:start + chunk_size][None].astype(np.float32)).to(device)
            outputs.append(model.decode(latent, cond_latent, query)[0].cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main(argv=None):
    import torch

    from .field_io import render_png, write_streamlines, write_surface_vtp, write_volume_vti
    from .models.implicit_field import DragHead, ImplicitFieldNet, LiftHead

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", default="g3_field_prediction")
    parser.add_argument("--grid", type=int, nargs=3, default=(96, 64, 48), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--bounds", type=float, nargs=6,
                        default=(-1.5, 3.0, -1.25, 1.25, -0.6, 1.6),
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        help="query bounds in body-normalized coordinates")
    parser.add_argument("--geometry-points", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--u-x", type=float, default=30.0)
    parser.add_argument("--u-y", type=float, default=0.0)
    parser.add_argument("--u-z", type=float, default=0.0)
    parser.add_argument("--density", type=float, default=1.225)
    parser.add_argument("--viscosity", type=float, default=1.7894e-5)
    parser.add_argument("--temperature", type=float, default=288.15)
    parser.add_argument("--ref-length", type=float, default=5.0)
    parser.add_argument("--ref-area", type=float, default=1.0)
    parser.add_argument("--p-ref", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--patch-id", default=None,
                        help="optional AOX patch id for the Result Viewer VTP manifest")
    parser.add_argument("--coefficient-expert", default=None,
                        help="explicit coefficient label domain; defaults to checkpoint primary")
    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    checkpoint_version = int(checkpoint.get("geometry_preprocessing_version", 1))
    if checkpoint_version != GEOMETRY_PREPROCESSING_VERSION:
        raise ValueError(
            f"checkpoint geometry preprocessing version {checkpoint_version} does not "
            f"match inference version {GEOMETRY_PREPROCESSING_VERSION}; retrain or use "
            "the matching legacy inference code"
        )
    model = ImplicitFieldNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    expert_specs = checkpoint.get("coefficient_experts")
    if not expert_specs:
        expert_specs = {"legacy": {
            "drag_head_state": checkpoint.get("drag_head_state"),
            "lift_head_state": checkpoint.get("lift_head_state"),
            "log_cd_mean": checkpoint.get("log_cd_mean", 0.0),
            "log_cd_std": checkpoint.get("log_cd_std", 1.0),
            "cl_mean": checkpoint.get("cl_mean", 0.0),
            "cl_std": checkpoint.get("cl_std", 1.0),
        }}
    default_expert = checkpoint.get("default_coefficient_expert", next(iter(expert_specs)))
    selected_expert = args.coefficient_expert or default_expert
    if selected_expert not in expert_specs:
        parser.error(
            f"unknown --coefficient-expert {selected_expert!r}; "
            f"choose from {', '.join(expert_specs)}"
        )
    expert_modules = {}
    for name, spec in expert_specs.items():
        drag_head = lift_head = None
        if spec.get("drag_head_state"):
            drag_head = DragHead().to(device)
            drag_head.load_state_dict(spec["drag_head_state"])
            drag_head.eval()
        if spec.get("lift_head_state"):
            lift_head = LiftHead().to(device)
            lift_head.load_state_dict(spec["lift_head_state"])
            lift_head.eval()
        expert_modules[name] = (drag_head, lift_head)

    mesh = load_mesh(args.stl)
    lo, hi = mesh.bounds
    center = (lo + hi) / 2.0
    scale = float(np.max(hi - lo)) or 1.0
    geometry_points, geometry_normals = sample_surface(mesh, args.geometry_points, seed=0)
    geometry = np.concatenate([(geometry_points - center) / scale, geometry_normals], axis=1)
    conditions, u_ref, q_ref = _conditions(args)
    cond_mean = np.asarray(checkpoint["condition_mean"], np.float32)
    cond_std = np.asarray(checkpoint["condition_std"], np.float32)
    conditions = (conditions - cond_mean) / cond_std

    with torch.no_grad():
        geometry_t = torch.from_numpy(geometry[None].astype(np.float32)).to(device)
        condition_t = torch.from_numpy(conditions[None]).to(device)
        latent, cond_latent = model.encode(geometry_t, condition_t)
        expert_predictions = {}
        latent_np = latent[0].detach().cpu().numpy()
        for name, spec in expert_specs.items():
            drag_head, lift_head = expert_modules[name]
            prediction = {"drag_coefficient": None, "lift_coefficient": None}
            prediction["deployment_status"] = spec.get("deployment_status", "validated")
            if drag_head is not None:
                prediction["drag_coefficient"] = float(np.exp(
                    drag_head(latent, cond_latent).item()
                    * float(spec.get("log_cd_std", 1.0))
                    + float(spec.get("log_cd_mean", 0.0))
                ))
            if lift_head is not None:
                prediction["lift_coefficient"] = float(
                    lift_head(latent, cond_latent).item()
                    * float(spec.get("cl_std", 1.0))
                    + float(spec.get("cl_mean", 0.0))
                )
            ood = spec.get("ood") or {}
            if ood.get("centroid") is not None and float(ood.get("radius", 0.0)) > 0.0:
                distance = float(np.linalg.norm(latent_np - np.asarray(ood["centroid"], float)))
                score = distance / float(ood["radius"])
                threshold = float(ood.get("threshold", 1.0))
                prediction.update({
                    "ood_score": score,
                    "ood_threshold": threshold,
                    "in_distribution": bool(score <= threshold),
                })
            expert_predictions[name] = prediction
        selected_prediction = expert_predictions[selected_expert]
        drag_coefficient = selected_prediction["drag_coefficient"]
        lift_coefficient = selected_prediction["lift_coefficient"]

    nx, ny, nz = args.grid
    xmin, xmax, ymin, ymax, zmin, zmax = args.bounds
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    query = np.column_stack([xx.ravel(order="F"), yy.ravel(order="F"), zz.ravel(order="F")])
    volume_pred = _predict_chunks(model, latent, cond_latent, query, device, args.chunk_size)
    volume_cp = volume_pred[:, 0]
    volume_velocity = volume_pred[:, 1:] * u_ref
    volume_pressure = args.p_ref + volume_cp * q_ref

    vertices = np.asarray(mesh.vertices)
    surface_query = (vertices - center) / scale
    surface_pred = _predict_chunks(model, latent, cond_latent, surface_query, device, args.chunk_size)
    surface_cp = surface_pred[:, 0]
    surface_velocity = surface_pred[:, 1:] * u_ref
    surface_pressure = args.p_ref + surface_cp * q_ref

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    origin = center + scale * np.asarray([xmin, ymin, zmin])
    spacing = scale * np.asarray([(xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1),
                                  (zmax - zmin) / (nz - 1)])
    volume_path = out / "volume_field.vti"
    surface_path = out / "surface_pressure.vtp"
    streamlines_path = out / "streamlines.vtp"
    volume = write_volume_vti(volume_path, (nx, ny, nz), origin, spacing,
                              volume_cp, volume_velocity, volume_pressure)
    write_surface_vtp(
        surface_path, vertices, np.asarray(mesh.faces), surface_cp,
        surface_pressure, surface_velocity,
    )
    flow_direction = np.asarray([args.u_x, args.u_y, args.u_z], dtype=float) / u_ref
    write_streamlines(volume, streamlines_path, (lo, hi), flow_direction)
    if args.png:
        render_png(surface_path, streamlines_path, out / "pressure_streamlines.png")

    # AOX Result Viewer compatibility. Django discovers flow.vti, bakes the
    # STRM/VFLD binaries, and associates the VTP field manifest with the body.
    flow_path = out / "flow.vti"
    shutil.copyfile(volume_path, flow_path)
    optimized_stl_path = out / f"{Path(args.stl).stem}_optimized.stl"
    shutil.copyfile(args.stl, optimized_stl_path)
    vtp_dir = out / "vtp"
    vtp_dir.mkdir(exist_ok=True)
    viewer_surface_path = vtp_dir / surface_path.name
    shutil.copyfile(surface_path, viewer_surface_path)
    viewer_manifest = {
        "files": [{
            "patch_id": args.patch_id,
            "patch_type": "source",
            "label": Path(args.stl).name,
            "category": "optimization",
            "output_file": optimized_stl_path.name,
            "vtp_file": viewer_surface_path.name,
            "field": "Pressure_Coefficient",
            "association": "PointData",
        }]
    }
    (vtp_dir / "manifest.json").write_text(json.dumps(viewer_manifest, indent=2))
    summary = {
        "volume": str(volume_path), "surface": str(surface_path),
        "streamlines": str(streamlines_path),
        "flow_volume": str(flow_path),
        "optimized_stl": str(optimized_stl_path),
        "viewer_manifest": str(vtp_dir / "manifest.json"),
        "pressure_range": [float(volume_pressure.min()), float(volume_pressure.max())],
        "speed_range": [float(np.linalg.norm(volume_velocity, axis=1).min()),
                        float(np.linalg.norm(volume_velocity, axis=1).max())],
        "drag_coefficient": drag_coefficient,
        "lift_coefficient": lift_coefficient,
        "coefficient_expert": selected_expert,
        "coefficient_experts": expert_predictions,
        "coefficient_warning": (
            "selected expert is experimental and requires solver verification"
            if selected_prediction.get("deployment_status") == "experimental"
            else (
                None if selected_prediction.get("in_distribution", True)
                else "geometry is outside this expert's training distribution; run G2 verification"
            )
        ),
        "grid": [nx, ny, nz],
        "device": device,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
    }
    (out / "prediction.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
