"""Attach independently calibrated Cd/Cl experts to a trained field model."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data.field_dataset import CONDITION_NAMES
from .geometry.surface_sampling import GEOMETRY_PREPROCESSING_VERSION


@dataclass
class ExpertCase:
    case_id: str
    group_id: str
    path: Path
    conditions: np.ndarray
    cd: float
    cl: float


def load_cases(
    manifest_path: Path,
    expected_preprocessing_version: int | None = None,
) -> list[ExpertCase]:
    payload = json.loads(manifest_path.read_text())
    rows = payload["cases"] if isinstance(payload, dict) else payload
    result = []
    for row in rows:
        version = int(row.get("geometry_preprocessing_version", 1))
        if (
            expected_preprocessing_version is not None
            and version != expected_preprocessing_version
        ):
            raise ValueError(
                f"{manifest_path}: case {row.get('case_id')} uses geometry "
                f"preprocessing version {version}; expected "
                f"{expected_preprocessing_version}"
            )
        path = Path(row["npz"])
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        with np.load(path) as data:
            coefficients = row.get("coefficients", {})
            cd_raw = row.get(
                "cd_final",
                coefficients.get("cd", data["cd_final"] if "cd_final" in data else np.nan),
            )
            cl_raw = row.get(
                "cl_final",
                coefficients.get("cl", data["cl_final"] if "cl_final" in data else np.nan),
            )
            try:
                cd = float(cd_raw)
            except (TypeError, ValueError):
                cd = np.nan
            try:
                cl = float(cl_raw)
            except (TypeError, ValueError):
                cl = np.nan
        if not ((np.isfinite(cd) and cd > 0.0) or np.isfinite(cl)):
            continue
        conditions = np.asarray([
            float(row.get("conditions", {}).get(name, 0.0)) for name in CONDITION_NAMES
        ], dtype=np.float32)
        result.append(ExpertCase(
            case_id=str(row["case_id"]),
            group_id=str(row.get("group_id", row["case_id"])),
            path=path,
            conditions=conditions,
            cd=cd,
            cl=cl,
        ))
    return result


def split_groups(cases, fraction, seed):
    rng = np.random.default_rng(seed)
    groups = np.asarray(sorted({case.group_id for case in cases}), dtype=object)
    if len(groups) < 2:
        indices = rng.permutation(len(cases))
        n_val = min(max(1, int(round(len(cases) * fraction))), len(cases) - 1)
        validation = set(indices[:n_val].tolist())
        return (
            [case for index, case in enumerate(cases) if index not in validation],
            [case for index, case in enumerate(cases) if index in validation],
        )
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * fraction)))
    validation = set(groups[:n_val].tolist())
    return (
        [case for case in cases if case.group_id not in validation],
        [case for case in cases if case.group_id in validation],
    )


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), max(float(values.std()), 1e-8)


def main(argv=None):
    import torch

    from .models.implicit_field import DragHead, ImplicitFieldNet, LiftHead

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--expert", nargs=2, action="append", metavar=("NAME", "MANIFEST"), required=True,
        help="repeat to merge manifests into one named label-domain expert",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--geometry-points", type=int, default=4096)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.base, map_location=device, weights_only=False)
    checkpoint_version = int(checkpoint.get("geometry_preprocessing_version", 1))
    if checkpoint_version != GEOMETRY_PREPROCESSING_VERSION:
        raise ValueError(
            f"base checkpoint uses geometry preprocessing version {checkpoint_version}; "
            f"expected {GEOMETRY_PREPROCESSING_VERSION}"
        )
    model = ImplicitFieldNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    cond_mean = np.asarray(checkpoint["condition_mean"], np.float32)
    cond_std = np.asarray(checkpoint["condition_std"], np.float32)
    grouped: dict[str, list[ExpertCase]] = {}
    sources: dict[str, list[str]] = {}
    for name, manifest in args.expert:
        path = Path(manifest).resolve()
        grouped.setdefault(name, []).extend(load_cases(
            path, expected_preprocessing_version=GEOMETRY_PREPROCESSING_VERSION
        ))
        sources.setdefault(name, []).append(str(path))

    expert_specs = copy.deepcopy(checkpoint.get("coefficient_experts", {}))
    report = {}
    for expert_index, (name, all_cases) in enumerate(grouped.items()):
        unique = {case.case_id: case for case in all_cases}
        cases = list(unique.values())
        train_cases, val_cases = split_groups(cases, args.val_fraction, args.seed + expert_index)
        train_condition_values = np.stack([case.conditions for case in train_cases])
        condition_range = {
            name: {
                "min": float(train_condition_values[:, index].min()),
                "max": float(train_condition_values[:, index].max()),
            }
            for index, name in enumerate(checkpoint["condition_names"])
        }
        cd_values = [case.cd for case in train_cases if np.isfinite(case.cd) and case.cd > 0]
        cl_values = [case.cl for case in train_cases if np.isfinite(case.cl)]
        if not cd_values and not cl_values:
            raise ValueError(f"expert {name!r} has no finite coefficient labels")
        log_cd_mean, log_cd_std = mean_std(np.log(cd_values)) if cd_values else (0.0, 1.0)
        cl_mean, cl_std = mean_std(cl_values) if cl_values else (0.0, 1.0)

        drag = DragHead().to(device) if cd_values else None
        lift = LiftHead().to(device) if cl_values else None
        parameters = []
        if drag is not None:
            parameters.extend(drag.parameters())
        if lift is not None:
            parameters.extend(lift.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-5)

        def embed(case, sample_seed):
            with np.load(case.path) as data:
                points = np.asarray(data["geometry_points"], np.float32)
                normals = np.asarray(data["geometry_normals"], np.float32)
            local_rng = np.random.default_rng(sample_seed)
            indices = local_rng.choice(
                len(points), args.geometry_points, replace=len(points) < args.geometry_points
            )
            geometry = np.concatenate([points[indices], normals[indices]], axis=1)
            conditions = (case.conditions - cond_mean) / cond_std
            with torch.no_grad():
                return model.encode(
                    torch.from_numpy(geometry[None]).to(device),
                    torch.from_numpy(conditions[None]).to(device),
                )

        # Fixed validation embeddings keep model selection deterministic.
        validation_embeddings = {
            case.case_id: embed(case, args.seed + 900_000 + index)
            for index, case in enumerate(val_cases)
        }

        def errors(case_list, fixed=None):
            cd_errors, cl_errors = [], []
            with torch.no_grad():
                for index, case in enumerate(case_list):
                    latent, condition_latent = (
                        fixed[case.case_id] if fixed is not None
                        else embed(case, args.seed + 800_000 + index)
                    )
                    if drag is not None and np.isfinite(case.cd) and case.cd > 0:
                        prediction = torch.exp(drag(latent, condition_latent) * log_cd_std + log_cd_mean)
                        cd_errors.append(abs(float(prediction.item()) - case.cd))
                    if lift is not None and np.isfinite(case.cl):
                        prediction = lift(latent, condition_latent) * cl_std + cl_mean
                        cl_errors.append(abs(float(prediction.item()) - case.cl))
            return cd_errors, cl_errors

        best_score, best_states, best_metrics = float("inf"), None, None
        for epoch in range(1, args.epochs + 1):
            if drag is not None:
                drag.train()
            if lift is not None:
                lift.train()
            order = rng.permutation(len(train_cases))
            losses = []
            for index in order:
                case = train_cases[int(index)]
                latent, condition_latent = embed(
                    case, args.seed + epoch * 100_003 + int(index)
                )
                terms = []
                if drag is not None and np.isfinite(case.cd) and case.cd > 0:
                    target = (np.log(case.cd) - log_cd_mean) / log_cd_std
                    terms.append((drag(latent, condition_latent) - target).square().mean())
                if lift is not None and np.isfinite(case.cl):
                    target = (case.cl - cl_mean) / cl_std
                    terms.append((lift(latent, condition_latent) - target).square().mean())
                if not terms:
                    continue
                loss = sum(terms)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            if drag is not None:
                drag.eval()
            if lift is not None:
                lift.eval()
            cd_errors, cl_errors = errors(val_cases, validation_embeddings)
            metrics = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_cd_mae": float(np.mean(cd_errors)) if cd_errors else None,
                "val_cl_mae": float(np.mean(cl_errors)) if cl_errors else None,
            }
            score = sum(value for value in (metrics["val_cd_mae"], metrics["val_cl_mae"])
                        if value is not None)
            if score < best_score:
                best_score, best_metrics = score, metrics
                best_states = {
                    "drag": copy.deepcopy(drag.state_dict()) if drag is not None else None,
                    "lift": copy.deepcopy(lift.state_dict()) if lift is not None else None,
                }
            if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
                print(json.dumps({"expert": name, **metrics}))

        if drag is not None:
            drag.load_state_dict(best_states["drag"])
        if lift is not None:
            lift.load_state_dict(best_states["lift"])
        latent_rows = []
        for index, case in enumerate(train_cases):
            latent, _ = embed(case, args.seed + 700_000 + index)
            latent_rows.append(latent[0].detach().cpu().numpy())
        latent_rows = np.stack(latent_rows)
        centroid = latent_rows.mean(axis=0)
        distances = np.linalg.norm(latent_rows - centroid, axis=1)
        radius = max(float(np.quantile(distances, 0.95)), 1e-8)
        expert_specs[name] = {
            "drag_head_state": best_states["drag"],
            "lift_head_state": best_states["lift"],
            "log_cd_mean": log_cd_mean, "log_cd_std": log_cd_std,
            "cl_mean": cl_mean, "cl_std": cl_std,
            "ood": {
                "method": "geometry_latent_l2_q95", "centroid": centroid.tolist(),
                "radius": radius, "threshold": 1.0,
            },
            "condition_range": condition_range,
            "sources": sources[name],
            "train_cases": [case.case_id for case in train_cases],
            "val_cases": [case.case_id for case in val_cases],
            "best_metrics": best_metrics,
            "deployment_status": "validated" if len(cases) >= 20 else "experimental",
        }
        report[name] = {
            "cases": len(cases), "train": len(train_cases), "val": len(val_cases),
            "best_metrics": best_metrics, "sources": sources[name],
        }

    checkpoint["coefficient_expert_version"] = 1
    checkpoint["coefficient_experts"] = expert_specs
    checkpoint.setdefault("default_coefficient_expert", "g2_su2_clean")
    checkpoint["coefficient_expert_training"] = report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    args.out.with_suffix(".experts.json").write_text(json.dumps(report, indent=2))
    print(f"saved {len(expert_specs)} experts -> {args.out}")


if __name__ == "__main__":
    main()
