"""Train the G2-conditioned implicit pressure, velocity, Cd, and Cl surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data.field_dataset import (
    CONDITION_NAMES,
    G1SurfaceDataset,
    G2FieldDataset,
    condition_stats,
    load_field_manifest,
)


def split_cases(cases, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    groups = sorted({case.group_id for case in cases})
    if len(groups) >= 2:
        rng.shuffle(groups)
        n_val = max(1, int(round(len(groups) * val_fraction)))
        val_groups = set(groups[:n_val])
        train = [case for case in cases if case.group_id not in val_groups]
        val = [case for case in cases if case.group_id in val_groups]
        mode = "group"
    else:
        indices = rng.permutation(len(cases))
        n_val = min(max(1, int(round(len(cases) * val_fraction))), max(1, len(cases) - 1))
        val_idx = set(indices[:n_val].tolist())
        train = [case for i, case in enumerate(cases) if i not in val_idx]
        val = [case for i, case in enumerate(cases) if i in val_idx]
        mode = "case-fallback"
    if not train or not val:
        raise ValueError("field training requires at least two cases")
    return train, val, mode


def _metrics(pred, target):
    cp_mae = (pred[..., 0] - target[..., 0]).abs().mean().item()
    velocity_rmse = ((pred[..., 1:] - target[..., 1:]) ** 2).mean().sqrt().item()
    return cp_mae, velocity_rmse


def _g2_values(cases, name: str, positive: bool = False) -> list[float]:
    values = []
    for case in cases:
        raw = case.metadata.get("coefficients", {}).get(name, np.nan)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = np.nan
        if np.isfinite(value) and (not positive or value > 0.0):
            values.append(value)
    return values


def _g1_values(cases, key: str, positive: bool = False) -> list[float]:
    values = []
    for case in cases:
        with np.load(case.path) as data:
            value = float(data[key])
        if np.isfinite(value) and (not positive or value > 0.0):
            values.append(value)
    return values


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    return float(np.mean(values)), max(float(np.std(values)), 1e-8)


def main(argv=None):
    import torch
    from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

    from .models.implicit_field import DragHead, ImplicitFieldNet, LiftHead

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--g1-manifest", default=None,
                        help="optional G1 surface Cp/UMean/Cd manifest")
    parser.add_argument("--g1-cl-manifest", default=None,
                        help="optional G1 surface Cp/UMean/Cl manifest")
    parser.add_argument("--out", default="g3_field_model.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--geometry-points", type=int, default=4096)
    parser.add_argument("--volume-queries", type=int, default=8192)
    parser.add_argument("--surface-queries", type=int, default=2048)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--surface-weight", type=float, default=2.0)
    parser.add_argument("--velocity-weight", type=float, default=1.0)
    parser.add_argument("--g2-coefficient-weight", type=float, default=0.25)
    parser.add_argument("--g2-selection-weight", type=float, default=0.5)
    parser.add_argument("--g1-steps-per-epoch", type=int, default=8)
    parser.add_argument("--g1-surface-queries", type=int, default=4096)
    parser.add_argument("--g1-surface-weight", type=float, default=0.5)
    parser.add_argument("--g1-velocity-weight", type=float, default=0.25)
    parser.add_argument("--g1-drag-weight", type=float, default=0.25)
    parser.add_argument("--g1-lift-weight", type=float, default=0.25)
    parser.add_argument("--g1-selection-weight", type=float, default=1.0,
                        help="weight of held-out G1 Cd MAE for checkpoint selection")
    parser.add_argument("--g1-cl-selection-weight", type=float, default=1.0,
                        help="weight of held-out G1 Cl MAE for checkpoint selection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-balanced-sampling", action="store_true",
                        help="sample cases inversely to their optimization-group size")
    parser.add_argument("--coefficient-expert-name", default="g2_su2_clean",
                        help="label domain stored for the primary Cd/Cl heads")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cases = load_field_manifest(args.manifest)
    train_cases, val_cases, split_mode = split_cases(cases, args.val_fraction, args.seed)
    cond_mean, cond_std = condition_stats(train_cases)

    g1_train_cases = g1_val_cases = []
    g1_split_mode = None
    if args.g1_manifest:
        g1_train_cases, g1_val_cases, g1_split_mode = split_cases(
            load_field_manifest(args.g1_manifest), args.val_fraction, args.seed
        )
    g1_cl_train_cases = g1_cl_val_cases = []
    g1_cl_split_mode = None
    if args.g1_cl_manifest:
        g1_cl_train_cases, g1_cl_val_cases, g1_cl_split_mode = split_cases(
            load_field_manifest(args.g1_cl_manifest), args.val_fraction, args.seed
        )

    cd_values = _g2_values(train_cases, "cd", positive=True)
    if args.g1_drag_weight > 0.0:
        cd_values += _g1_values(g1_train_cases, "cd_final", positive=True)
    log_cd_mean, log_cd_std = _mean_std([np.log(value) for value in cd_values])
    cl_values = _g2_values(train_cases, "cl")
    if args.g1_lift_weight > 0.0:
        cl_values += _g1_values(g1_cl_train_cases, "cl_final")
    cl_mean, cl_std = _mean_std(cl_values)

    common = dict(
        geometry_points=args.geometry_points,
        volume_queries=args.volume_queries,
        surface_queries=args.surface_queries,
        condition_mean=cond_mean,
        condition_std=cond_std,
        log_cd_mean=log_cd_mean,
        log_cd_std=log_cd_std,
        cl_mean=cl_mean,
        cl_std=cl_std,
        seed=args.seed,
    )
    train_ds = G2FieldDataset(train_cases, **common)
    val_ds = G2FieldDataset(val_cases, **common)
    def training_sampler(split_cases, num_samples):
        if not args.group_balanced_sampling:
            return RandomSampler(split_cases, replacement=True, num_samples=num_samples)
        group_counts = {}
        for case in split_cases:
            group_counts[case.group_id] = group_counts.get(case.group_id, 0) + 1
        weights = [1.0 / group_counts[case.group_id] for case in split_cases]
        generator = torch.Generator().manual_seed(args.seed)
        return WeightedRandomSampler(
            weights, num_samples=num_samples, replacement=True, generator=generator
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=training_sampler(train_cases, args.steps_per_epoch * args.batch_size),
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    def surface_loaders(cases_train, cases_val, *, key, transform, mean, std):
        if not cases_train:
            return None, None, None, None
        options = dict(
            geometry_points=args.geometry_points,
            surface_queries=args.g1_surface_queries,
            condition_mean=cond_mean,
            condition_std=cond_std,
            coefficient_key=key,
            coefficient_transform=transform,
            coefficient_mean=mean,
            coefficient_std=std,
            seed=args.seed,
        )
        ds_train = G1SurfaceDataset(cases_train, **options)
        ds_val = G1SurfaceDataset(cases_val, **options)
        loader_train = DataLoader(
            ds_train, batch_size=args.batch_size,
            sampler=training_sampler(
                cases_train, args.g1_steps_per_epoch * args.batch_size
            ),
            num_workers=0,
        )
        return ds_train, ds_val, loader_train, DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0)

    g1_train_ds, g1_val_ds, g1_loader, g1_val_loader = surface_loaders(
        g1_train_cases, g1_val_cases, key="cd_final", transform="log",
        mean=log_cd_mean, std=log_cd_std,
    )
    g1_cl_train_ds, g1_cl_val_ds, g1_cl_loader, g1_cl_val_loader = surface_loaders(
        g1_cl_train_cases, g1_cl_val_cases, key="cl_final", transform="standard",
        mean=cl_mean, std=cl_std,
    )

    has_cd = bool(cd_values)
    has_cl = bool(cl_values)
    model = ImplicitFieldNet(condition_dim=len(CONDITION_NAMES)).to(device)
    drag_head = DragHead().to(device) if has_cd else None
    lift_head = LiftHead().to(device) if has_cl else None
    parameters = list(model.parameters())
    if drag_head is not None:
        parameters += list(drag_head.parameters())
    if lift_head is not None:
        parameters += list(lift_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    message = f"device={device} G2={len(cases)} train={len(train_cases)} val={len(val_cases)} split={split_mode}"
    if args.g1_manifest:
        message += (f" G1-Cd={len(g1_train_cases) + len(g1_val_cases)}"
                    f" train={len(g1_train_cases)} val={len(g1_val_cases)} split={g1_split_mode}")
    if args.g1_cl_manifest:
        message += (f" G1-Cl={len(g1_cl_train_cases) + len(g1_cl_val_cases)}"
                    f" train={len(g1_cl_train_cases)} val={len(g1_cl_val_cases)} split={g1_cl_split_mode}")
    print(message)

    def train_surface_aux(loader, dataset, head, coefficient_weight):
        losses = []
        if loader is None or head is None:
            return losses
        dataset.set_epoch(epoch)
        for batch in loader:
            geometry = batch["geometry"].to(device)
            conditions = batch["conditions"].to(device)
            points = batch["surface_points"].to(device)
            targets = batch["surface_targets"].to(device)
            latent, cond_latent = model.encode(geometry, conditions)
            pred = model.decode(latent, cond_latent, points)
            coefficient_pred = head(latent, cond_latent)
            cp_loss = torch.mean((pred[..., 0] - targets[..., 0]) ** 2)
            velocity_loss = torch.mean((pred[..., 1:] - targets[..., 1:]) ** 2)
            coefficient_loss = torch.mean(
                (coefficient_pred - batch["coefficient_target"].to(device)) ** 2
            )
            loss = (args.g1_surface_weight * cp_loss
                    + args.g1_velocity_weight * velocity_loss
                    + coefficient_weight * coefficient_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(loss.item())
        return losses

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        if drag_head is not None:
            drag_head.train()
        if lift_head is not None:
            lift_head.train()
        losses, epoch_latents = [], []
        for batch in train_loader:
            geometry = batch["geometry"].to(device)
            conditions = batch["conditions"].to(device)
            vp = batch["volume_points"].to(device)
            vt = batch["volume_targets"].to(device)
            sp = batch["surface_points"].to(device)
            st = batch["surface_cp"].to(device)
            latent, cond_latent = model.encode(geometry, conditions)
            epoch_latents.append(latent.detach().cpu().numpy())
            volume_pred = model.decode(latent, cond_latent, vp)
            surface_pred = model.decode(latent, cond_latent, sp)[..., :1]
            cp_loss = torch.mean((volume_pred[..., 0] - vt[..., 0]) ** 2)
            velocity_loss = torch.mean((volume_pred[..., 1:] - vt[..., 1:]) ** 2)
            surface_loss = torch.mean((surface_pred - st) ** 2)
            loss = cp_loss + args.velocity_weight * velocity_loss + args.surface_weight * surface_loss
            mask = batch["global_mask"].to(device)
            targets = batch["global_targets"].to(device)
            if drag_head is not None and bool(mask[:, 0].any()):
                loss = loss + args.g2_coefficient_weight * torch.mean(
                    (drag_head(latent, cond_latent)[mask[:, 0]] - targets[:, :1][mask[:, 0]]) ** 2
                )
            if lift_head is not None and bool(mask[:, 1].any()):
                loss = loss + args.g2_coefficient_weight * torch.mean(
                    (lift_head(latent, cond_latent)[mask[:, 1]] - targets[:, 1:2][mask[:, 1]]) ** 2
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(loss.item())

        g1_losses = train_surface_aux(g1_loader, g1_train_ds, drag_head, args.g1_drag_weight)
        g1_cl_losses = train_surface_aux(g1_cl_loader, g1_cl_train_ds, lift_head, args.g1_lift_weight)
        scheduler.step()

        model.eval()
        if drag_head is not None:
            drag_head.eval()
        if lift_head is not None:
            lift_head.eval()
        val_cp, val_vel, g2_cd_errors, g2_cl_errors = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                geometry = batch["geometry"].to(device)
                conditions = batch["conditions"].to(device)
                latent, cond_latent = model.encode(geometry, conditions)
                pred = model.decode(latent, cond_latent, batch["volume_points"].to(device))
                cp, vel = _metrics(pred, batch["volume_targets"].to(device))
                val_cp.append(cp)
                val_vel.append(vel)
                mask = batch["global_mask"].to(device)
                values = batch["global_values"].to(device)
                if drag_head is not None and bool(mask[:, 0].any()):
                    pred_cd = torch.exp(drag_head(latent, cond_latent) * log_cd_std + log_cd_mean)
                    g2_cd_errors.append(torch.abs(pred_cd[mask[:, 0]] - values[:, :1][mask[:, 0]]).mean().item())
                if lift_head is not None and bool(mask[:, 1].any()):
                    pred_cl = lift_head(latent, cond_latent) * cl_std + cl_mean
                    g2_cl_errors.append(torch.abs(pred_cl[mask[:, 1]] - values[:, 1:2][mask[:, 1]]).mean().item())

        def auxiliary_errors(loader, head, mean, std, transform):
            errors = []
            if loader is None or head is None:
                return errors
            with torch.no_grad():
                for batch in loader:
                    latent, cond_latent = model.encode(
                        batch["geometry"].to(device), batch["conditions"].to(device)
                    )
                    predicted = head(latent, cond_latent)
                    predicted = torch.exp(predicted * std + mean) if transform == "log" else predicted * std + mean
                    errors.append(float(torch.abs(predicted - batch["coefficient"].to(device)).mean()))
            return errors

        g1_cd_errors = auxiliary_errors(g1_val_loader, drag_head, log_cd_mean, log_cd_std, "log")
        g1_cl_errors = auxiliary_errors(g1_cl_val_loader, lift_head, cl_mean, cl_std, "standard")
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_cp_mae": float(np.mean(val_cp)),
            "val_velocity_rmse_u_ref": float(np.mean(val_vel)),
        }
        if g2_cd_errors:
            row["g2_val_cd_mae"] = float(np.mean(g2_cd_errors))
        if g2_cl_errors:
            row["g2_val_cl_mae"] = float(np.mean(g2_cl_errors))
        if g1_losses:
            row["g1_train_loss"] = float(np.mean(g1_losses))
        if g1_cd_errors:
            row["g1_val_cd_mae"] = float(np.mean(g1_cd_errors))
        if g1_cl_losses:
            row["g1_cl_train_loss"] = float(np.mean(g1_cl_losses))
        if g1_cl_errors:
            row["g1_val_cl_mae"] = float(np.mean(g1_cl_errors))

        score = row["val_cp_mae"] + row["val_velocity_rmse_u_ref"]
        score += args.g2_selection_weight * (row.get("g2_val_cd_mae", 0.0) + row.get("g2_val_cl_mae", 0.0))
        score += args.g1_selection_weight * row.get("g1_val_cd_mae", 0.0)
        score += args.g1_cl_selection_weight * row.get("g1_val_cl_mae", 0.0)
        row["selection_score"] = score
        history.append(row)
        if score < best:
            best = score
            latent_samples = np.concatenate(epoch_latents, axis=0)
            latent_centroid = latent_samples.mean(axis=0)
            latent_distances = np.linalg.norm(latent_samples - latent_centroid, axis=1)
            ood_radius = max(float(np.quantile(latent_distances, 0.95)), 1e-8)
            primary_expert = {
                "drag_head_state": drag_head.state_dict() if drag_head is not None else None,
                "lift_head_state": lift_head.state_dict() if lift_head is not None else None,
                "log_cd_mean": log_cd_mean,
                "log_cd_std": log_cd_std,
                "cl_mean": cl_mean,
                "cl_std": cl_std,
                "ood": {
                    "method": "geometry_latent_l2_q95",
                    "centroid": latent_centroid.tolist(),
                    "radius": ood_radius,
                    "threshold": 1.0,
                },
                "train_cases": [case.case_id for case in train_cases],
                "val_cases": [case.case_id for case in val_cases],
            }
            torch.save({
                "model_state": model.state_dict(),
                "drag_head_state": drag_head.state_dict() if drag_head is not None else None,
                "lift_head_state": lift_head.state_dict() if lift_head is not None else None,
                "model_config": {"condition_dim": len(CONDITION_NAMES)},
                "condition_names": CONDITION_NAMES,
                "condition_mean": cond_mean,
                "condition_std": cond_std,
                "training": vars(args),
                "split_mode": split_mode,
                "train_cases": [case.case_id for case in train_cases],
                "val_cases": [case.case_id for case in val_cases],
                "best_score": best,
                "best_epoch": epoch,
                "best_metrics": row,
                "g1_split_mode": g1_split_mode,
                "g1_train_cases": [case.case_id for case in g1_train_cases],
                "g1_val_cases": [case.case_id for case in g1_val_cases],
                "g1_cl_split_mode": g1_cl_split_mode,
                "g1_cl_train_cases": [case.case_id for case in g1_cl_train_cases],
                "g1_cl_val_cases": [case.case_id for case in g1_cl_val_cases],
                "log_cd_mean": log_cd_mean,
                "log_cd_std": log_cd_std,
                "cl_mean": cl_mean,
                "cl_std": cl_std,
                "coefficient_expert_version": 1,
                "default_coefficient_expert": args.coefficient_expert_name,
                "coefficient_experts": {args.coefficient_expert_name: primary_expert},
            }, output)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(row))

    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    print(f"saved best checkpoint -> {output}")
    return history[-1]


if __name__ == "__main__":
    main()
