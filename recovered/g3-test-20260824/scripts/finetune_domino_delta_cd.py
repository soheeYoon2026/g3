#!/usr/bin/env python
"""Scaled-up universal fine-tune on diverse SU2 shapes, validated vs SU2 Cd.

Dedups near-identical optimisation-baseline copies, fine-tunes DoMINO on the
diverse set, and validates predicted Cd (integrated with the case's SU2 REF_AREA)
against SU2's authoritative history.csv Cd on held-out watertight cases —
pretrained vs fine-tuned.
"""
import argparse, json, os, time, numpy as np, torch
from huggingface_hub import snapshot_download
from physicsnemo.cfd.evaluation.datasets.adapters.drivaerml import DrivAerMLAdapter
from physicsnemo.cfd.evaluation.models.wrappers.domino.wrapper import DominoWrapper
from physicsnemo.models.domino.utils import normalize, unnormalize

def geo(root, rid):
    return json.loads(open(os.path.join(root, f"run_{rid}", f"conditions_{rid}.json")).read())


def forward_all(model, dd):
    smin, smax = dd["surface_min_max"][:, 0], dd["surface_min_max"][:, 1]
    gc = 2.0 * (dd["geometry_coordinates"] - smin) / (smax - smin) - 1
    enc = model.geo_rep_surface(gc, dd["surf_grid"], dd["sdf_surf_grid"])
    gl = model.surface_local_geo_encodings(0.5 * enc, dd["surface_mesh_centers"], dd["surf_grid"])
    pos = model.fc_p_surf(dd["pos_surface_center_of_mass"])
    return model.solution_calculator_surf(
        dd["surface_mesh_centers"], gl, pos, dd["surface_mesh_neighbors"],
        dd["surface_normals"], dd["surface_neighbors_normals"],
        dd["surface_areas"].unsqueeze(-1), dd["surface_neighbors_areas"].unsqueeze(-1),
        dd["global_params_values"], dd["global_params_reference"])


def coefficients_from_cpcf(cpcf, centers, n, A, aref):
    if torch.sum(torch.sum(centers * n, dim=1) * A) < 0:
        n = -n
    Cp, Cf = cpcf[:, 0], cpcf[:, 1:4]
    force = (torch.sum((-Cp[:, None] * n) * A[:, None], dim=0)
             - torch.sum(Cf * A[:, None], dim=0)) / aref
    return float(force[0]), float(force[2])


def source_job(row):
    source_id = str(row.get("source", {}).get("source_id", ""))
    return source_id.split(":", 1)[0] if source_id else str(row.get("group_id", row["run"]))


def source_design(row):
    source_id = str(row.get("source", {}).get("source_id", ""))
    return source_id.split(":", 1)[1] if ":" in source_id else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-mode", choices=("decoder", "encoder-tail", "full"), default="decoder")
    parser.add_argument("--base-checkpoint",
                        help="optional fine-tuned state dict to use instead of NVIDIA pretrained weights")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--field-loss-weight", type=float, default=0.25)
    parser.add_argument("--cd-loss-weight", type=float, default=1.0)
    parser.add_argument("--delta-cd-loss-weight", type=float, default=5.0)
    parser.add_argument("--coefficient-points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    root = os.path.abspath(args.root)
    split = json.load(open(args.split))
    train_rows = split["train_cases"]
    validation_rows = split.get("validation_cases", [])
    test_rows = split["test_cases"]

    ck = snapshot_download("nvidia/domino_drivaerml")
    surf = f"{ck}/domino_drivaerml_surface_checkpoint"
    w = DominoWrapper().load(checkpoint_path=f"{surf}/DoMINO.0.501.mdlus",
                            stats_path=f"{surf}/scaling_factors.pkl",
                            device=args.device, domino_config=f"{surf}/config.yaml")
    model, sf = w._model, w._surf_factors
    if args.base_checkpoint:
        model.load_state_dict(torch.load(args.base_checkpoint, map_location=args.device))
    pre = {k: v.detach().clone() for k, v in model.state_dict().items()}

    ad = DrivAerMLAdapter(root=root)

    def load_samples(rows):
      samples = []
      for row in rows:
        rid = str(row["run"])
        try:
            g = geo(root, rid)
            required = ("speed", "density", "ref_area", "su2_cd", "su2_cl")
            invalid = [key for key in required if key not in g or not np.isfinite(float(g[key]))]
            invalid += [key for key in ("speed", "density", "ref_area")
                        if key in g and np.isfinite(float(g[key])) and float(g[key]) <= 0]
            if invalid:
                raise ValueError(f"invalid required conditions: {sorted(set(invalid))}")
            dd = w.prepare_inputs(ad.load_case(f"run_{rid}"))["data_dict"]
            N = dd["surface_mesh_centers"].shape[1]
            smin, smax = dd["surface_min_max"][:, 0], dd["surface_min_max"][:, 1]
            gc = 2.0 * (dd["geometry_coordinates"] - smin) / (smax - smin) - 1
            with torch.no_grad():
                enc = model.geo_rep_surface(gc, dd["surf_grid"], dd["sdf_surf_grid"]).detach()
            tgt = normalize(dd["surface_fields"] / (g["speed"] ** 2 * g["density"]), sf[0], sf[1])
            samples.append(dict(dd=dd, gc=gc, enc=enc, tgt=tgt, N=N, g=g, rid=rid,
                                job=source_job(row), design=source_design(row)))
        except Exception as e:
            raise RuntimeError(f"failed to load run_{rid}: {type(e).__name__}: {e}") from e
      return samples

    print(f"preprocessing train={len(train_rows)} validation={len(validation_rows)} test={len(test_rows)}...", flush=True)
    tr = load_samples(train_rows)
    val = load_samples(validation_rows) if validation_rows else []
    test = load_samples(test_rows)
    print(f"train {len(tr)} / validation {len(val)} / isolated test {len(test)}", flush=True)

    K = 4096
    lossf = torch.nn.MSELoss()

    def delta_pairs(samples, transitions=None):
        if transitions is not None:
            by_run = {int(sample["rid"]): sample for sample in samples}
            return [
                (by_run[int(row["from_geometry"]["run"])], by_run[int(row["to_geometry"]["run"])])
                for row in transitions
            ]
        grouped = {}
        for sample in samples:
            grouped.setdefault(sample["job"], []).append(sample)
        pairs = []
        for values in grouped.values():
            if len(values) < 2:
                continue
            anchor = next((sample for sample in values if sample["design"] == "FINAL"), values[0])
            pairs.extend((anchor, sample) for sample in values if sample is not anchor)
        return pairs

    def evaluate(samples, tag, transitions=None):
        model.eval(); cd_errs, cl_errs, cd_by_run = [], [], {}
        for s in samples:
            with torch.no_grad():
                pred = forward_all(model, s["dd"])[0]
            cpcf = unnormalize(pred, sf[0], sf[1]) * (2.0 * s["g"]["density"])
            centers = s["dd"]["surface_mesh_centers"][0]
            n = s["dd"]["surface_normals"][0]; A = s["dd"]["surface_areas"][0]
            cd, cl = coefficients_from_cpcf(cpcf, centers, n, A, s["g"]["ref_area"])
            if not np.isfinite(cd) or not np.isfinite(cl):
                raise ValueError(f"non-finite prediction for run_{s['rid']}: Cd={cd}, Cl={cl}")
            cd_errs.append(abs(cd - s["g"]["su2_cd"]))
            cl_errs.append(abs(cl - s["g"]["su2_cl"]))
            cd_by_run[s["rid"]] = cd
        pair_errors, direction_hits = [], []
        for anchor, candidate in delta_pairs(samples, transitions):
            predicted_delta = cd_by_run[candidate["rid"]] - cd_by_run[anchor["rid"]]
            actual_delta = candidate["g"]["su2_cd"] - anchor["g"]["su2_cd"]
            pair_errors.append(abs(predicted_delta - actual_delta))
            direction_hits.append((predicted_delta == 0 and actual_delta == 0) or predicted_delta * actual_delta > 0)
        result = {"cd_mae": float(np.mean(cd_errs)), "cl_mae": float(np.mean(cl_errs)),
                  "delta_cd_mae": float(np.mean(pair_errors)) if pair_errors else None,
                  "delta_direction_accuracy": float(np.mean(direction_hits)) if direction_hits else None,
                  "delta_pairs": len(pair_errors)}
        delta_text = (f" Delta-Cd MAE {result['delta_cd_mae']:.6f} direction {100*result['delta_direction_accuracy']:.1f}%"
                      if pair_errors else "")
        print(f"  [{tag}] Cd MAE {result['cd_mae']:.6f} Cl MAE {result['cl_mae']:.6f}{delta_text}", flush=True)
        return result

    def sampled_prediction(sample, idx):
        dd = sample["dd"]
        smc = dd["surface_mesh_centers"][:, idx]
        smn = dd["surface_mesh_neighbors"][:, idx]
        sn = dd["surface_normals"][:, idx]
        snn = dd["surface_neighbors_normals"][:, idx]
        sa = dd["surface_areas"][:, idx].unsqueeze(-1)
        sna = dd["surface_neighbors_areas"][:, idx].unsqueeze(-1)
        pos = dd["pos_surface_center_of_mass"][:, idx]
        enc = (model.geo_rep_surface(sample["gc"], dd["surf_grid"], dd["sdf_surf_grid"])
               if args.train_mode == "encoder-tail" else sample["enc"])
        gl = model.surface_local_geo_encodings(0.5 * enc, smc, dd["surf_grid"])
        pe = model.fc_p_surf(pos)
        return model.solution_calculator_surf(
            smc, gl, pe, smn, sn, snn, sa, sna,
            dd["global_params_values"], dd["global_params_reference"])

    def sampled_cd(sample, idx, prediction):
        dd = sample["dd"]
        cpcf = unnormalize(prediction[0], sf[0], sf[1]) * (2.0 * sample["g"]["density"])
        centers = dd["surface_mesh_centers"][0]
        normals = dd["surface_normals"][0]
        areas = dd["surface_areas"][0]
        orientation = -1.0 if torch.sum(torch.sum(centers * normals, dim=1) * areas).item() < 0 else 1.0
        sample_normals = normals[idx] * orientation
        sample_areas = areas[idx]
        area_scale = torch.sum(areas) / torch.clamp(torch.sum(sample_areas), min=1e-12)
        cp, cf = cpcf[:, 0], cpcf[:, 1:4]
        force_x = area_scale * torch.sum((-cp * sample_normals[:, 0] - cf[:, 0]) * sample_areas)
        return force_x / sample["g"]["ref_area"]

    model.load_state_dict(pre); model.eval()
    train_transitions = split.get("train_transitions")
    validation_transitions = split.get("validation_transitions")
    test_transitions = split.get("test_transitions")
    base_val = evaluate(val, "PRETRAINED-VALIDATION", validation_transitions) if val else None
    base = evaluate(test, "PRETRAINED-ISOLATED-TEST", test_transitions)
    if args.train_mode in {"decoder", "encoder-tail"}:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in (model.fc_p_surf, model.solution_calculator_surf):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if args.train_mode == "encoder-tail":
            for modules in (
                model.geo_rep_surface.geo_conv_out,
                model.geo_rep_surface.geo_processors,
                model.geo_rep_surface.geo_processor_out,
            ):
                for parameter in modules[-1].parameters():
                    parameter.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"train_mode={args.train_mode} trainable={sum(p.numel() for p in trainable):,} total={sum(p.numel() for p in model.parameters()):,}", flush=True)
    if args.train_mode == "encoder-tail":
        decoder_parameters = list(model.fc_p_surf.parameters()) + list(model.solution_calculator_surf.parameters())
        decoder_ids = {id(parameter) for parameter in decoder_parameters}
        encoder_parameters = [parameter for parameter in trainable if id(parameter) not in decoder_ids]
        opt = torch.optim.Adam([
            {"params": decoder_parameters, "lr": 2e-4},
            {"params": encoder_parameters, "lr": 2e-5},
        ])
    else:
        opt = torch.optim.Adam(trainable, lr=args.learning_rate)
    train_pairs = delta_pairs(tr, train_transitions)
    print(f"Delta-Cd training pairs: {len(train_pairs)}", flush=True)
    t0 = time.time()
    best_state, best_validation, validation_history = None, float("inf"), []
    for ep in range(args.epochs):
        model.train()
        for j in torch.randperm(len(tr)):
            s = tr[j]
            idx = torch.randperm(s["N"], device=s["dd"]["surface_mesh_centers"].device)[:min(K, s["N"])]
            pred = sampled_prediction(s, idx)
            field_loss = lossf(pred, s["tgt"][:, idx])
            predicted_cd = sampled_cd(s, idx, pred)
            cd_target = predicted_cd.new_tensor(s["g"]["su2_cd"])
            cd_loss = torch.nn.functional.smooth_l1_loss(predicted_cd, cd_target, beta=0.005)
            loss = args.field_loss_weight * field_loss + args.cd_loss_weight * cd_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        for pair_index in torch.randperm(len(train_pairs)):
            anchor, candidate = train_pairs[int(pair_index)]
            anchor_count = min(args.coefficient_points, anchor["N"])
            candidate_count = min(args.coefficient_points, candidate["N"])
            anchor_idx = torch.randperm(anchor["N"], device=anchor["dd"]["surface_mesh_centers"].device)[:anchor_count]
            candidate_idx = torch.randperm(candidate["N"], device=candidate["dd"]["surface_mesh_centers"].device)[:candidate_count]
            anchor_cd = sampled_cd(anchor, anchor_idx, sampled_prediction(anchor, anchor_idx))
            candidate_cd = sampled_cd(candidate, candidate_idx, sampled_prediction(candidate, candidate_idx))
            predicted_delta = candidate_cd - anchor_cd
            actual_delta = predicted_delta.new_tensor(candidate["g"]["su2_cd"] - anchor["g"]["su2_cd"])
            delta_loss = torch.nn.functional.smooth_l1_loss(predicted_delta, actual_delta, beta=0.002)
            opt.zero_grad(); (args.delta_cd_loss_weight * delta_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        print(f"epoch {ep + 1}/{args.epochs}", flush=True)
        if val:
            metrics = evaluate(val, f"VALIDATION-EPOCH-{ep + 1}", validation_transitions)
            validation_history.append({"epoch": ep + 1, **metrics})
            score = (metrics["delta_cd_mae"] if metrics["delta_cd_mae"] is not None else metrics["cd_mae"])
            score += 0.25 * metrics["cd_mae"]
            if score < best_validation:
                best_validation = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"{args.epochs} epochs in {time.time()-t0:.0f}s", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    ft_val = evaluate(val, "FINE-TUNED-VALIDATION", validation_transitions) if val else None
    ft = evaluate(test, "FINE-TUNED-ISOLATED-TEST", test_transitions)
    required_metrics = [base["cd_mae"], base["cl_mae"], ft["cd_mae"], ft["cl_mae"]]
    if ft_val:
        required_metrics += [ft_val["cd_mae"], ft_val["cl_mae"]]
    if not all(np.isfinite(value) for value in required_metrics):
        raise RuntimeError("refusing to save checkpoint with non-finite validation metrics")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    with open(args.out + ".metrics.json", "w") as stream:
        json.dump({"train_mode": args.train_mode, "seed": args.seed,
                   "field_loss_weight": args.field_loss_weight, "cd_loss_weight": args.cd_loss_weight,
                   "delta_cd_loss_weight": args.delta_cd_loss_weight,
                   "pretrained_validation": base_val,
                   "validation_history": validation_history, "finetuned_validation": ft_val,
                   "pretrained_test": base, "finetuned_test": ft}, stream, indent=2)
    print("=" * 60)
    print(f"  Cd MAE {base['cd_mae']:.6f} -> {ft['cd_mae']:.6f}")
    print(f"  Cl MAE {base['cl_mae']:.6f} -> {ft['cl_mae']:.6f}")
    if base["delta_cd_mae"] is not None:
        print(f"  Delta-Cd MAE {base['delta_cd_mae']:.6f} -> {ft['delta_cd_mae']:.6f}")
        print(f"  Delta direction {100*base['delta_direction_accuracy']:.1f}% -> {100*ft['delta_direction_accuracy']:.1f}%")
    print(f"  saved {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
