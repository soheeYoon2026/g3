#!/usr/bin/env python
"""Scaled-up universal fine-tune on diverse SU2 shapes, validated vs SU2 Cd.

Dedups near-identical optimisation-baseline copies, fine-tunes DoMINO on the
diverse set, and validates predicted Cd (integrated with the case's SU2 REF_AREA)
against SU2's authoritative history.csv Cd on held-out watertight cases —
pretrained vs fine-tuned.
"""
import argparse, json, os, time, numpy as np, torch
from pathlib import Path
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


def sampled_cd(pred, sample, idx, factors, scale_to_full):
    """Differentiable Cd from a random subset of surface cells.

    The training step only forwards K cells, so integrate those and scale by
    N/K — an unbiased estimate of the full-surface integral that still carries
    gradients back into the solution head.
    """
    cpcf = unnormalize(pred[0], factors[0], factors[1]) * (2.0 * sample["g"]["density"])
    dd = sample["dd"]
    centers = dd["surface_mesh_centers"][0][idx]
    normals = dd["surface_normals"][0][idx]
    areas = dd["surface_areas"][0][idx]
    if torch.sum(torch.sum(centers * normals, dim=1) * areas) < 0:
        normals = -normals
    cp, cf = cpcf[:, 0], cpcf[:, 1:4]
    force = (torch.sum((-cp[:, None] * normals) * areas[:, None], dim=0)
             - torch.sum(cf * areas[:, None], dim=0))
    return force[0] * scale_to_full / float(sample["g"]["ref_area"])


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
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--pairs", type=Path,
                        help="ΔCp pair manifest; adds a paired-difference loss term")
    parser.add_argument("--delta-weight", type=float, default=1.0,
                        help="weight of the ΔCp difference loss")
    parser.add_argument("--patch-weight", type=float, default=3.0,
                        help="extra weight on cells whose CFD ΔCp is large")
    parser.add_argument("--cd-weight", type=float, default=0.0,
                        help="weight of the integrated-Cd loss (multi-objective)")
    parser.add_argument("--delta-cd-weight", type=float, default=0.0,
                        help="weight of the paired ΔCd loss — the product metric itself")
    parser.add_argument("--delta-cd-scale", type=float, default=0.01,
                        help="ΔCd error treated as unit cost; keeps the term "
                             "comparable to the field losses without a huge weight")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    root = os.path.abspath(args.root)
    split = json.load(open(args.split))
    train_ids = [str(row["run"]) for row in split["train_cases"]]
    validation_ids = [str(row["run"]) for row in split.get("validation_cases", [])]
    test_ids = [str(row["run"]) for row in split["test_cases"]]

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

    def load_samples(ids):
      samples = []
      for rid in ids:
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
            samples.append(dict(dd=dd, gc=gc, enc=enc, tgt=tgt, N=N, g=g, rid=rid))
        except Exception as e:
            raise RuntimeError(f"failed to load run_{rid}: {type(e).__name__}: {e}") from e
      return samples

    print(f"preprocessing train={len(train_ids)} validation={len(validation_ids)} test={len(test_ids)}...", flush=True)
    tr = load_samples(train_ids)
    val = load_samples(validation_ids) if validation_ids else []
    test = load_samples(test_ids)
    print(f"train {len(tr)} / validation {len(val)} / isolated test {len(test)}", flush=True)

    K = 4096
    lossf = torch.nn.MSELoss()

    # ΔCp pairing: integrating Cp to one Cd hides a small deformation, so train
    # on the pair difference too (2026-08-26: CFD ΔCp in the deformed patch is
    # 1.3-2.3x the far field, but the model only reproduces it at corr ~0.3).
    by_run = {int(s["rid"]): s for s in tr}
    pair_index = []
    if args.pairs:
        payload = json.loads(Path(args.pairs).read_text())
        raw_pairs = payload["pairs"] if isinstance(payload, dict) else payload
        from scipy.spatial import cKDTree

        for entry in raw_pairs:
            b, v = int(entry["baseline"]), int(entry["variant"])
            if b not in by_run or v not in by_run:
                print(f"  [pairs] skip {b}->{v} (run not in train set)", flush=True)
                continue
            # G2 re-meshes every variant, so cell counts differ: map the variant's
            # cells onto the baseline's by nearest centre once, up front.
            bc = by_run[b]["dd"]["surface_mesh_centers"][0].detach().cpu().numpy()
            vc = by_run[v]["dd"]["surface_mesh_centers"][0].detach().cpu().numpy()
            _, mapping = cKDTree(vc).query(bc)
            device = by_run[b]["dd"]["surface_mesh_centers"].device
            # carry the measured ΔCd so the paired pass can optimise the product
            # metric directly, not only the field difference
            true_delta_cd = entry.get("true_delta_cd")
            if true_delta_cd is None:
                true_delta_cd = (by_run[v]["g"]["su2_cd"] - by_run[b]["g"]["su2_cd"])
            pair_index.append((b, v, torch.as_tensor(mapping, dtype=torch.long, device=device),
                               float(true_delta_cd)))
        print(f"  [pairs] {len(pair_index)} usable ΔCp pairs, "
              f"delta_weight={args.delta_weight} patch_weight={args.patch_weight} "
              f"delta_cd_weight={args.delta_cd_weight}", flush=True)

    def evaluate(samples, tag):
        model.eval(); cd_errs, cl_errs = [], []
        for s in samples:
            with torch.no_grad():
                pred = forward_all(model, s["dd"])[0]
            cpcf = unnormalize(pred, sf[0], sf[1]) * (2.0 * s["g"]["density"])
            centers = s["dd"]["surface_mesh_centers"][0]
            n = s["dd"]["surface_normals"][0]; A = s["dd"]["surface_areas"][0]
            cd, cl = coefficients_from_cpcf(cpcf, centers, n, A, s["g"]["ref_area"])
            if not np.isfinite(cd) or not np.isfinite(cl):
                print(f"  [guard] non-finite prediction run_{s['rid']} — excluded", flush=True)
                continue
            cd_errs.append(abs(cd - s["g"]["su2_cd"]))
            cl_errs.append(abs(cl - s["g"]["su2_cl"]))
        result = {"cd_mae": float(np.mean(cd_errs)), "cl_mae": float(np.mean(cl_errs))}
        print(f"  [{tag}] Cd MAE {result['cd_mae']:.6f} Cl MAE {result['cl_mae']:.6f}", flush=True)
        return result

    model.load_state_dict(pre); model.eval()
    base_val = evaluate(val, "PRETRAINED-VALIDATION") if val else None
    base = evaluate(test, "PRETRAINED-ISOLATED-TEST")
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
    t0 = time.time()
    best_state, best_validation, validation_history = None, float("inf"), []
    for ep in range(args.epochs):
        model.train()
        for j in torch.randperm(len(tr)):
            s = tr[j]
            idx = torch.randperm(s["N"], device=s["dd"]["surface_mesh_centers"].device)[:min(K, s["N"])]
            b = dict(smc=s["dd"]["surface_mesh_centers"][:, idx], smn=s["dd"]["surface_mesh_neighbors"][:, idx],
                     sn=s["dd"]["surface_normals"][:, idx], snn=s["dd"]["surface_neighbors_normals"][:, idx],
                     sa=s["dd"]["surface_areas"][:, idx].unsqueeze(-1), sna=s["dd"]["surface_neighbors_areas"][:, idx].unsqueeze(-1),
                     pos=s["dd"]["pos_surface_center_of_mass"][:, idx])
            enc = (
                model.geo_rep_surface(s["gc"], s["dd"]["surf_grid"], s["dd"]["sdf_surf_grid"])
                if args.train_mode == "encoder-tail" else s["enc"]
            )
            gl = model.surface_local_geo_encodings(0.5 * enc, b["smc"], s["dd"]["surf_grid"])
            pe = model.fc_p_surf(b["pos"])
            pred = model.solution_calculator_surf(b["smc"], gl, pe, b["smn"], b["sn"], b["snn"], b["sa"], b["sna"],
                                                  s["dd"]["global_params_values"], s["dd"]["global_params_reference"])
            opt.zero_grad(); loss = lossf(pred, s["tgt"][:, idx])
            if args.cd_weight > 0:
                # Field accuracy alone did not carry over to integrated Cd
                # (challenger-paired-v1, 2026-08-26), so optimise both.
                cd_pred = sampled_cd(pred, s, idx, sf, s["N"] / float(len(idx)))
                cd_true = torch.as_tensor(float(s["g"]["su2_cd"]), device=cd_pred.device)
                loss = loss + args.cd_weight * (cd_pred - cd_true) ** 2
            if not torch.isfinite(loss):
                print(f"  [guard] non-finite loss on run_{s['rid']} epoch {ep + 1} — skipped", flush=True)
                opt.zero_grad(); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        # paired ΔCp pass: same cells for both shapes, so the difference of the
        # two predictions can be matched against the CFD difference directly.
        for b_run, v_run, mapping, true_delta_cd in pair_index:
            sb, sv = by_run[b_run], by_run[v_run]
            idx = torch.randperm(sb["N"], device=sb["dd"]["surface_mesh_centers"].device)[:min(K, sb["N"])]
            v_idx = mapping[idx]

            def head(sample, index):
                d = sample["dd"]
                gl = model.surface_local_geo_encodings(
                    0.5 * sample["enc"], d["surface_mesh_centers"][:, index], d["surf_grid"])
                pe = model.fc_p_surf(d["pos_surface_center_of_mass"][:, index])
                return model.solution_calculator_surf(
                    d["surface_mesh_centers"][:, index], gl, pe,
                    d["surface_mesh_neighbors"][:, index], d["surface_normals"][:, index],
                    d["surface_neighbors_normals"][:, index],
                    d["surface_areas"][:, index].unsqueeze(-1),
                    d["surface_neighbors_areas"][:, index].unsqueeze(-1),
                    d["global_params_values"], d["global_params_reference"])

            pred_b, pred_v = head(sb, idx), head(sv, v_idx)
            pred_delta = pred_v - pred_b
            true_delta = sv["tgt"][:, v_idx] - sb["tgt"][:, idx]
            # weight cells where the CFD actually changed — that is the signal
            magnitude = true_delta.detach().abs()
            weight = 1.0 + args.patch_weight * (magnitude / (magnitude.mean() + 1e-8)).clamp(max=5.0)
            loss = (weight * (pred_delta - true_delta) ** 2).mean() * args.delta_weight
            if args.delta_cd_weight > 0:
                # Field accuracy did not carry over to ΔCd sign (83-pair audit,
                # 2026-08-27: magnitudes 0.4-0.8x correct, direction near random),
                # so penalise the integrated difference the product reports.
                cd_b = sampled_cd(pred_b, sb, idx, sf, sb["N"] / float(len(idx)))
                cd_v = sampled_cd(pred_v, sv, v_idx, sf, sv["N"] / float(len(v_idx)))
                target = torch.as_tensor(true_delta_cd, device=cd_b.device, dtype=cd_b.dtype)
                residual = (cd_v - cd_b - target) / args.delta_cd_scale
                loss = loss + args.delta_cd_weight * residual ** 2
            if not torch.isfinite(loss):
                print(f"  [guard] non-finite ΔCp loss on {b_run}->{v_run} — skipped", flush=True)
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        print(f"epoch {ep + 1}/{args.epochs}", flush=True)
        if val:
            metrics = evaluate(val, f"VALIDATION-EPOCH-{ep + 1}")
            validation_history.append({"epoch": ep + 1, **metrics})
            score = metrics["cd_mae"] + metrics["cl_mae"]
            if score < best_validation:
                best_validation = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"{args.epochs} epochs in {time.time()-t0:.0f}s", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    ft_val = evaluate(val, "FINE-TUNED-VALIDATION") if val else None
    ft = evaluate(test, "FINE-TUNED-ISOLATED-TEST")
    required_metrics = [base["cd_mae"], base["cl_mae"], ft["cd_mae"], ft["cl_mae"]]
    if ft_val:
        required_metrics += [ft_val["cd_mae"], ft_val["cl_mae"]]
    if not all(np.isfinite(value) for value in required_metrics):
        raise RuntimeError("refusing to save checkpoint with non-finite validation metrics")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    with open(args.out + ".metrics.json", "w") as stream:
        json.dump({"train_mode": args.train_mode, "seed": args.seed, "pretrained_validation": base_val,
                   "validation_history": validation_history, "finetuned_validation": ft_val,
                   "pretrained_test": base, "finetuned_test": ft}, stream, indent=2)
    print("=" * 60)
    print(f"  Cd MAE {base['cd_mae']:.6f} -> {ft['cd_mae']:.6f}")
    print(f"  Cl MAE {base['cl_mae']:.6f} -> {ft['cl_mae']:.6f}")
    print(f"  saved {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
