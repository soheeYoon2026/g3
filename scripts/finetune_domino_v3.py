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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    split = json.load(open(args.split))
    train_ids = [str(row["run"]) for row in split["train_cases"]]
    test_ids = [str(row["run"]) for row in split["test_cases"]]

    ck = snapshot_download("nvidia/domino_drivaerml")
    surf = f"{ck}/domino_drivaerml_surface_checkpoint"
    w = DominoWrapper().load(checkpoint_path=f"{surf}/DoMINO.0.501.mdlus",
                            stats_path=f"{surf}/scaling_factors.pkl",
                            device=args.device, domino_config=f"{surf}/config.yaml")
    model, sf = w._model, w._surf_factors
    pre = {k: v.detach().clone() for k, v in model.state_dict().items()}

    ad = DrivAerMLAdapter(root=root)

    def load_samples(ids):
      samples = []
      for rid in ids:
        try:
            g = geo(root, rid)
            dd = w.prepare_inputs(ad.load_case(f"run_{rid}"))["data_dict"]
            N = dd["surface_mesh_centers"].shape[1]
            smin, smax = dd["surface_min_max"][:, 0], dd["surface_min_max"][:, 1]
            gc = 2.0 * (dd["geometry_coordinates"] - smin) / (smax - smin) - 1
            with torch.no_grad():
                enc = model.geo_rep_surface(gc, dd["surf_grid"], dd["sdf_surf_grid"]).detach()
            tgt = normalize(dd["surface_fields"] / (g["speed"] ** 2 * g["density"]), sf[0], sf[1])
            samples.append(dict(dd=dd, enc=enc, tgt=tgt, N=N, g=g, rid=rid))
        except Exception as e:
            raise RuntimeError(f"failed to load run_{rid}: {type(e).__name__}: {e}") from e
      return samples

    print(f"preprocessing train={len(train_ids)} test={len(test_ids)}...", flush=True)
    tr, val = load_samples(train_ids), load_samples(test_ids)
    print(f"train {len(tr)} / isolated test {len(val)}", flush=True)

    K = 4096
    lossf = torch.nn.MSELoss()

    def val_report(tag):
        model.eval(); cd_errs, cl_errs = [], []
        for s in val:
            with torch.no_grad():
                pred = forward_all(model, s["dd"])[0]
            cpcf = unnormalize(pred, sf[0], sf[1]) * (2.0 * s["g"]["density"])
            centers = s["dd"]["surface_mesh_centers"][0]
            n = s["dd"]["surface_normals"][0]; A = s["dd"]["surface_areas"][0]
            cd, cl = coefficients_from_cpcf(cpcf, centers, n, A, s["g"]["ref_area"])
            cd_errs.append(abs(cd - s["g"]["su2_cd"]))
            cl_errs.append(abs(cl - s["g"]["su2_cl"]))
        result = {"cd_mae": float(np.mean(cd_errs)), "cl_mae": float(np.mean(cl_errs))}
        print(f"  [{tag}] Cd MAE {result['cd_mae']:.6f} Cl MAE {result['cl_mae']:.6f}", flush=True)
        return result

    model.load_state_dict(pre); model.eval()
    base = val_report("PRETRAINED")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for j in torch.randperm(len(tr)):
            s = tr[j]
            idx = torch.randperm(s["N"], device=s["dd"]["surface_mesh_centers"].device)[:min(K, s["N"])]
            b = dict(smc=s["dd"]["surface_mesh_centers"][:, idx], smn=s["dd"]["surface_mesh_neighbors"][:, idx],
                     sn=s["dd"]["surface_normals"][:, idx], snn=s["dd"]["surface_neighbors_normals"][:, idx],
                     sa=s["dd"]["surface_areas"][:, idx].unsqueeze(-1), sna=s["dd"]["surface_neighbors_areas"][:, idx].unsqueeze(-1),
                     pos=s["dd"]["pos_surface_center_of_mass"][:, idx])
            gl = model.surface_local_geo_encodings(0.5 * s["enc"], b["smc"], s["dd"]["surf_grid"])
            pe = model.fc_p_surf(b["pos"])
            pred = model.solution_calculator_surf(b["smc"], gl, pe, b["smn"], b["sn"], b["snn"], b["sa"], b["sna"],
                                                  s["dd"]["global_params_values"], s["dd"]["global_params_reference"])
            opt.zero_grad(); loss = lossf(pred, s["tgt"][:, idx]); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        print(f"epoch {ep + 1}/{args.epochs}", flush=True)
    print(f"{args.epochs} epochs in {time.time()-t0:.0f}s", flush=True)
    model.eval(); ft = val_report("FINE-TUNED")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("=" * 60)
    print(f"  Cd MAE {base['cd_mae']:.6f} -> {ft['cd_mae']:.6f}")
    print(f"  Cl MAE {base['cl_mae']:.6f} -> {ft['cl_mae']:.6f}")
    print(f"  saved {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
