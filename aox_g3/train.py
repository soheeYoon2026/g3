"""Train a G3 surrogate on synthetic or manifest data.

Phase 0 default is the torch-free sklearn baseline on synthetic bodies, so
``python -m aox_g3.train`` runs out of the box. Point it at a real manifest
(labels from G1/G2/G4 or DrivAerML) to train on aerodynamics.

    python -m aox_g3.train                          # synthetic smoke train
    python -m aox_g3.train --data cars.json          # real STL+label manifest
    python -m aox_g3.train --backend pointnet ...    # needs torch + GPU
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

from .config import TARGETS, DEFAULT_SAMPLE
from .data.dataset import SyntheticAeroDataset, ManifestDataset
from .models.baseline_sklearn import PooledMLPRegressor


def _load_samples(args):
    if args.data == "synthetic":
        ds = SyntheticAeroDataset(n=args.n, cfg=DEFAULT_SAMPLE, seed=args.seed)
    else:
        ds = ManifestDataset(args.data, cfg=DEFAULT_SAMPLE)
    samples = list(ds)
    clouds = [s.cloud for s in samples]
    targets = np.stack([s.targets for s in samples])
    return clouds, targets


def _split(n, val_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]


def _report(y_true, y_pred, label):
    print(f"\n{label} metrics ({len(y_true)} samples):")
    for j, name in enumerate(TARGETS):
        r2 = r2_score(y_true[:, j], y_pred[:, j])
        mae = mean_absolute_error(y_true[:, j], y_pred[:, j])
        # Cd is reported in "drag counts" (1 count = 1e-4) as the industry does.
        extra = f"  (~{mae * 1e4:.1f} drag counts)" if name == "cd" else ""
        print(f"  {name:>3}:  R2 = {r2:6.3f}   MAE = {mae:.4f}{extra}")
    return r2_score(y_true[:, 0], y_pred[:, 0])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train a G3 aero surrogate.")
    ap.add_argument("--backend", choices=["sklearn", "pointnet"], default="sklearn")
    ap.add_argument("--data", default="synthetic",
                    help="'synthetic' or path to a JSON manifest of STL+labels")
    ap.add_argument("--n", type=int, default=400, help="synthetic sample count")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=100, help="pointnet only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="g3_model.pkl")
    args = ap.parse_args(argv)

    clouds, targets = _load_samples(args)
    tr, va = _split(len(clouds), args.val_frac, args.seed)
    print(f"Loaded {len(clouds)} samples -> train {len(tr)} / val {len(va)}")

    if args.backend == "sklearn":
        model = PooledMLPRegressor(seed=args.seed)
        model.fit([clouds[i] for i in tr], targets[tr])
        _report(targets[tr], model.predict([clouds[i] for i in tr]), "train")
        val_r2 = _report(targets[va], model.predict([clouds[i] for i in va]), "val")
        model.save(args.out)
    else:
        val_r2 = _train_pointnet(clouds, targets, tr, va, args)

    print(f"\nSaved model -> {args.out}")
    return val_r2


def _train_pointnet(clouds, targets, tr, va, args):
    """Torch training loop, isolated so the sklearn path needs no torch."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .models.pointnet import PointNetRegressor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(np.stack([c.features for c in clouds]), dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    dl = DataLoader(TensorDataset(X[tr], y[tr]), batch_size=32, shuffle=True)

    model = PointNetRegressor(in_features=X.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()
    for ep in range(args.epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(X[va].to(device)).cpu().numpy()
    val_r2 = _report(targets[va], pred, "val")
    torch.save(model.state_dict(), args.out.replace(".pkl", ".pt"))
    return val_r2


if __name__ == "__main__":
    main()
