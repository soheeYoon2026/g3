"""End-to-end Phase 0 smoke test — runs today, no torch, no CFD labels.

Proves the full path works and carries signal:
  synthetic bodies -> point-cloud sampling -> pooled features -> MLP regression
  -> validation R2 -> save -> reload -> single-sample inference.

Pass criteria: validation Cd R2 clears a floor (the features must actually
predict the analytic pseudo-drag). If this fails, the data plumbing is broken —
fix it before touching the GPU stack.

    python scripts/smoke_test.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aox_g3.config import TARGETS, SampleConfig
from aox_g3.data.dataset import SyntheticAeroDataset
from aox_g3.models.baseline_sklearn import PooledMLPRegressor
from aox_g3.geometry.ffd import FFD

R2_FLOOR = 0.80  # Cd validation R2 the pipeline must clear on synthetic data


def main() -> int:
    print("=" * 64)
    print("AOX G3 Phase 0 smoke test")
    print("=" * 64)

    cfg = SampleConfig(n_surface_points=1536)  # smaller = faster smoke test
    ds = SyntheticAeroDataset(n=300, cfg=cfg, seed=1)
    samples = list(ds)
    clouds = [s.cloud for s in samples]
    targets = np.stack([s.targets for s in samples])
    print(f"[1/5] built {len(clouds)} synthetic bodies, "
          f"{cfg.n_surface_points} pts each; targets={TARGETS}")

    n = len(clouds)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    va, tr = idx[: n // 5], idx[n // 5:]

    model = PooledMLPRegressor(seed=0)
    model.fit([clouds[i] for i in tr], targets[tr])
    print(f"[2/5] trained PooledMLPRegressor on {len(tr)} samples")

    pred = model.predict([clouds[i] for i in va])
    from sklearn.metrics import r2_score, mean_absolute_error
    cd_r2 = r2_score(targets[va, 0], pred[:, 0])
    cd_mae = mean_absolute_error(targets[va, 0], pred[:, 0])
    cl_r2 = r2_score(targets[va, 1], pred[:, 1])
    print(f"[3/5] val:  Cd R2={cd_r2:.3f} (MAE {cd_mae*1e4:.1f} counts)   "
          f"Cl R2={cl_r2:.3f}")

    # Persistence + single-sample inference round-trip.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.pkl"
        model.save(str(p))
        reloaded = PooledMLPRegressor.load(str(p))
        one = reloaded.predict([clouds[va[0]]])[0]
    print(f"[4/5] save/reload/infer OK -> Cd={one[0]:.4f} Cl={one[1]:.4f} "
          f"(truth Cd={targets[va[0],0]:.4f})")

    # FFD sanity: zero displacement is identity; a real displacement moves points.
    ffd = FFD(clouds[0].points, dims=(4, 3, 3))
    ident = ffd.deform(np.zeros((ffd.n_control, 3)))
    id_err = np.abs(ident - clouds[0].points).max()
    delta = np.zeros((ffd.n_control, 3)); delta[0] = [0.1, 0, 0]
    moved = np.abs(ffd.deform(delta) - clouds[0].points).max()
    print(f"[5/5] FFD: identity err={id_err:.2e} (want ~0)   "
          f"max move from 1 CP={moved:.3f} (want >0)")

    ok = cd_r2 >= R2_FLOOR and id_err < 1e-6 and moved > 1e-3
    print("=" * 64)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}  "
          f"(Cd R2 {cd_r2:.3f} >= {R2_FLOOR}? {cd_r2 >= R2_FLOOR})")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
