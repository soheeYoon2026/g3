# G3 production engine — NVIDIA PhysicsNeMo / DoMINO

The Phase 0 baseline (`aox_g3`, pooled features + sklearn/PointNet) exists to
prove the data path. The **production G3 engine** is a point-based neural
operator that predicts *surface fields* (pressure, wall-shear) which integrate
to Cd/Cl. Recommended model: **DoMINO** from NVIDIA PhysicsNeMo.

## Why DoMINO

- Matches the AOX contract exactly: STL in → surface field → Cd/Cl.
- Reproducible external-aero results on **DrivAerML** (drag R² ≈ 0.97, MAE a few
  drag counts).
- **Code is Apache-2.0** (`github.com/NVIDIA/physicsnemo`) — free for commercial
  use.

MIT-licensed fallbacks if you want zero NVIDIA coupling: `neuraloperator`
(GINO/FNO) or `thuml/Transolver`.

## License map (read before shipping)

| Asset | License | Commercial SaaS |
|---|---|---|
| PhysicsNeMo code / architectures | Apache-2.0 | ✅ free |
| HF pretrained weights (`nvidia/…` DrivAerML) | NVIDIA Open Model License | ✅ commercial use permitted — use as a seed |
| DoMINO-Automotive-Aero **NIM** turnkey container | NVIDIA AI Enterprise | ⚠️ needs AI Enterprise subscription for multi-user serving — **avoid the dependency** |
| DrivAerML / AhmedML / WindsorML **data** | CC-BY-SA 4.0 | ✅ attribution + share-alike |
| DrivAerNet / DrivAerNet++ **data** | CC-BY-**NC** 4.0 | ❌ non-commercial — never train production weights on it |

**Production stance:** train your own weights with Apache-2.0 code on CC-BY-SA
data + your G1/G2/G4 labels. Avoid the NIM path (AI Enterprise cost) and avoid
CC-BY-NC data.

## Get going

```bash
# 1. GPU env with torch + CUDA, then:
pip install nvidia-physicsnemo

# 2. Fetch a seed checkpoint (confirm exact repo id on the HF model card) OR
#    train your own — see aox_g3/data/label_interface.py for G1/G2/G4 labeling.

# 3. Point the harness at it:
export DOMINO_CHECKPOINT=/path/to/checkpoint
python run_domino_inference.py --stl ../../g4-docker-image/ahmed_1.stl
```

Without the stack installed, the harness prints exactly what is missing and how
to get it — the integration seam (`run()` in `run_domino_inference.py`) is where
you connect the physicsnemo DoMINO inference API.

Reference example in the physicsnemo repo:
`examples/cfd/external_aerodynamics/domino`
