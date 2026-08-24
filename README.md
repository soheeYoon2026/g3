# AOX G3 — AI-model-based aerodynamic solver (Phase 0 spike)

G3 is the **neural-surrogate** engine tier for AOX. Where G1 (OpenFOAM), G2
(SU2 adjoint), and G4 (XLB-LBM) are iterative PDE solvers taking minutes to
hours, G3 predicts **Cd/Cl (and, in production, surface fields) from an STL in
seconds** — the "instant, interactive, batch-screening" tier that sits at the
top of the design funnel and warm-starts the G2/G4 optimisation loops.

> This directory is the **Phase 0 scaffold**, not the production model. Its job
> is to prove the end-to-end data path runs and carries signal *today*, on CPU,
> with no torch and no CFD labels, and to leave clean seams for the GPU model
> and for label generation from G1/G2/G4.

## What runs right now (CPU, no torch)

```bash
pip install -r requirements.txt        # numpy scipy trimesh scikit-learn (+matplotlib)

python scripts/smoke_test.py           # end-to-end: synth -> train -> eval -> FFD, asserts PASS
python scripts/sample_stl_demo.py \
    --stl ../g4-docker-image/ahmed_1.stl --png sampling.png   # sampler on a real body
python -m aox_g3.train                 # train baseline on synthetic bodies
python -m aox_g3.infer --stl ../g4-docker-image/ahmed_1.stl --model g3_model.pkl
```

Verified on this repo: smoke test **PASS** (synthetic val Cd R²≈0.90), 5/5
geometry unit tests pass, sampler runs on the 150k-face `ahmed_1.stl`.

⚠️ `infer` on a real STL with a **synthetic-trained** model returns a
non-physical number — it proves the seam, not aerodynamics. Real predictions
need training on real labels (below).

## G2 pressure + 3-D velocity field surrogate

The field path consumes G2/SU2 ``flow.vtu`` and ``surface_flow.vtu`` files,
learns ``Cp`` plus the three velocity components at arbitrary coordinates, and
writes inference results that ParaView, PyVista, and the AOX streamline baker
can read directly.

```bash
pip install -r requirements-fields.txt

# 1. Compact variable G2 meshes into sampled, normalized NPZ cases.
python scripts/prepare_g2_fields.py \
  --case-glob '/data/g2/*/RBF_DSN_*/' \
  --out-dir data/g2_fields/cases \
  --manifest data/g2_fields/manifest.json

# 2a. Optionally materialize valid G1 drag jobs as surface Cp/UMean/Cd cases.
python scripts/prepare_g1_surfaces.py \
  --csv data/g1/job_output_paths_all_regions_with_cd.csv \
  --out-dir data/g1_surfaces_v1

# 2b. Train the geometry-conditioned implicit field network. G2 supplies the
# volume field; G1 supplies auxiliary surface fields and Cd.
python -m aox_g3.train_fields \
  --manifest data/g2_fields/manifest.json \
  --g1-manifest data/g1_surfaces_v1/manifest.json \
  --out models/g3_field.pt

# 3. Query an STL on a regular 3-D grid and render pressure + streamlines.
python -m aox_g3.infer_fields \
  --stl car.stl --model models/g3_field.pt \
  --out-dir predictions/car --png
```

Inference outputs ``volume_field.vti`` (Cp, pressure, velocity, speed),
``surface_pressure.vtp``, ``streamlines.vtp``, and optionally a combined PNG.
Coordinates are normalized by the body bounding box during training and mapped
back to the uploaded STL coordinate system during inference.

The v6 quality-gated, multi-expert design is documented in
[docs/G3_V6_ARCHITECTURE.md](docs/G3_V6_ARCHITECTURE.md). It keeps G1, G2,
and G4 coefficient calibrations separate, removes low-resolution Cd/Cl labels
from regression loss, and records a per-expert latent OOD score at inference.

The production-safe nightly lifecycle and 10% shadow-canary rollout are
documented in [docs/G3_NIGHTLY_CANARY.md](docs/G3_NIGHTLY_CANARY.md). Model
promotion uses atomic production/challenger pointers and retains the previous
checkpoint for rollback.

Seoul GPU EC2에서 학습·추론을 실행하는 운영 명령과 현재 재학습
주의사항은 [G3 학습·추론 운영 Runbook](docs/G3_TRAINING_INFERENCE_RUNBOOK.md)에
정리해 두었다.

G3에서 사용한 NVIDIA 라이브러리는
[NVIDIA 라이브러리 목록](LIBRARIES.md)에 정리되어 있다.

## Layout

```
g3/
├── aox_g3/
│   ├── config.py              # sampling knobs + TARGETS=(cd,cl); shared train/infer contract
│   ├── geometry/
│   │   ├── stl_sampler.py     # STL -> surface point cloud + normals + SDF  (the model input)
│   │   └── ffd.py             # FFD lattice deform; linear + differentiable (the shape params)
│   ├── data/
│   │   ├── dataset.py         # SyntheticAeroDataset (runs now) + ManifestDataset (real STL+labels)
│   │   └── label_interface.py # adapter to generate labels from G1/G2/G4  <-- the AOX moat
│   ├── models/
│   │   ├── baseline_sklearn.py# pooled-feature MLP (torch-free Phase 0 workhorse)
│   │   └── pointnet.py        # PointNet regressor (torch; on the path to DoMINO)
│   ├── train.py               # model-agnostic training CLI
│   └── infer.py               # STL -> Cd/Cl, the product contract
├── domino/                    # production engine: NVIDIA PhysicsNeMo / DoMINO harness + license map
├── scripts/                   # smoke_test.py, sample_stl_demo.py
├── tests/                     # geometry unit tests (FFD identity/jacobian, dataset signal)
└── requirements.txt
```

## The three seams that make this "reuse", not "from scratch"

1. **Geometry contract** (`geometry/`) — STL → point cloud + SDF, and FFD
   deformation. The same `stl_to_pointcloud` runs at train and inference so the
   model never sees a shifted input distribution. FFD is linear in the
   control-point displacement, so `d(points)/d(delta)` is a constant matrix and
   the whole shape→Cd map is differentiable for gradient-based optimisation.

2. **Label generation** (`data/label_interface.py`) — thin adapters that call
   **G2** (`su2_optimization_pipeline_v8.py`) and **G4** (`run_stl.py --mode
   eval`) to turn STLs into (Cd, Cl [, surface fields]). This is why G3 is
   cheap for AOX and expensive for everyone else: the label bottleneck is
   already paid for.

3. **Model swap** (`models/` → `domino/`) — the baseline and PointNet share the
   `(clouds) -> Cd/Cl` API with the production DoMINO harness, so `train`/`infer`
   don't change when you graduate the engine.

## Path to a real model (see `domino/README.md`)

- **Adopt** NVIDIA PhysicsNeMo / **DoMINO** (Apache-2.0 code). MIT fallback:
  `neuraloperator` (GINO) or `thuml/Transolver`.
- **Train your own weights** on **DrivAerML/AhmedML/WindsorML (CC-BY-SA,
  commercial-clean)** + G1/G2/G4 labels. Seed from an HF checkpoint (NVIDIA Open
  Model License permits commercial use) if useful.
- **Never** train production weights on **DrivAerNet/++ (CC-BY-NC)**, and avoid
  the DoMINO **NIM** turnkey container (needs NVIDIA AI Enterprise for SaaS).
- Rough cost: ~500 high-fidelity cars → Cd R²≈0.95–0.98; training ≈ 8×H100 for
  ~4h (≈$100–160). The expensive part is the CFD labels — which G1/G2/G4 make.

## Integration with the optimisation loop

The surrogate is differentiable, so the existing FFD/scalar-normal loop wraps it
directly: predict Cd → autograd `dCd/d(FFD delta)` → step. Two safer modes to
launch alongside: **warm-start** G2/G4 from G3's prediction (G3 error absorbed by
the downstream solver), and optional **RL** (Gymnasium env, action = morph delta,
reward = −Cd) only for non-differentiable / multi-objective / global search.

## Top risk

Out-of-distribution generalisation: a surrogate trained on one body family
degrades hard on arbitrary uploads (published: RegDGCNN R² 0.90 single-family →
0.64 multi-family). Mitigate by covering the design space with your own solvers,
flagging low-confidence/OOD predictions, and routing tight-margin cases back to
G2/G4 for verification. Position G3 as **explore/screen/accelerate**, never as a
replacement for the numerical solvers on final decisions.
