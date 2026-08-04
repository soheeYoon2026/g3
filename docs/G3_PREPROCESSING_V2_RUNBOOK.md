# G3 preprocessing v2: test and run

All commands start in the G3 repository on the local Linux machine:

```bash
cd /home/adro1234/2026/SU2_work/g3
```

## 1. Run the fast automated tests

```bash
.venv/bin/python -m pytest -q
```

This covers deterministic area-weighted sampling, barycentric field
interpolation, identical STL training/inference sampling, a one-epoch PyTorch
training run, checkpoint reload, and STL inference.

## 2. Convert one real G2 case

```bash
.venv/bin/python scripts/prepare_g2_fields.py \
  --case /path/to/G2/RBF_DSN_001 \
  --out-dir var/preprocessing-v2-smoke/cases \
  --manifest var/preprocessing-v2-smoke/manifest.json \
  --n-volume 10000 \
  --n-surface 4096
```

Validate the generated NPZ:

```bash
.venv/bin/python scripts/validate_preprocessed_case.py \
  var/preprocessing-v2-smoke/cases/GROUP__RBF_DSN_001.npz
```

## 3. Run a small training smoke test

The manifest must contain at least two independent groups.

```bash
.venv/bin/python -m aox_g3.train_fields \
  --manifest var/preprocessing-v2-smoke/manifest.json \
  --out var/preprocessing-v2-smoke/tiny.pt \
  --epochs 3 \
  --batch-size 1 \
  --steps-per-epoch 2 \
  --geometry-points 512 \
  --volume-queries 1024 \
  --surface-queries 512 \
  --device cpu
```

## 4. Infer one STL with the smoke checkpoint

```bash
.venv/bin/python -m aox_g3.infer_fields \
  --stl /path/to/car.stl \
  --model var/preprocessing-v2-smoke/tiny.pt \
  --out-dir var/preprocessing-v2-smoke/inference \
  --grid 16 16 16 \
  --geometry-points 512 \
  --ref-length 5.0 \
  --ref-area 1.0 \
  --device cpu
```

The smoke checkpoint only proves that the pipeline works. Its aerodynamic
predictions are not production quality.

## 5. Full GPU challenger workflow

Do not replace `models/registry/production.pt`. On the GPU instance:

1. Regenerate every G1/G2/G4 NPZ with preprocessing version 2.
2. Verify every manifest excludes frozen validation cases.
3. Train a versioned field checkpoint and coefficient experts.
4. Evaluate production and challenger on the same family-level holdouts.
5. Stage the result as `challenger.pt` only if every offline gate passes.
6. Promote only after shadow-canary gates pass.

Preprocessing-v1 checkpoints intentionally fail with preprocessing-v2
inference code. Deploy the new inference code and v2 checkpoint together.
