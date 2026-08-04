#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/g3
RUN_DIR="${1:?usage: run_manual_csv_challenger.sh RUN_DIR}"
PY=/home/ubuntu/venv_g3/bin/python
CSV="$RUN_DIR/selected.csv"
MANUAL_SCRIPTS="$RUN_DIR/scripts"
export PYTHONPATH="$RUN_DIR/overlay:$MANUAL_SCRIPTS:$ROOT:$ROOT/scripts"
export PYTHONUNBUFFERED=1
export PYTHONSAFEPATH=1

cd "$ROOT"

step() {
  echo
  echo "===== $(date -Is) $1 ====="
}

step "prepare G2 fields"
if [[ -s "$RUN_DIR/g2-events/manifest.json" ]]; then
  echo "reuse completed $RUN_DIR/g2-events/manifest.json"
else
  "$PY" "$MANUAL_SCRIPTS/prepare_smoke_g2_s3.py" \
    --csv "$CSV" \
    --out-dir "$RUN_DIR/g2-events" \
    --manifest "$RUN_DIR/g2-events/manifest.json"
fi

step "prepare G1 drag surfaces"
if [[ -s "$RUN_DIR/g1-events/manifest.json" ]]; then
  echo "reuse completed $RUN_DIR/g1-events/manifest.json"
else
  "$PY" "$MANUAL_SCRIPTS/prepare_g1_surfaces.py" \
    --csv "$CSV" \
    --out-dir "$RUN_DIR/g1-events" \
    --solver G1 \
    --objective-function drag \
    --dedupe-key project_uid
fi

step "prepare G1 lift surfaces"
if [[ -s "$RUN_DIR/g1-lift-events/manifest.json" ]]; then
  echo "reuse completed $RUN_DIR/g1-lift-events/manifest.json"
else
  "$PY" "$MANUAL_SCRIPTS/prepare_g1_surfaces.py" \
    --csv "$CSV" \
    --out-dir "$RUN_DIR/g1-lift-events" \
    --solver G1 \
    --objective-function lift \
    --dedupe-key project_uid
fi

step "prepare G4 geometry"
if [[ -s "$RUN_DIR/g4-events/manifest.json" ]]; then
  echo "reuse completed $RUN_DIR/g4-events/manifest.json"
else
  "$PY" "$MANUAL_SCRIPTS/prepare_smoke_g4_geometry.py" \
    --csv "$CSV" \
    --out-dir "$RUN_DIR/g4-events"
fi

step "quality-gate new G1 coefficient labels"
"$PY" "$MANUAL_SCRIPTS/filter_coefficient_manifest.py" \
  --manifest "$RUN_DIR/g1-events/manifest.json" \
  --out "$RUN_DIR/g1-events/manifest.filtered.json" \
  --coefficient cd \
  --min-points 4000 \
  --min-value 0.005 \
  --max-value 1.2

"$PY" "$MANUAL_SCRIPTS/filter_coefficient_manifest.py" \
  --manifest "$RUN_DIR/g1-lift-events/manifest.json" \
  --out "$RUN_DIR/g1-lift-events/manifest.filtered.json" \
  --coefficient cl \
  --min-points 4000 \
  --min-value -3.0 \
  --max-value 3.0

step "merge training manifests and freeze holdouts"
"$PY" scripts/filter_training_manifests.py \
  --manifest data/g2_fields_v6/manifest.json \
  --manifest "$RUN_DIR/g2-events/manifest.json" \
  --exclude-manifest data/validation/g2-normal.json \
  --exclude-manifest data/validation/g2-high-drag.json \
  --out "$RUN_DIR/g2-training.json"

"$PY" scripts/filter_training_manifests.py \
  --manifest data/g1_surfaces_v1/manifest.json \
  --manifest data/smoke_g1/manifest.json \
  --manifest "$RUN_DIR/g1-events/manifest.filtered.json" \
  --exclude-manifest data/validation/g1.json \
  --out "$RUN_DIR/g1-training.json"

"$PY" scripts/filter_training_manifests.py \
  --manifest "$RUN_DIR/g1-lift-events/manifest.filtered.json" \
  --out "$RUN_DIR/g1-lift-training.json"

"$PY" scripts/filter_training_manifests.py \
  --manifest data/smoke_g4/manifest.json \
  --manifest "$RUN_DIR/g4-events/manifest.json" \
  --exclude-manifest data/validation/g4.json \
  --out "$RUN_DIR/g4-training.json"

"$PY" scripts/split_g2_coefficient_domains.py \
  --manifest "$RUN_DIR/g2-training.json" \
  --normal-out "$RUN_DIR/g2-normal.json" \
  --high-out "$RUN_DIR/g2-high-drag.json"

step "train shared G2 field backbone"
"$PY" -m aox_g3.train_fields \
  --manifest "$RUN_DIR/g2-training.json" \
  --out "$RUN_DIR/backbone.pt" \
  --epochs 250 \
  --group-balanced-sampling \
  --device cuda

step "train solver-specific coefficient experts"
"$PY" -m aox_g3.train_coefficient_experts \
  --base "$RUN_DIR/backbone.pt" \
  --out "$RUN_DIR/experts.pt" \
  --expert g2_su2_clean "$RUN_DIR/g2-normal.json" \
  --expert g2_su2_high_drag "$RUN_DIR/g2-high-drag.json" \
  --expert g1_openfoam "$RUN_DIR/g1-training.json" \
  --expert g1_openfoam "$RUN_DIR/g1-lift-training.json" \
  --expert g4_lbm "$RUN_DIR/g4-training.json" \
  --epochs 300 \
  --device cuda

"$PY" scripts/annotate_expert_policy.py \
  --checkpoint "$RUN_DIR/experts.pt" \
  --out "$RUN_DIR/challenger.pt"

step "evaluate fixed production holdouts"
"$PY" scripts/evaluate_fixed_holdout.py \
  --checkpoint models/registry/production.pt \
  --output "$RUN_DIR/production.evaluation.json" \
  --dataset g2_normal g2_su2_clean data/validation/g2-normal.json \
  --dataset g2_high_drag g2_su2_high_drag data/validation/g2-high-drag.json \
  --dataset g1 g1_openfoam data/validation/g1.json \
  --dataset g4 g4_lbm data/validation/g4.json \
  --device cuda

"$PY" scripts/evaluate_fixed_holdout.py \
  --checkpoint "$RUN_DIR/challenger.pt" \
  --output "$RUN_DIR/challenger.evaluation.json" \
  --dataset g2_normal g2_su2_clean data/validation/g2-normal.json \
  --dataset g2_high_drag g2_su2_high_drag data/validation/g2-high-drag.json \
  --dataset g1 g1_openfoam data/validation/g1.json \
  --dataset g4 g4_lbm data/validation/g4.json \
  --device cuda

"$PY" "$MANUAL_SCRIPTS/compare_offline_gates.py" \
  --config deploy/g3-nightly.json \
  --production "$RUN_DIR/production.evaluation.json" \
  --challenger "$RUN_DIR/challenger.evaluation.json" \
  --out "$RUN_DIR/offline-gate.json"

step "manual challenger complete; registry unchanged"
