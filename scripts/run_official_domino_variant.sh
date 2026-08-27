#!/bin/bash
# Usage: run_official_variant.sh <tag> <data_dir> <epochs> [freeze_module ...]
set -u
TAG=$1; DATA=$2; EPOCHS=$3; shift 3
RUN=/home/ubuntu/g3-v2/var/official-$TAG
SNAP=$(ls -d ~/.cache/huggingface/hub/models--nvidia--domino_drivaerml/snapshots/*/domino_drivaerml_surface_checkpoint)

rm -rf "$RUN"
mkdir -p "$RUN/models"
cp "$SNAP/DoMINO.0.501.mdlus" "$RUN/models/"

FREEZE=()
if [ "$#" -gt 0 ]; then
  joined=$(IFS=,; echo "$*")
  FREEZE=("+train.freeze_modules=[$joined]")
fi

cd ~/g3/physicsnemo-repo/examples/cfd/external_aerodynamics/domino/src
cp "$SNAP/config.yaml" "conf/aox_$TAG.yaml"
~/g3/venv/bin/python train.py --config-name "aox_$TAG" \
  project.name=aox_g3 "exp_tag=$TAG" \
  project_dir="$RUN" output="$RUN" resume_dir="$RUN/models" \
  data.input_dir="$DATA/train" data.input_dir_val="$DATA/val" \
  data.scaling_factors="$SNAP/scaling_factors.pkl" \
  "train.epochs=$EPOCHS" train.optimizer.lr=1e-4 \
  model.surface_points_sample=5000 model.geom_points_sample=16000 \
  data.volume_sample_from_disk=false \
  'data.bounding_box_surface.min=[-3.3,-1.5,-0.7]' 'data.bounding_box_surface.max=[4.6,1.5,5.1]' \
  'data.bounding_box.min=[-6.5,-2.7,-3.0]' 'data.bounding_box.max=[7.7,2.7,7.4]' \
  "${FREEZE[@]}"
echo "=== TRAIN $TAG rc=$? ==="
touch "/tmp/DONE_$TAG"
