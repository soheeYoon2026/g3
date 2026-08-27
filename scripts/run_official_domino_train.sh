#!/bin/bash
# Retrain from the NVIDIA pretrained checkpoint on the non-dimensionalised fields.
# The first run fed dimensional Pa against Cp-scale normalisation statistics; the
# training data is now p/(rho*U^2), matching the checkpoint the run resumes from.
set -u
RUN=${G3_OFFICIAL_RUN_DIR:-/home/ubuntu/g3-v2/var/official-train-v2}
DATA=${G3_OFFICIAL_NPY_DIR:-/home/ubuntu/g3-v2/data/official-npy-cp}
SRC=${G3_DOMINO_SRC:-$HOME/g3/physicsnemo-repo/examples/cfd/external_aerodynamics/domino/src}
PY=${G3_PYTHON:-$HOME/g3/venv/bin/python}
SNAP=$(ls -d ~/.cache/huggingface/hub/models--nvidia--domino_drivaerml/snapshots/*/domino_drivaerml_surface_checkpoint)

# resume_dir must already hold the pretrained checkpoint, otherwise train.py
# starts from random weights instead of fine-tuning.
rm -rf "$RUN"
mkdir -p "$RUN/models"
cp "$SNAP/DoMINO.0.501.mdlus" "$RUN/models/"

cd "$SRC"
cp "$SNAP/config.yaml" conf/aox_finetune_v2.yaml
# epochs must exceed the checkpoint's own 501 or train.py exits immediately; the
# sample counts must stay under our smallest case (21,144 STL points); the
# bounding boxes are measured from our geometry, not DrivAerML's.
"$PY" train.py --config-name aox_finetune_v2 \
  project.name=aox_g3 exp_tag=official2 \
  project_dir="$RUN" \
  output="$RUN" \
  resume_dir="$RUN/models" \
  data.input_dir="$DATA/train" \
  data.input_dir_val="$DATA/val" \
  data.scaling_factors="$SNAP/scaling_factors.pkl" \
  train.epochs=560 train.optimizer.lr=1e-4 \
  model.surface_points_sample=5000 model.geom_points_sample=16000 \
  data.volume_sample_from_disk=false \
  'data.bounding_box_surface.min=[-3.3,-1.5,-0.7]' 'data.bounding_box_surface.max=[4.6,1.5,5.1]' \
  'data.bounding_box.min=[-6.5,-2.7,-3.0]' 'data.bounding_box.max=[7.7,2.7,7.4]'
echo "=== TRAIN rc=$? ==="
touch /tmp/OFFICIAL_TRAIN_V2_DONE
