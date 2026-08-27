#!/bin/bash
# Score a set of training checkpoints on the standing unseen-car gate so the
# epoch is chosen on the metric we promote against, not on validation loss.
set -u
MODELS=$1
OUT=$2
shift 2
: > "$OUT"
for epoch in "$@"; do
  ck="$MODELS/DoMINO.0.$epoch.mdlus"
  [ -f "$ck" ] || { echo "epoch $epoch: 없음"; continue; }
  ~/g3/venv/bin/python /tmp/mdlus_to_state_dict.py --mdlus "$ck" \
      --out /tmp/sweep_ck.pt > /dev/null 2>&1 || { echo "epoch $epoch: 변환 실패"; continue; }
  line=$(cd /tmp/v2eval && ~/g3/venv/bin/python scripts/evaluate_domino_v3.py \
      --root ~/g3-v2/data/domino-g2-reaudit-v1 --fine-tuned /tmp/sweep_ck.pt \
      --split /tmp/v2eval/split-unseen-car-gate.json --partition test 2>/dev/null | tail -1)
  printf '{"epoch": %s, "gate": %s}\n' "$epoch" "$line" >> "$OUT"
  echo "epoch $epoch 완료"
done
~/g3/venv/bin/python - "$OUT" <<'PY'
import json, sys
print(f"{'epoch':>6s} {'Cd MAE':>9s} {'Spearman':>9s} {'Cl MAE':>9s}")
for raw in open(sys.argv[1]):
    row = json.loads(raw)
    g = row["gate"]
    print(f"{row['epoch']:>6d} {g['cd_mae']:9.4f} {g['cd_spearman']:+9.3f} {g['cl_mae']:9.4f}")
PY
