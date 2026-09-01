#!/bin/bash
# LES labels for the drag-increasing benchmark, one case at a time.
#
# The v7 campaign reached cd_std 0.0004 with 4-level grids frozen from the
# baseline; v8 and v9 skipped the freeze and got four times the scatter, which
# made six of seven pairs unusable. So the baseline runs first and its refinement
# is reused for every variant of that car.
set -u
cd "$(dirname "$0")/src"
export XLA_FLAGS=--xla_gpu_autotune_level=0   # autotune shifts Cd by 6%
export LES_MOVIE_FRAMES=0
export LES_PLACEMENT=auto

STL_DIR=${STL_DIR:-$HOME/gtr-les/stl_bench}
OUT=${OUT:-$HOME/gtr-les/bench_results}
GRADE=${GRADE:-precision}
# which cars this host runs; the two hosts split the campaign
CARS=${CARS:-"carA carB"}
# inflow speed in lattice units; 0.10 is the validated default, 0.12 the
# stability limit, 0.14 diverges. A finer base cell needs a lower value:
# carB's 41.1 mm cell diverged at 0.10 where carA's 50.0 mm did not.
U_INF=${U_INF:-0.10}
mkdir -p "$OUT"

run_one() {
  local uid="$1" stl="$2"
  if [ -f "$OUT/les_${uid}_result.json" ]; then
    echo "=== skip $uid (이미 있음) ==="
    return 0
  fi
  echo "=== $(date -u +%FT%TZ) START $uid ==="
  ~/venv_xlb/bin/python les_service_run.py --uid "$uid" --stl "$stl" --grade "$GRADE" --u-inf "$U_INF" \
    2>&1 | tee "$OUT/${uid}.out"
  local rc=${PIPESTATUS[0]}
  echo "=== $(date -u +%FT%TZ) DONE $uid rc=$rc ==="
  for f in "runs/les_${uid}_result.json" "les_${uid}_result.json"; do
    [ -f "$f" ] && cp "$f" "$OUT/les_${uid}_result.json" && break
  done
  return $rc
}

for car in $CARS; do
  # baseline first: the variants reuse the refinement boxes it derives
  unset LES_FIX_REFINE_JSON
  run_one "${car}_base" "$STL_DIR/${car}_base.stl"

  # LES_FIX_REFINE_JSON holds the JSON itself, not a path; the runner prints it
  # on a "refine-derived" line
  refine=$(grep -h "refine-derived" "$OUT/${car}_base.out" 2>/dev/null | tail -1 |
           sed "s/.*refine-derived] //")
  if [ -n "$refine" ]; then
    export LES_FIX_REFINE_JSON="$refine"
    echo "$refine" > "$OUT/${car}_refine.json"
    echo "[campaign] $car 정제 격자 고정: $refine"
  else
    echo "[campaign] 중단: $car 의 refine-derived 를 못 읽었다."
    echo "[campaign] 격자를 고정하지 않으면 변종마다 격자가 달라져 노이즈가 4배가 되고,"
    echo "[campaign] v8/v9 처럼 라벨을 통째로 못 쓰게 된다. 여기서 멈춘다."
    continue
  fi

  for variant in blunt_tail wide_rear raise_roof; do
    run_one "${car}_${variant}" "$STL_DIR/${car}_${variant}.stl"
  done
done

touch "$OUT/CAMPAIGN_DONE"
echo "[campaign] 전체 완료"
