#!/usr/bin/env bash
# The bulk campaign. The 98.9% of rows that have a flight plan are now the
# binding constraint: even a perfect model on the other 1.1% leaves the board at
# 201, so nothing below that moves without this half.
#
# Three analyses looked for a bulk lever in the FEATURES and found none. So this
# tests the LEARNER. CatBoost grows oblivious trees -- every node at a given
# depth splits on the same feature -- and LIRF's bulk RMSE is 404 against
# 176-261 at the other nine airports. An airport-specific interaction has to be
# spent at a whole level rather than in one branch.
#
# THRESHOLD, fixed before any result is seen, judged on rmse_bulk because that
# is the half being changed:
#
#     ship on 2+ of 3 folds AND mean better than -1.5s
#
# The control runs FIRST. The last campaign lost its control to a bug and had to
# borrow an older run's numbers, which left the comparison arguable.
set -uo pipefail

REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_bulk.sh <lease-id>}"
LOG="$REPO/logs/bulk.log"

cd "$REPO"
mkdir -p logs
PY=./.venv/Scripts/python.exe

$PY "$BROKER" wait "$LEASE" --timeout 3600 || {
    echo "[wait] not granted" | tee -a "$LOG"; exit 1; }
rm -f "$REPO/.yield-requested"
$PY "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do
    sleep 240
    $PY "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || {
        echo "[heartbeat] yield requested" >&2
        touch "$REPO/.yield-requested"; }
  done ) &
HB=$!
trap 'kill $HB 2>/dev/null; $PY "$BROKER" release "$LEASE" >/dev/null 2>&1' EXIT

export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage

# crossval's --drop is opt-OUT. This is v8's shipped feature set exactly, so the
# control is comparable to the board score we already have for it.
DROP="airport_plan,ref_taxi_s,ref_level,prev_stand_gap,prev_stand_headway,prev_rwy_gap,mvt_sec_00,callsign_op,arr_sub_day,arr_sub_stand"

run () {
  local tag="$1"; shift
  [ -f "$REPO/.yield-requested" ] && { echo "[bulk] yielding before $tag"; return 1; }
  echo; echo "########## $tag ##########"
  $PY -u -m prc.crossval --tag "$tag" --folds 1,3,5 --iterations 2500 \
      --threads 6 --seeds 1 --noplan-model --residual --baseline blend_apt \
      --drop "$DROP" --noplan-routes LIRF "$@" || echo "[bulk] $tag FAILED"
}

{
  echo "=== bulk campaign start $(date -u +%H:%M:%SZ) ==="
  run b_ctrl
  run b_lossguide --grow-policy Lossguide --max-leaves 64
  run b_depthwise --grow-policy Depthwise
  echo "=== bulk campaign exit $? at $(date -u +%H:%M:%SZ) ==="
} 2>&1 | tee -a "$LOG"

kill $HB 2>/dev/null
$PY "$BROKER" release "$LEASE" >/dev/null 2>&1
echo "lease released at $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
