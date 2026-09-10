#!/usr/bin/env bash
# Four arms, three folds each, in the order they matter.
#
#   b_ctrl        v8's exact 50-feature configuration -- the control, first,
#                 because the last two campaigns had to borrow one
#   b_combo       what v10 SHOULD have been: bulk-plan-only + routing at
#                 LFPG/LSZH + callsign_op, through the shared prepare_residual
#                 so the filter-then-baseline order matches what is measured
#   b_lossguide   per-node splits instead of oblivious trees
#   b_depthwise   the other per-node policy
#
# THRESHOLDS, fixed before any result exists:
#
#   b_combo      ship on 2+ of 3 folds AND mean rmse_bagged better than -1.5s.
#                Judged on the OVERALL metric, not on bulk or no-plan alone,
#                because it changes both halves and the parts already won
#                separately -- that is exactly what misled me into shipping it.
#   b_lossguide  judged on rmse_bulk, 2+ of 3 and mean better than -1.5s.
#   b_depthwise  same as lossguide.
#
# Note for whoever kills this: TaskStop leaves the shell running. Twice now an
# orphaned wrapper has carried on to its next arm, once unbrokered. Kill the
# pid.
set -uo pipefail

REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_combo_lossguide.sh <lease-id>}"
LOG="$REPO/logs/combo.log"

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

# crossval --drop is opt-OUT. This list is v8's shipped set exactly.
DROP_CTRL="airport_plan,ref_taxi_s,ref_level,prev_stand_gap,prev_stand_headway,prev_rwy_gap,mvt_sec_00,callsign_op,arr_sub_day,arr_sub_stand"
# the combination keeps callsign_op, as v10 did
DROP_COMBO="airport_plan,ref_taxi_s,ref_level,prev_stand_gap,prev_stand_headway,prev_rwy_gap,mvt_sec_00,arr_sub_day,arr_sub_stand"

run () {
  local tag="$1"; local drop="$2"; local routes="$3"; shift 3
  [ -f "$REPO/.yield-requested" ] && { echo "[campaign] yielding before $tag"; return 1; }
  echo; echo "########## $tag ##########"
  $PY -u -m prc.crossval --tag "$tag" --folds 1,3,5 --iterations 2500 \
      --threads 8 --seeds 1 --noplan-model --residual --baseline blend_apt \
      --drop "$drop" --noplan-routes "$routes" "$@" \
      || echo "[campaign] $tag FAILED"
}

{
  echo "=== campaign start $(date -u +%H:%M:%SZ) ==="
  run b_ctrl       "$DROP_CTRL"  "LIRF"
  run b_combo      "$DROP_COMBO" "LIRF,LFPG,LSZH" --bulk-plan-only
  run b_lossguide  "$DROP_CTRL"  "LIRF" --grow-policy Lossguide --max-leaves 64
  run b_depthwise  "$DROP_CTRL"  "LIRF" --grow-policy Depthwise
  echo "=== campaign exit $? at $(date -u +%H:%M:%SZ) ==="
} 2>&1 | tee -a "$LOG"

kill $HB 2>/dev/null
$PY "$BROKER" release "$LEASE" >/dev/null 2>&1
echo "lease released at $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
