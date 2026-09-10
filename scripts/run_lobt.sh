#!/usr/bin/env bash
# The LOBT arms. Runs ALONGSIDE lease 344's campaign on the 5 GB that was free,
# with 5 threads against its 8, because this is the stronger lead and waiting
# six hours for it costs more than both running slightly slower.
#
# The finding: on 630 rows of fold (1,7) the airport reports a taxi of 4,192s
# where the Network Manager sees 1,384s, and LOBT -- NM's last CALCULATED
# off-block -- sits at 4,381s, almost exactly on the target. gap_lobt correlates
# 0.618 with the target on those rows against gap_aobt's -0.061. They hold 16.6%
# of the fold's squared error and are mostly EGLL, not LIRF.
#
# As a pure arithmetic substitution, threshold chosen out of fold each time
# (converging on 4,200s in all six independently): wins 6/6, mean -18.79s.
#
# THRESHOLD for shipping, fixed before any result exists, judged on rmse_bagged
# against b_ctrl from lease 344:
#
#     2+ of 3 folds AND mean better than -3.0s
#
# A higher bar than the usual -1.5s because the arithmetic estimate is -18.79s:
# if a refit cannot clear -3.0 then the mechanism is not surviving contact with
# the model and I would rather know that than ship a third untested stack.
set -uo pipefail

REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_lobt.sh <lease-id>}"
LOG="$REPO/logs/lobt.log"

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
        touch "$REPO/.yield-lobt"; }
  done ) &
HB=$!
trap 'kill $HB 2>/dev/null; $PY "$BROKER" release "$LEASE" >/dev/null 2>&1' EXIT

export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage

# v8's shipped set. lobt_minus_aobt is dropped in the baseline arm and kept in
# the feature arm, so the two arms differ in exactly one thing each.
DROP_BASE="airport_plan,ref_taxi_s,ref_level,prev_stand_gap,prev_stand_headway,prev_rwy_gap,mvt_sec_00,callsign_op,arr_sub_day,arr_sub_stand,lobt_minus_aobt"
DROP_FEAT="airport_plan,ref_taxi_s,ref_level,prev_stand_gap,prev_stand_headway,prev_rwy_gap,mvt_sec_00,callsign_op,arr_sub_day,arr_sub_stand"

run () {
  local tag="$1"; local drop="$2"; shift 2
  [ -f "$REPO/.yield-lobt" ] && { echo "[lobt] yielding before $tag"; return 1; }
  echo; echo "########## $tag ##########"
  $PY -u -m prc.crossval --tag "$tag" --folds 1,3,5 --iterations 2500 \
      --threads 5 --seeds 1 --noplan-model --residual \
      --drop "$drop" --noplan-routes LIRF "$@" || echo "[lobt] $tag FAILED"
}

{
  echo "=== lobt arms start $(date -u +%H:%M:%SZ) ==="
  # the feature arm first: it keeps blend_apt, so it is the smaller change and
  # the one that can ship alongside everything else already validated
  run lobt_feat "$DROP_FEAT" --baseline blend_apt
  run lobt_base "$DROP_BASE" --baseline lobt_switch
  echo "=== lobt arms exit $? at $(date -u +%H:%M:%SZ) ==="
} 2>&1 | tee -a "$LOG"

kill $HB 2>/dev/null
$PY "$BROKER" release "$LEASE" >/dev/null 2>&1
echo "lease released at $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
