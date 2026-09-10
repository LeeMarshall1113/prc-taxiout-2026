#!/usr/bin/env bash
# v9 fold tests. Waits for lease prc-taxiout-335, then runs a control and two
# arms on the same three folds.
#
# THRESHOLD, declared before any result is seen, because deciding what counts as
# a win after looking is how this project shipped two regressions in week one:
#
#     extend to folds 2,4,6 only on 2+ of 3 AND a mean gain better than -1.5s
#     ship only on 4+ of 6
#
# Run 1 (bulk) is judged on rmse_bulk: the no-plan side is untouched, so
# attribution is clean. Run 2 (route) is judged on rmse_noplan, for the same
# reason in reverse.
#
# Sizing, measured rather than hoped: ~38 min per fold at one seed, so three
# configs x three folds is ~5.7h against a 5h lease -- tight, hence preemptible
# and ordered so the control lands first. The v8 attempt died of exactly this
# arithmetic done optimistically.
set -uo pipefail

REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE=prc-taxiout-335
LOG="$REPO/logs/v9.log"

cd "$REPO"
mkdir -p logs
PY=./.venv/Scripts/python.exe

echo "=== waiting for $LEASE at $(date -u +%H:%M:%SZ) ===" | tee -a "$LOG"
$PY "$BROKER" wait "$LEASE" --timeout 43200 || {
    echo "[wait] not granted -- stopping" | tee -a "$LOG"; exit 1; }
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

# crossval's --drop is opt-OUT: anything not named here is IN the model. So the
# control must explicitly drop callsign_op and the sequence proxy, or they ride
# along in every arm and the comparison measures nothing. This list is
# final.py's DROP minus the wave-2 and weather columns v8 ships via --keep.
BASE_DROP="airport_plan,ref_taxi_s,ref_level,prev_stand_gap,prev_stand_headway,prev_rwy_gap,mvt_sec_00"
CTRL_DROP="$BASE_DROP,callsign_op"
FOLDS="1,3,5"

run () {
  local tag="$1"; shift
  [ -f "$REPO/.yield-requested" ] && { echo "[v9] yielding before $tag"; return 1; }
  echo; echo "########## $tag ##########"
  $PY -u -m prc.crossval --tag "$tag" --folds "$FOLDS" --iterations 2500 \
      --threads 5 --seeds 1 --noplan-model --residual --baseline blend_apt \
      "$@" || echo "[v9] $tag FAILED"
}

{
  echo "=== v9 start $(date -u +%H:%M:%SZ) ==="

  # Control first: it repeats v8's shipped configuration under today's code, so
  # both arms are compared against a run from the same tree rather than against
  # numbers recorded before four bug fixes landed.
  run v9_base  --drop "$CTRL_DROP" --noplan-routes LIRF
  run v9_bulk  --drop "$CTRL_DROP" --noplan-routes LIRF --bulk-plan-only
  run v9_route --drop "$BASE_DROP" --noplan-routes LIRF,LFPG,LSZH

  echo "=== v9 exit $? at $(date -u +%H:%M:%SZ) ==="
} 2>&1 | tee -a "$LOG"

kill $HB 2>/dev/null
$PY "$BROKER" release "$LEASE" >/dev/null 2>&1
echo "lease released at $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
