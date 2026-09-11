#!/usr/bin/env bash
# One control run with --save-preds, which is the measurement this campaign has
# been missing since it started.
#
# Twice this week I estimated where the error lives by comparing the truth to a
# stand-in -- a per-airport mean, then a raw timestamp gap -- shipped the result,
# and got zero on the board both times. Tonight I built a third such budget and
# it valued 134 Rome rows at 80 seconds; the board evidence from v9 says the
# model sits within a few hundred seconds of the truth on exactly those rows, so
# the real figure is a fraction of a second. Wrong by about 400x, in the same
# direction as the two failures.
#
# The common cause is that we have never had the model's own out-of-fold
# predictions in a joinable form. `--save-preds` existed but wrote no row id, so
# nothing could be joined to it and every analysis fell back to a proxy. That was
# fixed on 2026-09-10; this is the first run that uses it.
#
# Nothing is decided from this run. It produces the residuals that let the next
# decision be arithmetic instead of guesswork.
set -uo pipefail

REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_savepreds.sh <lease-id>}"
LOG="$REPO/logs/savepreds.log"

cd "$REPO"; mkdir -p logs; PY=./.venv/Scripts/python.exe

echo "=== waiting for $LEASE at $(date -u +%H:%M:%SZ) ===" | tee -a "$LOG"
$PY "$BROKER" wait "$LEASE" --timeout 43200 || {
    echo "[wait] not granted -- stopping" | tee -a "$LOG"; exit 1; }
rm -f "$REPO/.yield-requested"
$PY "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do
    sleep 240
    $PY "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || {
        echo "[heartbeat] yield requested" >&2; touch "$REPO/.yield-requested"; }
  done ) &
HB=$!
trap 'kill $HB 2>/dev/null; $PY "$BROKER" release "$LEASE" >/dev/null 2>&1' EXIT

export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage

# v8/v11's shipped feature set exactly, so the residuals describe the model that
# is actually on the board rather than a neighbour of it.
DROP="airport_plan,ref_taxi_s,ref_level,prev_stand_gap,prev_stand_headway,prev_rwy_gap,mvt_sec_00,callsign_op,arr_sub_day,arr_sub_stand,lobt_minus_aobt"

{
  echo "=== save-preds start $(date -u +%H:%M:%SZ) ==="
  $PY -u -m prc.crossval --tag sp_ctrl --folds 1,3,5 --iterations 2500 \
      --threads 6 --seeds 1 --noplan-model --residual --baseline blend_apt \
      --drop "$DROP" --noplan-routes LIRF --bulk-plan-only --save-preds
  echo "=== save-preds exit $? at $(date -u +%H:%M:%SZ) ==="
} 2>&1 | tee -a "$LOG"

kill $HB 2>/dev/null
$PY "$BROKER" release "$LEASE" >/dev/null 2>&1
echo "lease released at $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
