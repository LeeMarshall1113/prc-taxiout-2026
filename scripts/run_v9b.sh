#!/usr/bin/env bash
# v9, reordered. The substitution classifier runs FIRST.
#
# The original run_v9.sh put three fold tests ahead of it. Those are worth
# single-digit seconds each; the classifier's measured ceiling is ~51 (board
# 288.9 -> 237.8, against a leader at 245.29), so it goes first and the folds
# take whatever lease time is left.
#
# THRESHOLDS, fixed before any result is seen:
#   classifier: beats the band rate on 2+ of 3 folds AND captures >= 25% of the
#               band->oracle gap. Below 25% the mechanism is not resolving rows
#               and the band patch alone is the better ship.
#   fold tests: extend to folds 2,4,6 on 2+ of 3 AND mean better than -1.5s.
set -uo pipefail

REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_v9b.sh <lease-id>}"
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

# crossval's --drop is opt-OUT, so the control must name callsign_op and the
# sequence proxy explicitly or they ride along in every arm.
BASE_DROP="airport_plan,ref_taxi_s,ref_level,prev_stand_gap,prev_stand_headway,prev_rwy_gap,mvt_sec_00"
CTRL_DROP="$BASE_DROP,callsign_op"

guard () { [ -f "$REPO/.yield-requested" ] && { echo "[v9] yielding before $1"; return 1; }; return 0; }

{
  echo "=== v9b start $(date -u +%H:%M:%SZ) ==="

  echo; echo "########## substitution classifier ##########"
  guard classifier && $PY -u -m prc.substitution --fit --folds 1,3,5 \
      --iterations 600 --depth 6 --threads 5 || echo "[v9] classifier FAILED"

  echo; echo "########## fold tests ##########"
  for arm in "v9_base:$CTRL_DROP:LIRF:" \
             "v9_bulk:$CTRL_DROP:LIRF:--bulk-plan-only" \
             "v9_route:$BASE_DROP:LIRF,LFPG,LSZH:"; do
    tag="${arm%%:*}"; rest="${arm#*:}"
    drop="${rest%%:*}"; rest="${rest#*:}"
    routes="${rest%%:*}"; extra="${rest#*:}"
    guard "$tag" || break
    echo; echo "########## $tag ##########"
    $PY -u -m prc.crossval --tag "$tag" --folds 1,3,5 --iterations 2500 \
        --threads 5 --seeds 1 --noplan-model --residual --baseline blend_apt \
        --drop "$drop" --noplan-routes "$routes" $extra \
        || echo "[v9] $tag FAILED"
  done

  echo "=== v9b exit $? at $(date -u +%H:%M:%SZ) ==="
} 2>&1 | tee -a "$LOG"

kill $HB 2>/dev/null
$PY "$BROKER" release "$LEASE" >/dev/null 2>&1
echo "lease released at $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
