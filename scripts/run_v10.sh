#!/usr/bin/env bash
# v10: the two changes that each won 3 of 3 folds against the control.
#
#   --bulk-plan-only            bulk RMSE 216.45 -> 214.12  (-2.33, 3/3)
#   --noplan-routes +LFPG,LSZH  no-plan RMSE -34.00 mean    (3/3)
#   callsign_op                 rode with the routing arm, so its own
#                               contribution is not separable -- the bundle won
#
# They act on different halves of the model, so they should stack to about
# -4.7s. Against v8's 300.35 that is roughly 296, using the measured fold(1,7)
# -> board offset of -36.93.
set -uo pipefail

REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_v10.sh <lease-id>}"
LOG="$REPO/logs/v10.log"

cd "$REPO"
mkdir -p logs
PY=./.venv/Scripts/python.exe

echo "=== waiting for $LEASE at $(date -u +%H:%M:%SZ) ===" | tee -a "$LOG"
$PY "$BROKER" wait "$LEASE" --timeout 21600 || {
    echo "[wait] not granted -- stopping" | tee -a "$LOG"; exit 1; }
rm -f "$REPO/.yield-requested"

$PY "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do
    sleep 240
    $PY "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || exit 0
  done ) &
HB=$!
trap 'kill $HB 2>/dev/null' EXIT

export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage

WAVE2="dep_rwy_30min,dep_rwy_headway,arr_taxi_60min,sched_demand_30min"
WX="wx_temp_c,wx_dewpoint_c,wx_wind_kt,wx_gust_kt,wx_visibility_km,wx_ceiling_ft,wx_precip,wx_heavy_precip,wx_deice_risk,wx_thunder,wx_snow,wx_freezing,wx_lowvis,wx_age_min"

{
  echo "=== v10 start $(date -u +%H:%M:%SZ) ==="
  "$PY" -u -m prc.final \
      --seeds 3 --iterations 2500 --threads 6 \
      --noplan-model --noplan-routes LIRF,LFPG,LSZH \
      --bulk-plan-only \
      --residual --baseline blend_apt \
      --keep "$WAVE2,$WX,callsign_op" \
      --upload
  echo "=== v10 exit $? at $(date -u +%H:%M:%SZ) ==="
} 2>&1 | tee -a "$LOG"

kill $HB 2>/dev/null
$PY "$BROKER" release "$LEASE" >/dev/null 2>&1
echo "lease released at $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
