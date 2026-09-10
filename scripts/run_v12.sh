#!/usr/bin/env bash
# v12: v11's configuration plus --noplan-depth 4.
#
# The no-plan specialist has run at depth 6 since it was written, a number
# picked without measurement. A sweep of its variance knobs -- nine configs over
# three folds, costing 40s each because rmse on no-plan rows depends only on the
# specialist and the global model can be skipped entirely:
#
#   config       (1,7)    (3,9)   (5,11)   d(1,7)
#   ctrl        1926.2   1048.3   1863.7     +0.0
#   depth4      1866.3    989.1   1867.2    -59.8
#   combo       1867.4    959.1   1865.7    -58.8   (four knobs stacked)
#   l2_10       1890.4   1043.7   1890.6    -35.8
#   bag3        1909.2   1039.0   1886.8    -17.0
#   l2_30       1948.3   1095.4   1926.1    +22.1
#
# Stacking four knobs gained nothing over depth alone, so the effect IS depth.
# L2 hurt and bagging was worth ~1s, which kills the two hypotheses the residual
# autopsy pointed at -- its "92% instability" diagnosis was right about the
# symptom and wrong about the cure.
#
# depth 4 was tested on 2026-09-06 and lost. That grid ran BEFORE airport
# routing, so it tuned one model over 22,470 rows; Rome's specialist now fits
# about 1,200 and depth 6 overfits it. Routing silently invalidated the tuning
# and nobody re-ran it.
#
# Stacking with --bulk-plan-only is safe here in a way it was not for v10: that
# flag changes which rows the GLOBAL model trains on, and the specialist trains
# on the raw target and never sees the global model's residual. Disjoint models.
#
# Expected, holding the bulk at fold (1,7)'s control: board 300.3 -> ~295 from
# the depth change, on top of whatever v11 delivers.
set -uo pipefail

REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_v12.sh <lease-id>}"
LOG="$REPO/logs/v12.log"

cd "$REPO"
mkdir -p logs
PY=./.venv/Scripts/python.exe

$PY "$BROKER" wait "$LEASE" --timeout 7200 || {
    echo "[wait] not granted" | tee -a "$LOG"; exit 1; }
rm -f "$REPO/.yield-requested"
$PY "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do sleep 240; $PY "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || exit 0; done ) &
HB=$!
trap 'kill $HB 2>/dev/null' EXIT

export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage

WAVE2="dep_rwy_30min,dep_rwy_headway,arr_taxi_60min,sched_demand_30min"
WX="wx_temp_c,wx_dewpoint_c,wx_wind_kt,wx_gust_kt,wx_visibility_km,wx_ceiling_ft,wx_precip,wx_heavy_precip,wx_deice_risk,wx_thunder,wx_snow,wx_freezing,wx_lowvis,wx_age_min"

{
  echo "=== v12 start $(date -u +%H:%M:%SZ) ==="
  "$PY" -u -m prc.final \
      --seeds 3 --iterations 2500 --threads 6 \
      --noplan-model --noplan-routes LIRF --noplan-depth 4 \
      --bulk-plan-only \
      --residual --baseline blend_apt \
      --keep "$WAVE2,$WX,callsign_op" \
      --upload
  echo "=== v12 exit $? at $(date -u +%H:%M:%SZ) ==="
} 2>&1 | tee -a "$LOG"

kill $HB 2>/dev/null
$PY "$BROKER" release "$LEASE" >/dev/null 2>&1
echo "lease released at $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
