#!/usr/bin/env bash
# v11: --bulk-plan-only, and nothing else new.
#
# The global model stops training on the 22,470 no-flight-plan rows whose
# predictions it discards anyway. It is the one change that improved the bulk on
# every fold in both campaigns it appeared in:
#
#   v9_bulk   (1,7) 240.14 -> 235.50   (3,9) 207.58 -> 206.52   (5,11) 201.64 -> 200.36
#   b_combo   (1,7) 238.72 -> 232.53   (3,9) 207.47 -> 205.43   (5,11) 201.63 -> 199.20
#
# What v11 is NOT: the LFPG/LSZH routing from b_combo. That arm cost fold (1,7)
# its no-plan half (1892 -> 1980) and turned a bulk win into an overall loss of
# +3.62 on the only fold whose months match the scored set. Dropped.
#
# callsign_op IS kept, because the 232.53 bulk above was measured with it. It
# has never been validated on its own -- it rode into b_combo bundled with the
# routing -- so this ships it on the strength of the bundle's bulk number, and
# that is stated rather than hidden.
#
# Expected, by recombining the measured halves with the no-plan fraction backed
# out of each control fold: fold (1,7) 334.59 -> 330.27, wins 3/3, mean -2.62s.
# Through the measured -34.24 offset that is about 296 against v8's 300.35.
# It is an estimate from two fits, not a run.
set -uo pipefail

REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_v11.sh <lease-id>}"
LOG="$REPO/logs/v11.log"

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
  echo "=== v11 start $(date -u +%H:%M:%SZ) ==="
  "$PY" -u -m prc.final \
      --seeds 3 --iterations 2500 --threads 6 \
      --noplan-model --noplan-routes LIRF \
      --bulk-plan-only \
      --residual --baseline blend_apt \
      --keep "$WAVE2,$WX,callsign_op" \
      --upload
  echo "=== v11 exit $? at $(date -u +%H:%M:%SZ) ==="
} 2>&1 | tee -a "$LOG"

kill $HB 2>/dev/null
$PY "$BROKER" release "$LEASE" >/dev/null 2>&1
echo "lease released at $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
