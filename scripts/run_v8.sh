#!/usr/bin/env bash
# v8: airport routing (5/6, -10.20s) + per-airport baseline (3/3, -1.30s)
# + the minute_of_day overflow fix + today's stand_runway_pair_n train/serve fix.
#
# Lease prc-taxiout-331 is ALREADY GRANTED -- the earlier wrapper took it and
# then never started the work, so this attaches to the existing lease rather
# than queueing behind the machine again. Heartbeat runs from its own daemon
# (a work-loop heartbeat has lost two leases before).
set -u

REPO=/c/Users/hackathon/Documents/GitHub/prc-taxiout-2026
BROKER=/c/Users/hackathon/.compute-broker/broker.py
LEASE=prc-taxiout-331
LOG="$REPO/logs/v8.log"

cd "$REPO"
mkdir -p logs
rm -f "$REPO/.yield-requested"

PY="$REPO/.venv/Scripts/python.exe"

( while true; do
    sleep 240
    "$PY" "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || exit 0
  done ) &
HB=$!
trap 'kill $HB 2>/dev/null' EXIT

export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage

WAVE2="dep_rwy_30min,dep_rwy_headway,arr_taxi_60min,sched_demand_30min"
WX="wx_temp_c,wx_dewpoint_c,wx_wind_kt,wx_gust_kt,wx_visibility_km,wx_ceiling_ft,wx_precip,wx_heavy_precip,wx_deice_risk,wx_thunder,wx_snow,wx_freezing,wx_lowvis,wx_age_min"

{
  echo "=== v8 start $(date -u +%H:%M:%SZ) ==="
  "$PY" -u -m prc.final \
      --seeds 3 --iterations 2500 --threads 5 \
      --noplan-model --noplan-split-lirf \
      --residual --baseline blend_apt \
      --keep "$WAVE2,$WX" \
      --upload
  echo "=== v8 exit $? at $(date -u +%H:%M:%SZ) ==="
} 2>&1 | tee -a "$LOG"

kill $HB 2>/dev/null
"$PY" "$BROKER" release "$LEASE" >/dev/null 2>&1
echo "lease released"
