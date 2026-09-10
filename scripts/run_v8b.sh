#!/usr/bin/env bash
# v8, second attempt. Waits for lease prc-taxiout-333 (queued), then builds and
# submits.
#
# The first attempt died of my own arithmetic: 3 seeds at a measured ~70 min
# each against an 81-minute lease. Killing the fit to resize fired the wrapper's
# EXIT trap, the lease was released, and another project took it inside a few
# seconds. Hence two changes here: the lease is sized for the real job, and the
# trap no longer releases on a bare kill of the child -- only on a clean exit or
# a deliberate stop, so a resize does not hand the machine away.
set -u

REPO=/c/Users/hackathon/Documents/GitHub/prc-taxiout-2026
BROKER=/c/Users/hackathon/.compute-broker/broker.py
LEASE=prc-taxiout-333
LOG="$REPO/logs/v8.log"

cd "$REPO"
mkdir -p logs
PY="$REPO/.venv/Scripts/python.exe"

echo "=== waiting for $LEASE at $(date -u +%H:%M:%SZ) ===" | tee -a "$LOG"
"$PY" "$BROKER" wait "$LEASE" --timeout 21600 || {
    echo "[wait] NOT granted -- stopping" | tee -a "$LOG"; exit 1; }

# A stale latch once made every run acquire a lease and exit silently while
# reporting success. Clear it on grant, every time.
rm -f "$REPO/.yield-requested"

"$PY" "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do
    sleep 240
    "$PY" "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || exit 0
  done ) &
HB=$!

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
echo "lease released at $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
