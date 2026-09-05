#!/usr/bin/env bash
# Three seeds per fold on three folds, predictions saved so post-processing can
# be swept offline without retraining.
set -uo pipefail
REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_bagging.sh <lease-id>}"
python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do sleep 300; python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || \
    { echo "[heartbeat] yield requested or lease lost" >&2; touch "$REPO/.yield-requested"; }; done ) &
HB=$!
trap 'kill $HB 2>/dev/null; pkill -f "prc.crossval" 2>/dev/null; python "$BROKER" release "$LEASE" >/dev/null 2>&1' EXIT

cd "$REPO"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
WAVE2="dep_rwy_30min,dep_rwy_headway,arr_taxi_60min,sched_demand_30min"
REF="ref_taxi_s,ref_level"

# v1's feature set: everything rejected since is dropped.
./.venv/Scripts/python.exe -u -m prc.crossval --tag bag3 \
  --drop "airport_plan,$WAVE2,$REF" \
  --folds 1,3,5 --iterations 800 --threads 5 --seeds 3 --save-preds
