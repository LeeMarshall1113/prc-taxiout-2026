#!/usr/bin/env bash
# Sweep the no-plan specialist's variance knobs. Waits for its lease, runs, exits.
set -uo pipefail
REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_sweep.sh <lease-id>}"
cd "$REPO"; mkdir -p logs; PY=./.venv/Scripts/python.exe
$PY "$BROKER" wait "$LEASE" --timeout 10800 || { echo "[wait] not granted"; exit 1; }
$PY "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do sleep 240; $PY "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || exit 0; done ) &
HB=$!
trap 'kill $HB 2>/dev/null; $PY "$BROKER" release "$LEASE" >/dev/null 2>&1' EXIT
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
{ echo "=== sweep start $(date -u +%H:%M:%SZ) ==="
  $PY -u -m prc.sweep_noplan --folds 1,3,5 --threads 5
  echo "=== sweep exit $? at $(date -u +%H:%M:%SZ) ==="
} 2>&1 | tee -a logs/sweep.log
kill $HB 2>/dev/null
$PY "$BROKER" release "$LEASE" >/dev/null 2>&1
