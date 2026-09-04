#!/usr/bin/env bash
# Run training under a compute-broker lease with a real heartbeat daemon.
# A heartbeat driven from the work loop dies in the gaps; this one does not.
set -uo pipefail

REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_train.sh <lease-id> [train args...]}"
shift

python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
(
  while true; do
    sleep 300
    if ! python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1; then
      echo "[heartbeat] broker asked us to yield, or the lease is gone" >&2
      touch "$REPO/.yield-requested"
    fi
  done
) &
HB=$!
trap 'kill $HB 2>/dev/null; python "$BROKER" release "$LEASE" >/dev/null 2>&1' EXIT

cd "$REPO"
PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
  ./.venv/Scripts/python.exe -m prc.train "$@"
STATUS=$?
echo "[run_train] training exited with $STATUS"
exit $STATUS
