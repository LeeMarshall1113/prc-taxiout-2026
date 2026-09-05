#!/usr/bin/env bash
# Sequence of training variants under one compute-broker lease.
# Heartbeat is its own daemon: driven from the work loop it dies in the gaps.
set -uo pipefail

REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_ablation.sh <lease-id>}"

python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do sleep 300; python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || \
    { echo "[heartbeat] yield requested or lease lost" >&2; touch "$REPO/.yield-requested"; }; done ) &
HB=$!
trap 'kill $HB 2>/dev/null; python "$BROKER" release "$LEASE" >/dev/null 2>&1' EXIT

cd "$REPO"
export PYTHONIOENCODING=utf-8
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=./.venv/Scripts/python.exe

WAVE2="dep_rwy_30min,dep_rwy_headway,arr_taxi_60min,sched_demand_30min"
OLD="airport_plan,$WAVE2"     # the v1 feature set
NEW="airport_plan"            # v1 + wave 2
COMMON="--iterations 1500 --od-wait 100"

run () {
  local tag="$1"; shift
  if [ -f "$REPO/.yield-requested" ]; then echo "[ablation] yielding before $tag"; return 1; fi
  echo; echo "############ $tag ############"
  # shellcheck disable=SC2086
  $PY -m prc.train --tag "$tag" $COMMON "$@" || echo "[ablation] $tag FAILED"
}

run base        --drop "$OLD"
run w7200       --drop "$OLD" --winsor 7200
run w20k        --drop "$OLD" --winsor 20000
run wave2       --drop "$NEW"
run wave2_w7200 --drop "$NEW" --winsor 7200
run wave2_w20k  --drop "$NEW" --winsor 20000

echo; echo "############ summary (selection metric = trimmed) ############"
$PY - <<'PYEOF'
import json, pathlib
f = pathlib.Path("results/ablation.jsonl")
rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
rows.sort(key=lambda r: r["trimmed_rmse"])
best = rows[0]["trimmed_rmse"]
print(f"{'tag':<14} {'feats':>5} {'winsor':>7} {'trimmed':>9} {'vs best':>8} {'rmse':>9} {'bulk':>7} {'noplan':>8} {'it':>5}")
for r in rows:
    print(f"{r['tag']:<14} {r['features']:>5} {r['winsor']:>7,.0f} {r['trimmed_rmse']:>9.2f} "
          f"{r['trimmed_rmse']-best:>+8.2f} {r['rmse']:>9.2f} {r.get('rmse_bulk',0):>7.1f} "
          f"{r.get('rmse_noplan',0):>8.0f} {r['best_iteration']:>5}")
PYEOF
