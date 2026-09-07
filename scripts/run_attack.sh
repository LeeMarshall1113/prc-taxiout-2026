#!/usr/bin/env bash
# Two changes at once, both aimed at the tail arithmetic:
#   - more iterations (the v4 training curve was still falling at 1100)
#   - a dedicated model for the no-flight-plan rows, which carry ~61% of MSE
set -uo pipefail
REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_attack.sh <lease-id>}"
# A yield flag left by a PREVIOUS run must not silently cancel this one.
# It is a latch with no reset: written when a lease expires, never cleared,
# so every later run acquired its lease and quit without doing anything.
rm -f "$REPO/.yield-requested"
python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do sleep 300; python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || \
    { echo "[heartbeat] yield requested" >&2; touch "$REPO/.yield-requested"; }; done ) &
HB=$!
trap 'kill $HB 2>/dev/null; pkill -f "prc.crossval" 2>/dev/null; python "$BROKER" release "$LEASE" >/dev/null 2>&1' EXIT

cd "$REPO"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=./.venv/Scripts/python.exe
WAVE2="dep_rwy_30min,dep_rwy_headway,arr_taxi_60min,sched_demand_30min"
REF="ref_taxi_s,ref_level"
DROP="airport_plan,$WAVE2,$REF"

run () {
  [ -f "$REPO/.yield-requested" ] && { echo "[attack] yielding before $1"; return 1; }
  echo; echo "########## $1 ##########"
  shift 1
  # shellcheck disable=SC2086
  $PY -u -m prc.crossval --drop "$DROP" --folds 1,3,5 --threads 6 --seeds 1 "$@" \
    || echo "[attack] FAILED"
}

# cheap first: does the dedicated no-plan model help at the current size?
run "noplan @800"    --tag np800  --iterations 800  --noplan-model
# then: is the global model simply underfit?
run "iters 2500"     --tag it2500 --iterations 2500
# and both together
run "both @2500"     --tag both2500 --iterations 2500 --noplan-model

echo; echo "########## summary vs bag3 seed0 (800 iters, no split) ##########"
$PY -u - <<'PYEOF'
import json, pathlib, numpy as np
rows = [json.loads(l) for l in pathlib.Path("results/crossval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
by = {}
for r in rows: by[r["tag"]] = r
ref = by.get("bag3")
print(f"{'tag':<10} {'mean':>9} {'folds':>34}")
for tag in ("bag3", "np800", "it2500", "both2500"):
    if tag in by:
        r = by[tag]; per = "  ".join(f"{f['rmse_bagged']:7.1f}" for f in r["folds"])
        print(f"{tag:<10} {r['mean_rmse']:>9.2f}   {per}")
if ref:
    b = np.array([f["rmse_bagged"] for f in ref["folds"]])
    print("\nper-fold delta vs bag3 (negative = better):")
    for tag in ("np800", "it2500", "both2500"):
        if tag in by:
            d = np.array([f["rmse_bagged"] for f in by[tag]["folds"]]) - b
            print(f"  {tag:<10} {'  '.join(f'{x:+7.2f}' for x in d)}   mean {d.mean():+7.2f}   wins {int((d<0).sum())}/{len(d)}")
PYEOF
