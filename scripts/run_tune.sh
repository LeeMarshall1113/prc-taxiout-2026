#!/usr/bin/env bash
# Phase 1: tune the no-plan model (cheap sweep on cached global predictions).
# Phase 2: retest the features rejected while the metric was 61% tail.
set -uo pipefail
REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_tune.sh <lease-id>}"
python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do sleep 300; python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || \
    { echo "[heartbeat] yield requested" >&2; touch "$REPO/.yield-requested"; }; done ) &
HB=$!
trap 'kill $HB 2>/dev/null; pkill -f "prc.tune_noplan" 2>/dev/null; pkill -f "prc.crossval" 2>/dev/null; python "$BROKER" release "$LEASE" >/dev/null 2>&1' EXIT

cd "$REPO"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=./.venv/Scripts/python.exe
WAVE2="dep_rwy_30min,dep_rwy_headway,arr_taxi_60min,sched_demand_30min"
REF="ref_taxi_s,ref_level"

echo "########## phase 1: tune the no-plan model ##########"
$PY -u -m prc.tune_noplan --folds 1,3,5 --iterations 2500 --threads 6

[ -f "$REPO/.yield-requested" ] && { echo "[tune] yielding before phase 2"; exit 0; }

echo; echo "########## phase 2: retest features against the fixed metric ##########"
echo "Rejected when the metric was 61% tail; the split removed that noise."
run () {
  [ -f "$REPO/.yield-requested" ] && { echo "[tune] yielding before $1"; return 1; }
  echo; echo "--- $1 ---"
  # shellcheck disable=SC2086
  $PY -u -m prc.crossval --tag "$1" --folds 1,3,5 --iterations 2500 --threads 6 \
      --seeds 1 --noplan-model --drop "$2" || echo "[tune] $1 FAILED"
}
run split_wave2 "airport_plan,$REF"
run split_ref   "airport_plan,$WAVE2"
run split_all   "airport_plan"

echo; echo "########## phase 2 summary vs both2500 ##########"
$PY -u - <<'PYEOF'
import json, pathlib, numpy as np
by = {}
for l in pathlib.Path("results/crossval.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip():
        r = json.loads(l); by[r["tag"]] = r
if "both2500" in by:
    b = np.array([f["rmse_bagged"] for f in by["both2500"]["folds"]])
    print(f"both2500 (shipped as v5): mean {b.mean():.2f}   folds {'  '.join(f'{x:.1f}' for x in b)}")
    for tag in ("split_wave2", "split_ref", "split_all"):
        if tag in by:
            d = np.array([f["rmse_bagged"] for f in by[tag]["folds"]]) - b
            print(f"  {tag:<12} {'  '.join(f'{x:+7.2f}' for x in d)}   mean {d.mean():+7.2f}   wins {int((d<0).sum())}/{len(d)}")
PYEOF
