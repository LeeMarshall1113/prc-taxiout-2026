#!/usr/bin/env bash
# Cross-validation under one broker lease, heartbeat as its own daemon.
set -uo pipefail
REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_crossval.sh <lease-id>}"
python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do sleep 300; python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || \
    { echo "[heartbeat] yield requested or lease lost" >&2; touch "$REPO/.yield-requested"; }; done ) &
HB=$!
trap 'kill $HB 2>/dev/null; python "$BROKER" release "$LEASE" >/dev/null 2>&1' EXIT
cd "$REPO"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=./.venv/Scripts/python.exe
WAVE2="dep_rwy_30min,dep_rwy_headway,arr_taxi_60min,sched_demand_30min"

run () {
  [ -f "$REPO/.yield-requested" ] && { echo "[crossval] yielding before $1"; return 1; }
  echo; echo "########## $1 ##########"
  # shellcheck disable=SC2086
  $PY -m prc.crossval --tag "$1" "${@:2}" || echo "[crossval] $1 FAILED"
}

run base   --drop "airport_plan,$WAVE2"
run wave2  --drop "airport_plan"
run w20k   --drop "airport_plan,$WAVE2" --winsor 20000

echo; echo "########## cross-fold comparison ##########"
$PY - <<'PYEOF'
import json, pathlib, numpy as np
rows = [json.loads(l) for l in pathlib.Path("results/crossval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
by = {r["tag"]: r for r in rows}
print(f"{'tag':<8} {'mean RMSE':>10} {'sd':>7} {'mean bulk':>10}   per-fold RMSE")
for t, r in by.items():
    per = "  ".join(f"{f['rmse']:.0f}" for f in r["folds"])
    print(f"{t:<8} {r['mean_rmse']:>10.2f} {r['sd_rmse']:>7.2f} {r['mean_bulk']:>10.2f}   {per}")
if "base" in by:
    print("\nper-fold difference vs base (negative = better):")
    b = np.array([f["rmse"] for f in by["base"]["folds"]])
    for t, r in by.items():
        if t == "base": continue
        d = np.array([f["rmse"] for f in r["folds"]]) - b
        wins = int((d < 0).sum())
        print(f"  {t:<8} {'  '.join(f'{x:+7.1f}' for x in d)}   mean {d.mean():+7.2f}   wins {wins}/{len(d)} folds")
PYEOF
