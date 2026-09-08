#!/usr/bin/env bash
# Tune the GLOBAL model. depth 8 / lr 0.08 / l2 3 were day-one guesses and have
# never been measured, while the model they configure makes 98.5% of predictions.
set -uo pipefail
REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_hparam.sh <lease-id>}"
python "$BROKER" wait "$LEASE" --timeout 36000 || { echo "[wait] not granted"; exit 1; }
rm -f "$REPO/.yield-requested"
python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do sleep 300; python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || \
    { echo "[heartbeat] yield requested" >&2; touch "$REPO/.yield-requested"; }; done ) &
HB=$!
trap 'kill $HB 2>/dev/null; pkill -f "prc.crossval" 2>/dev/null; python "$BROKER" release "$LEASE" >/dev/null 2>&1' EXIT

cd "$REPO"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=./.venv/Scripts/python.exe
DROP="airport_plan,ref_taxi_s,ref_level"

run () {
  [ -f "$REPO/.yield-requested" ] && { echo "[hparam] yielding before $1"; return 1; }
  echo; echo "########## $1 ##########"
  $PY -u -m prc.crossval --tag "$1" --folds 1,3,5 --threads 6 --seeds 1 \
      --noplan-model --drop "$DROP" "${@:2}" || echo "[hparam] $1 FAILED"
}

# deeper trees: more feature interactions on 1.7M rows
run hp_d10  --depth 10 --lr 0.08 --iterations 2500
# slower and longer: the standard way to buy accuracy from a GBDT
run hp_slow --depth 8  --lr 0.04 --iterations 5000
# deeper AND slower
run hp_both --depth 10 --lr 0.04 --iterations 5000

echo; echo "########## global hyperparameters vs the shipped d8/lr0.08/2500 (wx) ##########"
$PY -u - <<'PYEOF'
import json, pathlib, numpy as np
by = {}
for l in pathlib.Path("results/crossval.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip():
        r = json.loads(l); by[r["tag"]] = r
b  = np.array([f["rmse_bagged"] for f in by["wx"]["folds"]])
bb = np.array([f["rmse_bulk"]   for f in by["wx"]["folds"]])
print(f"{'config':<10} {'overall per fold':>30} {'mean':>8} {'d':>7}  wins | {'bulk mean':>9} {'d':>7}")
print(f"{'wx (d8)':<10} {'  '.join(f'{x:8.2f}' for x in b):>30} {b.mean():>8.2f} {'--':>7}   --  | {bb.mean():>9.2f} {'--':>7}")
for tag in ("hp_d10", "hp_slow", "hp_both"):
    if tag in by:
        a  = np.array([f["rmse_bagged"] for f in by[tag]["folds"]])
        ab = np.array([f["rmse_bulk"]   for f in by[tag]["folds"]])
        d = a - b
        print(f"{tag:<10} {'  '.join(f'{x:8.2f}' for x in a):>30} {a.mean():>8.2f} {d.mean():>+7.2f} {int((d<0).sum())}/3 | "
              f"{ab.mean():>9.2f} {ab.mean()-bb.mean():>+7.2f}")
PYEOF
