#!/usr/bin/env bash
# Ordered by expected value: preemption keeps cutting these runs, so whatever
# is most likely to matter has to go first.
set -uo pipefail
REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_next.sh <lease-id>}"
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
  [ -f "$REPO/.yield-requested" ] && { echo "[next] yielding before $1 (results so far are already written)"; return 1; }
  echo; echo "########## $1 ##########"
  $PY -u -m prc.crossval --tag "$1" --folds 1,3,5 --iterations 2500 --threads 6 \
      --seeds 1 --noplan-model --drop "$DROP" "${@:2}" || echo "[next] $1 FAILED"
}

run residual --residual              # a board rival at ~291 models this way
run rwycfg                           # runway configuration, already built
run nplirf   --noplan-split-lirf     # two models inside the no-plan group
run huber    --loss "Huber:delta=2000"

echo; echo "########## everything vs wx ##########"
$PY -u - <<'PYEOF'
import json, pathlib, numpy as np
by = {}
for l in pathlib.Path("results/crossval.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip():
        r = json.loads(l); by[r["tag"]] = r
b = np.array([f["rmse_bagged"] for f in by["wx"]["folds"]])
print(f"{'config':<10} {'per fold':>30} {'mean':>8} {'delta':>8} {'wins':>6}")
print(f"{'wx':<10} {'  '.join(f'{x:8.2f}' for x in b):>30} {b.mean():>8.2f} {'--':>8} {'--':>6}")
for tag in ("residual", "rwycfg", "nplirf", "huber", "dayoff"):
    if tag in by:
        a = np.array([f["rmse_bagged"] for f in by[tag]["folds"]])
        if len(a) != len(b):
            print(f"{tag:<10} incomplete ({len(a)} folds)"); continue
        d = a - b
        print(f"{tag:<10} {'  '.join(f'{x:8.2f}' for x in a):>30} {a.mean():>8.2f} {d.mean():>+8.2f} {int((d<0).sum())}/3")
PYEOF
