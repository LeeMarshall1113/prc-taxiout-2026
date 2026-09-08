#!/usr/bin/env bash
# Three cheap ideas, each with a distinct mechanism, each paired against wx.
set -uo pipefail
REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_three.sh <lease-id>}"
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
  [ -f "$REPO/.yield-requested" ] && { echo "[three] yielding before $1"; return 1; }
  echo; echo "########## $1 ##########"
  $PY -u -m prc.crossval --tag "$1" --folds 1,3,5 --iterations 2500 --threads 6 \
      --seeds 1 --noplan-model --drop "$DROP" "${@:2}" || echo "[three] $1 FAILED"
}

# my own unexecuted idea: delete rows proven to be timestamp errors
run dayoff  --drop-dayoffset
# the continuous version of the same thing, without falsifying any label
run huber   --loss "Huber:delta=2000"
# runway configuration, reconstructed from movements we already hold
run rwycfg  --drop "airport_plan,ref_taxi_s,ref_level"

echo; echo "########## all three vs wx ##########"
$PY -u - <<'PYEOF'
import json, pathlib, numpy as np
by = {}
for l in pathlib.Path("results/crossval.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip():
        r = json.loads(l); by[r["tag"]] = r
b = np.array([f["rmse_bagged"] for f in by["wx"]["folds"]])
print(f"wx baseline   {'  '.join(f'{x:8.2f}' for x in b)}   mean {b.mean():8.2f}")
for tag in ("dayoff", "huber", "rwycfg"):
    if tag in by:
        a = np.array([f["rmse_bagged"] for f in by[tag]["folds"]])
        d = a - b
        print(f"{tag:<13} {'  '.join(f'{x:8.2f}' for x in a)}   mean {a.mean():8.2f}   "
              f"delta {d.mean():+7.2f}   wins {int((d<0).sum())}/{len(d)}")
PYEOF
