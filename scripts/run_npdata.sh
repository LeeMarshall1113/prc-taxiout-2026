#!/usr/bin/env bash
# Does the no-plan model do better with 2.08M rows than with 22k?
set -uo pipefail
REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_npdata.sh <lease-id>}"
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
  [ -f "$REPO/.yield-requested" ] && { echo "[npdata] yielding before $1"; return 1; }
  echo; echo "########## $1 ##########"
  $PY -u -m prc.crossval --tag "$1" --folds 1,3,5 --iterations 2500 --threads 6 \
      --seeds 1 --noplan-model --drop "$DROP" "${@:2}" || echo "[npdata] $1 FAILED"
}

run np_all      --noplan-train all      --noplan-iterations 1200
run np_weighted --noplan-train weighted --noplan-iterations 1200 --noplan-weight 20

echo; echo "########## vs the shipped no-plan model (tag: wx) ##########"
$PY -u - <<'PYEOF'
import json, pathlib, numpy as np
by = {}
for l in pathlib.Path("results/crossval.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip():
        r = json.loads(l); by[r["tag"]] = r
if "wx" in by:
    b = np.array([f["rmse_bagged"] for f in by["wx"]["folds"]])
    bn = np.array([f["rmse_noplan"] for f in by["wx"]["folds"]])
    print(f"wx (22k rows, shipped): mean {b.mean():.2f}   no-plan RMSE {'  '.join(f'{x:.0f}' for x in bn)}")
    for tag in ("np_all", "np_weighted"):
        if tag in by:
            d = np.array([f["rmse_bagged"] for f in by[tag]["folds"]]) - b
            n = np.array([f["rmse_noplan"] for f in by[tag]["folds"]])
            print(f"  {tag:<12} overall {'  '.join(f'{x:+7.2f}' for x in d)}  mean {d.mean():+7.2f}  wins {int((d<0).sum())}/{len(d)}")
            print(f"  {'':12} no-plan {'  '.join(f'{x:7.0f}' for x in n)}  (was {'  '.join(f'{x:.0f}' for x in bn)})")
PYEOF
