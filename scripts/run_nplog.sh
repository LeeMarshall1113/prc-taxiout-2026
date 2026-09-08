#!/usr/bin/env bash
set -uo pipefail
REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_nplog.sh <lease-id>}"
python "$BROKER" wait "$LEASE" --timeout 36000 || { echo "[wait] not granted"; exit 1; }
rm -f "$REPO/.yield-requested"
python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do sleep 300; python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || \
    { echo "[heartbeat] yield requested" >&2; touch "$REPO/.yield-requested"; }; done ) &
HB=$!
trap 'kill $HB 2>/dev/null; pkill -f "prc.crossval" 2>/dev/null; python "$BROKER" release "$LEASE" >/dev/null 2>&1' EXIT
cd "$REPO"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
./.venv/Scripts/python.exe -u -m prc.crossval --tag nplog --folds 1,3,5 --iterations 2500 \
  --threads 6 --seeds 1 --noplan-model --noplan-target log --drop "airport_plan,ref_taxi_s,ref_level"
echo; echo "########## no-plan log vs raw (tag: wx) ##########"
./.venv/Scripts/python.exe -u - <<'PYEOF'
import json, pathlib, numpy as np
by = {}
for l in pathlib.Path("results/crossval.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip():
        r = json.loads(l); by[r["tag"]] = r
if "wx" in by and "nplog" in by:
    for key, lab in (("rmse_bagged","overall"), ("rmse_noplan","no-plan")):
        a = np.array([f[key] for f in by["wx"]["folds"]], dtype=float)
        b = np.array([f[key] for f in by["nplog"]["folds"]], dtype=float)
        d = b - a
        print(f"{lab:<8} raw {'  '.join(f'{x:9.2f}' for x in a)}  mean {a.mean():9.2f}")
        print(f"{'':8} log {'  '.join(f'{x:9.2f}' for x in b)}  mean {b.mean():9.2f}  "
              f"delta {d.mean():+8.2f}  wins {int((d<0).sum())}/{len(d)}")
PYEOF
