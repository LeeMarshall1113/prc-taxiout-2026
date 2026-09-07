#!/usr/bin/env bash
# Does METAR weather earn its place? Paired against the current best config.
set -uo pipefail
REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_weather.sh <lease-id>}"
python "$BROKER" wait "$LEASE" --timeout 36000 || { echo "[wait] not granted"; exit 1; }
python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do sleep 300; python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || \
    { echo "[heartbeat] yield requested" >&2; touch "$REPO/.yield-requested"; }; done ) &
HB=$!
trap 'kill $HB 2>/dev/null; pkill -f "prc.crossval" 2>/dev/null; python "$BROKER" release "$LEASE" >/dev/null 2>&1' EXIT

cd "$REPO"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=./.venv/Scripts/python.exe
REF="ref_taxi_s,ref_level"
WX=$($PY -c "from prc.weather import FEATURES; print(','.join(FEATURES))")

run () {
  [ -f "$REPO/.yield-requested" ] && { echo "[weather] yielding before $1"; return 1; }
  echo; echo "########## $1 ##########"
  $PY -u -m prc.crossval --tag "$1" --folds 1,3,5 --iterations 2500 --threads 6 \
      --seeds 1 --noplan-model --drop "$2" || echo "[weather] $1 FAILED"
}

run nowx "airport_plan,$REF,$WX"
run wx   "airport_plan,$REF"

echo; echo "########## does weather earn its place? ##########"
$PY -u - <<'PYEOF'
import json, pathlib, numpy as np
by = {}
for l in pathlib.Path("results/crossval.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip():
        r = json.loads(l); by[r["tag"]] = r
if "nowx" in by and "wx" in by:
    a = np.array([f["rmse_bagged"] for f in by["nowx"]["folds"]])
    b = np.array([f["rmse_bagged"] for f in by["wx"]["folds"]])
    ab = np.array([f["rmse_bulk"] for f in by["nowx"]["folds"]])
    bb = np.array([f["rmse_bulk"] for f in by["wx"]["folds"]])
    print(f"{'fold':<10} {'no weather':>11} {'weather':>10} {'delta':>9} | {'bulk no':>9} {'bulk wx':>9} {'delta':>8}")
    for i, f in enumerate(by["wx"]["folds"]):
        print(f"{str(f['fold']):<10} {a[i]:>11.2f} {b[i]:>10.2f} {b[i]-a[i]:>+9.2f} | {ab[i]:>9.2f} {bb[i]:>9.2f} {bb[i]-ab[i]:>+8.2f}")
    print(f"{'mean':<10} {a.mean():>11.2f} {b.mean():>10.2f} {b.mean()-a.mean():>+9.2f} | {ab.mean():>9.2f} {bb.mean():>9.2f} {bb.mean()-ab.mean():>+8.2f}")
    print(f"\nfolds won by weather: {int((b<a).sum())}/{len(a)} overall, {int((bb<ab).sum())}/{len(ab)} on bulk")
else:
    print("incomplete; have:", sorted(by))
PYEOF
