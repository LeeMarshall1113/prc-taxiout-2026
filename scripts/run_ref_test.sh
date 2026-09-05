#!/usr/bin/env bash
# Paired test of the EUROCONTROL reference feature: same 3 folds, with and
# without it. Small enough to fit a contended box.
set -uo pipefail
REPO="C:/Users/hackathon/Documents/GitHub/prc-taxiout-2026"
BROKER="C:/Users/hackathon/.compute-broker/broker.py"
LEASE="${1:?usage: run_ref_test.sh <lease-id>}"
python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1
( while true; do sleep 300; python "$BROKER" heartbeat "$LEASE" >/dev/null 2>&1 || \
    { echo "[heartbeat] yield requested or lease lost" >&2; touch "$REPO/.yield-requested"; }; done ) &
HB=$!
# Kill the whole process group on exit: a bare TaskStop on this wrapper left an
# orphaned trainer running last time.
trap 'kill $HB 2>/dev/null; pkill -f "prc.crossval" 2>/dev/null; python "$BROKER" release "$LEASE" >/dev/null 2>&1' EXIT

cd "$REPO"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=./.venv/Scripts/python.exe
REF="ref_taxi_s,ref_level"
COMMON="--folds 1,3,5 --iterations 800 --threads 4 --drop"

run () {
  [ -f "$REPO/.yield-requested" ] && { echo "[ref-test] yielding before $1"; return 1; }
  echo; echo "########## $1 ##########"
  # shellcheck disable=SC2086
  $PY -u -m prc.crossval --tag "$1" $COMMON "$2" || echo "[ref-test] $1 FAILED"
}

run noref "airport_plan,$REF"
run ref   "airport_plan"

echo; echo "########## paired result ##########"
$PY -u - <<'PYEOF'
import json, pathlib, numpy as np
rows = {r["tag"]: r for r in (json.loads(l) for l in
        pathlib.Path("results/crossval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
if "noref" in rows and "ref" in rows:
    a = np.array([f["rmse"] for f in rows["noref"]["folds"]])
    b = np.array([f["rmse"] for f in rows["ref"]["folds"]])
    ab = np.array([f["rmse_bulk"] for f in rows["noref"]["folds"]])
    bb = np.array([f["rmse_bulk"] for f in rows["ref"]["folds"]])
    print(f"{'fold':<10} {'no ref':>10} {'with ref':>10} {'delta':>9} | {'bulk no':>9} {'bulk ref':>9} {'delta':>8}")
    for i, f in enumerate(rows["ref"]["folds"]):
        print(f"{str(f['fold']):<10} {a[i]:>10.2f} {b[i]:>10.2f} {b[i]-a[i]:>+9.2f} | "
              f"{ab[i]:>9.2f} {bb[i]:>9.2f} {bb[i]-ab[i]:>+8.2f}")
    print(f"{'mean':<10} {a.mean():>10.2f} {b.mean():>10.2f} {b.mean()-a.mean():>+9.2f} | "
          f"{ab.mean():>9.2f} {bb.mean():>9.2f} {bb.mean()-ab.mean():>+8.2f}")
    print(f"\nfolds won by ref: {int((b < a).sum())}/{len(a)} overall, {int((bb < ab).sum())}/{len(ab)} on bulk")
else:
    print("one config did not finish; have:", sorted(rows))
PYEOF
