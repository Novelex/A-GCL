#!/bin/bash
# Deliberately run ONE unit with an invalid input. Wrapped so it cannot fail the harness.
set -uo pipefail
S7=/users/3171356m/agcl_audit_s0/s7
OUT=$S7/smoke/EXPFAIL; rm -rf "$OUT"; mkdir -p "$OUT"
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache
/users/3171356m/A-GCL/.venv/bin/python - <<'PY' > "$OUT/worker.log" 2>&1
import sys; sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C
OUT=C.S7+"smoke/EXPFAIL"
C.write_unit(OUT,"bad_unit",payload_npz=dict(h=C.load_all()["NODE"]["NOPE"]))
PY
rc=$?
echo "worker exit code: $rc"
echo "worker last line: $(tail -1 "$OUT/worker.log")"
n_done=$(find "$OUT" -maxdepth 1 -name '*.DONE' | wc -l)
n_tmp=$(find "$OUT" -maxdepth 1 -name '*tmp*' | wc -l)
n_final=$(find "$OUT" -maxdepth 1 \( -name '*.npz' -o -name '*.json' \) | wc -l)
echo "DONE sentinels: $n_done   leftover temp: $n_tmp   promoted finals: $n_final"
chk=$(/users/3171356m/A-GCL/.venv/bin/python -c "
import sys; sys.path.insert(0,'/users/3171356m/agcl_audit_s0/s7'); import s7_core as C
print(C.is_done(C.S7+'smoke/EXPFAIL','bad_unit'))")
echo "completion checker reports unit complete: $chk"
if [ "$rc" -ne 0 ] && [ "$n_done" -eq 0 ] && [ "$n_final" -eq 0 ] && [ "$n_tmp" -eq 0 ] && [ "$chk" = "False" ]; then
  echo "EXPECTED_FAILURE_PATH_PASS"; touch "$S7/smoke/EXPECTED_FAILURE_PATH_PASS"; exit 0
fi
echo "EXPECTED_FAILURE_PATH_FAIL"; exit 1
