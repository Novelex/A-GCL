"""S16 Pass 4, PHASE 1: reproduce D53-D56 BEFORE fixing.

Named `repro_*` deliberately: it asserts the DEFECT IS PRESENT, so it exits 0 now
and 1 once the repairs land. It is not a regression test and must never carry the
`test_` prefix (that is D56 itself)."""
import sys, os, json, shutil, subprocess, tempfile, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
R = []
def repro(n, present, d=""):
    R.append((n, bool(present)))
    print(f"  [{'REPRODUCED' if present else 'NOT REPRODUCED':<14}] {n}" + (f" | {d}" if d else ""))

import s16_prov as P, s16_report as RP, s16_ledger as L

# ---------------------------------------------------------------- D53
print("=== D53: a missing/null `remaining` falsely passes completion ===")
NS="test"; UID="u"
def unit(**tally):
    shutil.rmtree(P.root(NS), ignore_errors=True); P.ensure(NS)
    jd=P.jobs_dir(NS)+UID; os.makedirs(jd, exist_ok=True)
    t=dict(unit=UID, namespace=NS, expected=9, validated_reused=0,
           newly_successful=9, failed=0, remaining=0); t.update(tally)
    for k in [k for k,v in t.items() if v is ...]: t.pop(k)
    json.dump(t, open(jd+"/TALLY.json","w"))
    json.dump(dict(state="done"), open(jd+"/STATUS.json","w"))
    open(jd+"/UNIT.done","w").write("d")
    return P.validate_unit_completion(NS, UID, 9)

ok,why = unit(remaining=...)                    # key absent entirely
repro("absent `remaining` accepted", ok, f"validate_unit_completion -> {ok}, why={why}")
ok,why = unit(remaining=None)                   # explicit null
repro("null `remaining` accepted", ok, f"-> {ok}")
ok,why = unit(remaining=False)                  # bool: False == 0 in python
repro("boolean False `remaining` accepted", ok, f"-> {ok}")
ok,why = unit(failed=False)                     # bool masquerading as 0
repro("boolean False `failed` accepted", ok, f"-> {ok}")
ok,why = unit(expected=...)                     # expected absent
repro("absent `expected` behaviour", not ok, f"-> {ok} (absent expected IS caught: {why[:1]})")
ok,why = unit(validated_reused="0", newly_successful="9")
repro("string counts accepted or crash", ok, f"-> {ok} why={why[:1]}")
shutil.rmtree(P.root(NS), ignore_errors=True)

# ---------------------------------------------------------------- D54
print("\n=== D54: `unpaired` is accepted but never used ===")
import inspect
src = inspect.getsource(RP.shift_gate)
body = src.split('"""')[-1]
repro("shift_gate takes `unpaired` and ignores it",
      "unpaired" in inspect.signature(RP.shift_gate).parameters and "unpaired" not in body,
      "the parameter never appears in the function body after the docstring")
tab = pd.DataFrame([dict(arch="BNT", protocol="lab", n_pairs=10, paired_diff=0.001)])
f = RP.shift_gate(tab, ["unpaired cells: 300 signed, 299 shift, 299 paired"])
repro("a nonempty unpaired list produces NO failure", f == [], f"shift_gate -> {f}")

# ---------------------------------------------------------------- D55
print("\n=== D55: incomplete evaluation rows survive collection/reporting ===")
csrc = open(f"{HERE}/s16_collect.py").read(); rsrc = open(f"{HERE}/s16_report.py").read()
import s16_collect as C, inspect
audit_src = inspect.getsource(C.audit)          # the ACTUAL acceptance path
repro("collector never checks the per-cell eval-point SET",
      "eval_point" not in audit_src,
      "audit() validates the bundle but never which evaluation points the record "
      "carries; `eval_point` exists only as a CSV column name in build_rows")
repro("collector does not require head/head_ema/probe_* to exist",
      not any(x in audit_src for x in ("head_ema", "REQUIRED_EVAL", "eval_contract")),
      "REQUIRED_FINITE covers only svm_tr_enc/svm_tr_full/size_delta_paired")
repro("collector does not require movement_max/clip_rate finite",
      "movement_max" not in audit_src,
      "they reach the CSV unvalidated")
repro("collector does not verify evaluated_state",
      "evaluated_state" not in audit_src,
      "it is only ever copied into the CSV, never compared to the frozen protocol")
repro("report checks only that (unit,fold) appears somewhere",
      "got = {(u,f) for u,f in zip(df.unit, df.fold)}" in rsrc,
      "a cell missing 3 of its 4 evaluation rows still satisfies this")

# ---------------------------------------------------------------- D56
print("\n=== D56: an expected-failing file carries the test_ prefix ===")
exists = os.path.exists(f"{HERE}/test_pass3_repro.py")
rc = subprocess.run([sys.executable, f"{HERE}/test_pass3_repro.py"],
                    capture_output=True, text=True,
                    env={**os.environ, "PYTHONPYCACHEPREFIX":"/users/3171356m/agcl_audit_s0/pycache"}
                    ).returncode if exists else None
repro("test_pass3_repro.py exits nonzero on CORRECT code", exists and rc != 0,
      f"exists={exists} exit={rc} -> standard `test_*` discovery reports a failure")

n = sum(1 for _,v in R if v)
print(f"\n{n}/{len(R)} defects reproduced")
sys.exit(0 if n == len(R) else 1)
