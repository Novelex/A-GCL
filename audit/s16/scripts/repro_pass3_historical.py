"""S16 Pass 3 reproduction record — HISTORICAL ARTIFACT, NOT A REGRESSION TEST.

Renamed out of the `test_*` namespace in Pass 4 (defect D56): this file asserts that
the D47-D51 defects are PRESENT, so on corrected code it exits 1 by design. Under
standard `test_*` discovery that read as a failing test and would have trained
reviewers to ignore a red result.

It exits 0 only while the defects are present, and 1 once they are fixed. That
inversion is itself the proof the repairs landed; `test_pass3.py` asserts the
corrected behaviour positively, and is the file the regression suite runs.

Run deliberately:  python repro_pass3_historical.py   (exit 1 == fixes are in place)

Each check asserts the DEFECT IS PRESENT. Exit 0 means all five reproduced and the
repairs in Phase 2 are justified. After the repairs this file is expected to FAIL —
that inversion is the proof the fix landed, and test_pass3.py then asserts the
corrected behaviour."""
import sys, os, json, glob, tempfile, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = []
def repro(n, present, detail=""):
    REPRO.append((n, bool(present)))
    print(f"  [{'REPRODUCED' if present else 'NOT REPRODUCED':<14}] {n}" + (f" | {detail}" if detail else ""))

import s16_c2_bounded_run as RUN
import s16_report as RP
import s16_ledger as L

print("=== R1: C2 accepts FIVE COPIES of the same fold as five valid folds ===")
y_ref = np.zeros(954, np.int64); ids = [f"s{i}" for i in range(954)]
with tempfile.TemporaryDirectory() as d:
    te = list(range(800, 954)); tr = list(range(800))     # ONE fold, individually legal
    for i in range(5):                                     # five DIFFERENT filenames
        np.savez(f"{d}/fold{i}.npz", y=y_ref, repr=np.zeros((954, 32), np.float32),
                 tr=np.asarray(tr), te=np.asarray(te))
    _r = RUN.validate_source("dup", f"{d}/fold*.npz", "y", "repr", 32, y_ref, ids)
    folds, prob = _r[0], _r[1]        # post-fix the function also returns fold sigs
    repro("C2 five identical folds accepted", (not prob) and folds is not None and len(folds) == 5,
          f"problems={prob!r}; the same 154 subjects are 'held out' 5x and 800 subjects "
          f"are NEVER scored")

print("\n=== R2: E2E approves a target with a sealed bundle but no completion contract ===")
src = open(f"{HERE}/_e2e_check.py").read()
missing = [k for k in ("TALLY", "STATUS.json", "UNIT.done", "POISON") if k not in src]
repro("E2E checker ignores the unit-completion contract", len(missing) == 4,
      f"absent from _e2e_check.py: {missing}")

print("\n=== R3: BNT shift gate demands equal AUC from independently trained models ===")
import torch, s16_models as MO, s16_train as TR, s16_policy as PL
TINY = PL.ExecPolicy("p3", "test", 3, 1, 2, 1e-5, .10, .05, .05, 16, .999, 5, 20, 90, 1, 0, 0, False)
rng = np.random.default_rng(11); N = 96
X = rng.normal(size=(N, 90, 93)).astype(np.float32)
A = rng.normal(size=(N, 90, 90)).astype(np.float32)
FC = ((A + A.transpose(0, 2, 1)) / 2).astype(np.float32)
for i in range(N): np.fill_diagonal(FC[i], 1.0)
yv = np.array([0, 1] * (N // 2)); tr_i = np.arange(N)
cfg = dict(K_or_hidden=4, lr=1e-3, wd=1e-4, loss="L-BCE", freeze_encoder=False,
           readout="roi", dropout=0.3, H=96)
from sklearn.metrics import roc_auc_score
aucs = []
for seed in (20260818, 20260819):          # SAME data, SAME transform, different init
    m, _, _, _ = TR.train_fold("BNT", X, FC, yv, tr_i, cfg, seed, policy=TINY)
    _, s_ = TR.extract(m, X, FC, np.arange(N), False)
    aucs.append(float(roc_auc_score(yv, s_)))
noise = abs(aucs[0] - aucs[1])
_tol = getattr(RP, "SHIFT_BNT_TOL", None)   # None once the gate is withdrawn
repro("seed-to-seed AUC noise exceeds the +/-0.01 gate",
      _tol is not None and noise > _tol,
      f"identical data, identical E level, only the seed differs: "
      f"AUC {aucs[0]:.4f} vs {aucs[1]:.4f}, |delta|={noise:.4f} vs tol {_tol} "
      f"-> the gate cannot distinguish a broken affine identity from ordinary "
      f"optimisation noise, so it would fail a CORRECT run")
_rs = open(f"{HERE}/s16_report.py").read()
repro("shift AUC tolerance is wired as a hard gate",
      "SHIFT_BNT_TOL = 0.01" in _rs and "sys.exit(4)" in _rs,
      "an AUC-magnitude tolerance feeds gate_failures, which exits 4")

print("\n=== R4: C-PERM message asserts leakage as proven ===")
rsrc = open(f"{HERE}/s16_report.py").read()
repro("C-PERM failure claims leakage is proven", "means the pipeline leaks" in rsrc,
      "an out-of-band permutation result proves investigation is REQUIRED, not that "
      "leakage is the cause")

print("\n=== R5: stale arithmetic in the prose ===")
bad = []
for f in ("DEFECTS_FOUND.md", "CORRECTION_PASS2_2026-08-25.md"):
    p = f"/users/3171356m/A-GCL/audit/s16/{f}"
    if os.path.exists(p):
        t = open(p).read()
        if "two thirds" in t: bad.append(f"{f}: says 'two thirds'")
cells, units, tags = L.expected_ledger()
repro("prose says 'two thirds' where the ledger says 56.6%", bool(bad),
      f"{bad}; 810/{len(cells)} = {810/len(cells)*100:.1f}%, two thirds would be {len(cells)*2//3}")

n = sum(1 for _, _, u in units if u.get("control") == "C-RAND")
print(f"\n  ledger-derived truth: C-RAND {n} units x {len(tags)} = {n*len(tags)} cells; "
      f"all controls {sum(1 for _,_,u in units if u.get('control'))} units x {len(tags)} = "
      f"{sum(1 for _,_,u in units if u.get('control'))*len(tags)} cells")

ok = sum(1 for _, v in REPRO if v)
print(f"\n{ok}/{len(REPRO)} defects reproduced")
sys.exit(0 if ok == len(REPRO) else 1)
