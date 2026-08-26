"""S16 Pass 3, PHASE 3: adversarial tests proving each Phase-2 correction.

D47 C2 fold structure   D48 unit-completion contract
D49 affine transport (replaces the withdrawn AUC gate)   D50 C-PERM wording
D51 ledger-derived arithmetic

No scientific data fitting: every fixture is synthetic."""
import sys, os, json, glob, shutil, tempfile, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
OK = []
def ck(n, c, d=""):
    OK.append(bool(c)); print(f"  [{'PASS' if c else 'FAIL'}] {n}" + (f" | {d}" if d else ""))

import s16_prov as P, s16_policy as PL, s16_ledger as L, s16_grid as G
import s16_c2_bounded_run as RUN, s16_report as RP

# ---------------------------------------------------------------- D47
print("=== D47: C2 fold structure ===")
y_ref = np.zeros(954, np.int64); ids = [f"s{i}" for i in range(954)]
def mk(d, folds, extra=None):
    for i, (tr, te) in enumerate(folds):
        z = dict(y=y_ref, repr=np.zeros((954, 32), np.float32),
                 tr=np.asarray(tr), te=np.asarray(te))
        if extra: z.update(extra)
        np.savez(f"{d}/f{i}.npz", **z)
    return f"{d}/f*.npz"
def canonical():
    b = [list(range(i*191, min((i+1)*191, 954))) for i in range(5)]
    return [(sorted(set(range(954)) - set(t)), t) for t in b]

with tempfile.TemporaryDirectory() as d:
    _, prob, sigs = RUN.validate_source("good", mk(d, canonical()), "y", "repr", 32, y_ref, ids)
    ck("a canonical 5-fold partition validates", not prob, f"{prob[:1]}")
    ck("five distinct signatures returned", sigs and len(set(sigs)) == 5, f"{len(set(sigs or []))}")

with tempfile.TemporaryDirectory() as d:              # THE Phase-1 defect
    one = canonical()[0]
    _, prob, _ = RUN.validate_source("dup5", mk(d, [one]*5), "y", "repr", 32, y_ref, ids)
    ck("FIVE COPIES of one fold rejected",
       any("do NOT have 5 distinct test sets" in p for p in prob),
       [p for p in prob if "distinct" in p][:1])
    ck("the un-tested subjects are named",
       any("never tested" in p for p in prob), "union != 0..953 is reported")

with tempfile.TemporaryDirectory() as d:              # overlapping folds
    f = canonical(); tr0, te0 = f[0]; tr1, te1 = f[1]
    f[1] = (tr1, sorted(set(te1) | {te0[0]}))
    _, prob, _ = RUN.validate_source("ov", mk(d, f), "y", "repr", 32, y_ref, ids)
    ck("overlapping test sets rejected", any("overlap in" in p for p in prob),
       [p for p in prob if "overlap in" in p][:1])

with tempfile.TemporaryDirectory() as d:              # missing subjects
    f = canonical(); tr4, te4 = f[4]; f[4] = (tr4, te4[:-3])
    _, prob, _ = RUN.validate_source("miss", mk(d, f), "y", "repr", 32, y_ref, ids)
    ck("missing subjects rejected", any("never tested" in p for p in prob),
       [p for p in prob if "never tested" in p][:1])

with tempfile.TemporaryDirectory() as d:              # wrong train complement
    f = canonical(); tr0, te0 = f[0]; f[0] = (tr0[:-5], te0)
    _, prob, _ = RUN.validate_source("cmp", mk(d, f), "y", "repr", 32, y_ref, ids)
    ck("tr not the complement of te rejected",
       any("complement" in p or "not an exact partition" in p for p in prob),
       [p for p in prob if "complement" in p or "partition" in p][:1])

# cross-source membership + zero-fitting, through the REAL main()
print("\n=== D47: cross-source membership, and zero fitting on any invalid source ===")
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
CALLS = {"n": 0}; _real = K.probe_pipe
def counting(*a, **kw):
    CALLS["n"] += 1; return _real(*a, **kw)
K.probe_pipe = counting; RUN.K.probe_pipe = counting

orig = list(RUN.SOURCES)
# Use the REAL cohort labels so the ONLY thing wrong is fold MEMBERSHIP. Using
# zeros made both sources fail on "labels differ", which would have let this test
# pass for the wrong reason.
import s16_c2_bounded as CB
y_real, sites_real, ids_real = CB.cohort()
def mk_real(d, folds):
    for i,(tr,te) in enumerate(folds):
        np.savez(f"{d}/f{i}.npz", y=y_real.astype(np.int64),
                 repr=np.zeros((954,32), np.float32),
                 tr=np.asarray(tr), te=np.asarray(te))
    return f"{d}/f*.npz"
def blocks(off):
    b=[[j for j in range(954) if (j+off)//191 == i and (j+off) < 955] for i in range(5)]
    b=[[j for j in range(954)][i*191:(i+1)*191] for i in range(5)]
    if off:                                  # rotate membership between the two sources
        flat=[x for g in b for x in g]; flat=flat[off:]+flat[:off]
        b=[flat[i*191:(i+1)*191] for i in range(5)]
    return [(sorted(set(range(954))-set(t)), sorted(t)) for t in b]

tmp = tempfile.mkdtemp(); RUN.OUT = tmp + "/"
d1, d2 = tempfile.mkdtemp(), tempfile.mkdtemp()
RUN.SOURCES = [("SRC-A [CALIBRATION]", mk_real(d1, blocks(0)), "y", "repr", 32),
               ("SRC-B",               mk_real(d2, blocks(7)), "y", "repr", 32)]
CALLS["n"] = 0
try: RUN.main(); rc = 0
except SystemExit as e: rc = e.code
res = json.load(open(tmp + "/C2_BOUNDED.json"))
probs = res.get("problems", [])
ck("each source is individually VALID (only membership differs)",
   not any("labels differ" in p or "never tested" in p or "partition" in p for p in probs),
   f"per-source problems: {[p for p in probs if 'differs from' not in p][:1]}")
ck("cross-source fold membership mismatch is caught",
   any("fold membership differs" in p for p in probs),
   [p for p in probs if "fold membership differs" in p][:1])
ck("halts BEFORE fitting (exit 6), not at calibration (exit 5)", rc == 6,
   f"exit={rc} — a cross-source mismatch must stop the run before any source is fitted")
ck("ZERO probe_pipe calls when any source is invalid", CALLS["n"] == 0,
   f"calls={CALLS['n']}")
ck("no source reports OK", all(r["status"] in ("SOURCE_VALIDATION_FAILED","NOT_RUN")
                               for r in res["results"]),
   f"{sorted({r['status'] for r in res['results']})}")
ck("halt names the cross-source cause", "cross-source fold-membership" in str(res.get("halted")),
   str(res.get("halted"))[:100])
RUN.SOURCES = orig; K.probe_pipe = _real
for x in (tmp, d1, d2): shutil.rmtree(x, ignore_errors=True)

# ---------------------------------------------------------------- D48
print("\n=== D48: unit-completion contract (shared by collector and E2E) ===")
NS = "test"; UID = "unit_x"; F = 9
def build_unit(**over):
    jd = P.jobs_dir(NS) + UID; os.makedirs(jd, exist_ok=True)
    t = dict(unit=UID, namespace=NS, expected=F, validated_reused=0,
             newly_successful=F, newly_attempted=F, failed=0, remaining=0)
    t.update(over.pop("tally", {}))
    json.dump(t, open(jd + "/TALLY.json", "w"))
    json.dump(dict(state=over.pop("state", "done")), open(jd + "/STATUS.json", "w"))
    open(jd + "/UNIT.done", "w").write("done")
    return jd
shutil.rmtree(P.root(NS), ignore_errors=True); P.ensure(NS)
jd = build_unit()
ok, why = P.validate_unit_completion(NS, UID, F)
ck("a complete unit validates", ok, f"{why}")

for name, mutate, expect in (
    ("UNIT.done removed",   lambda j: os.remove(j + "/UNIT.done"),               "UNIT.done absent"),
    ("TALLY.json removed",  lambda j: os.remove(j + "/TALLY.json"),              "TALLY.json absent"),
    ("STATUS.json removed", lambda j: os.remove(j + "/STATUS.json"),             "STATUS.json absent"),
    ("STATUS not terminal", lambda j: json.dump(dict(state="running"), open(j + "/STATUS.json", "w")), "not the terminal"),
    ("unit POISON present", lambda j: open(j + "/POISON", "w").write("mass failure"), "POISON"),
    ("TALLY corrupted",     lambda j: open(j + "/TALLY.json", "w").write("{not json"), "unreadable"),
):
    shutil.rmtree(P.root(NS), ignore_errors=True); P.ensure(NS)
    j = build_unit(); mutate(j)
    ok2, why2 = P.validate_unit_completion(NS, UID, F)
    ck(f"rejects: {name}", (not ok2) and any(expect in w for w in why2), "; ".join(why2)[:88])

for name, tally, expect in (
    ("failed != 0",             dict(failed=2),                              "failed=2"),
    ("remaining != 0",          dict(remaining=3),                           "remaining=3"),
    ("accounting identity off", dict(newly_successful=F - 1),                "accounting identity violated"),
    ("expected folds wrong",    dict(expected=F - 1),                        "expected"),
):
    shutil.rmtree(P.root(NS), ignore_errors=True); P.ensure(NS)
    build_unit(tally=tally)
    ok3, why3 = P.validate_unit_completion(NS, UID, F)
    ck(f"rejects: {name}", (not ok3) and any(expect in w for w in why3), "; ".join(why3)[:88])

shutil.rmtree(P.root(NS), ignore_errors=True); P.ensure(NS)
build_unit(); open(P.poison_path(NS), "w").write("namespace poisoned")
ok4, why4 = P.validate_unit_completion(NS, UID, F)
ck("rejects: GLOBAL namespace POISON", (not ok4) and any("GLOBAL POISON" in w for w in why4),
   "; ".join(why4)[:88])
shutil.rmtree(P.root(NS), ignore_errors=True)

esrc = open(f"{HERE}/_e2e_check.py").read(); csrc = open(f"{HERE}/s16_collect.py").read()
ck("E2E checker calls the SHARED validator", "P.validate_unit_completion(" in esrc)
ck("collector calls the SAME shared validator", "P.validate_unit_completion(" in csrc,
   "one definition, so the two cannot diverge")
ck("E2E requires the frozen evaluated state", "evaluated_state_" in esrc
   and "selection by VALIDATION only" in esrc,
   "raw-vs-EMA is fixed by protocol, never chosen after seeing results")
ck("E2E requires head_ema finite", '"head","head_ema"' in esrc)

# ---------------------------------------------------------------- D49
print("\n=== D49: affine transport — the DETERMINISTIC test that replaces the AUC gate ===")
import torch, s16_models as MO, s16_train as TR

# BNT's first affine layer is inp = Linear(D,H): z = W x + b.
# Under x' = (x+1)/2 we need W' x' + b' == W x + b for ALL x.
#   W'(x+1)/2 + b' = W x + b   =>   W' = 2W   and   b' = b - W.1
D, H, KC, N, R = 93, 96, 4, 48, 90
rng = np.random.default_rng(5)
X  = rng.normal(size=(N, R, D)).astype(np.float32)
Xs = ((X + 1.0) / 2.0).astype(np.float32)            # the shift transform
A  = rng.normal(size=(N, R, R)).astype(np.float32)
FC = ((A + A.transpose(0, 2, 1)) / 2).astype(np.float32)
for i in range(N): np.fill_diagonal(FC[i], 1.0)

def transported(seed, correct=True):
    m0 = MO.build_model("BNT", D, seed, KC, H=H); m0.eval()
    m1 = MO.build_model("BNT", D, seed, KC, H=H); m1.eval()
    m1.load_state_dict(m0.state_dict())            # ALL other params copied unchanged
    with torch.no_grad():
        W = m0.inp.weight.detach().clone(); b = m0.inp.bias.detach().clone()
        if correct:
            m1.inp.weight.copy_(2.0 * W); m1.inp.bias.copy_(b - W.sum(dim=1))
        else:
            m1.inp.weight.copy_(2.0 * W); m1.inp.bias.copy_(b)   # forgot -W.1
    return m0, m1

m0, m1 = transported(20260818, correct=True)
r0, s0 = TR.extract(m0, X,  FC, np.arange(N), False)
r1, s1 = TR.extract(m1, Xs, FC, np.arange(N), False)
dr = float(np.abs(r0 - r1).max()); ds = float(np.abs(s0 - s1).max())
scale = float(max(np.abs(r0).max(), 1.0))
# float32 accumulation over D=93 and H=96 with two attention blocks; 1e-4 relative
# is strict for this depth and is justified below by the incorrect-transport contrast.
TOL = 1e-4 * scale
ck("transported BNT reproduces the ORIGINAL representation",
   dr <= TOL, f"max|dr| = {dr:.3e} <= tol {TOL:.3e} (repr scale {scale:.3f})")
ck("transported BNT reproduces the ORIGINAL logits",
   ds <= 1e-4 * float(max(np.abs(s0).max(), 1.0)),
   f"max|dlogit| = {ds:.3e}")

mb0, mb1 = transported(20260818, correct=False)
rb0, sb0 = TR.extract(mb0, X,  FC, np.arange(N), False)
rb1, sb1 = TR.extract(mb1, Xs, FC, np.arange(N), False)
dbad = float(np.abs(rb0 - rb1).max())
ck("an INCORRECT transport (bias not corrected) FAILS the same test",
   dbad > TOL, f"max|dr| = {dbad:.3e} > tol {TOL:.3e} — the test has real power "
               f"({dbad/max(dr,1e-12):.0f}x the correct-transport residual)")

rsrc = open(f"{HERE}/s16_report.py").read()
ck("the AUC-magnitude hard gate is GONE",
   "SHIFT_BNT_TOL = 0.01" not in rsrc and "within_tol" not in rsrc,
   "affine equivalence no longer decides headline validity")
ck("shift AUC survives as a DESCRIPTIVE diagnostic",
   "DESCRIPTIVE ONLY" in rsrc and "shift_vs_signed" in rsrc)
ck("pair completeness is still mandatory",
   "def shift_gate" in rsrc and "not well posed" in rsrc,
   "zero-pair groups still stop the report")

# ---------------------------------------------------------------- D50
print("\n=== D50: C-PERM is hold-and-investigate, not proven leakage ===")
ck("the 'pipeline leaks' assertion is gone", "means the pipeline leaks" not in rsrc)
ck("the band is still an operational gate",
   "CPERM_BAND    = (0.45, 0.55)" in rsrc and "def cperm_gate" in rsrc)
for phrase in ("STOP headline", "investigate permutation variance", "class balance",
               "aggregation", "does NOT prove leakage"):
    ck(f"message contains: {phrase!r}", phrase in rsrc)

# ---------------------------------------------------------------- D51
print("\n=== D51: arithmetic derived from the FROZEN ledger ===")
cells, units, tags = L.expected_ledger(); Ff = len(tags)
crand = [u for _, _, u in units if u.get("control") == "C-RAND"]
ctrl  = [u for _, _, u in units if u.get("control")]
byE = {}
for _, _, u in units: byE[u["E"]] = byE.get(u["E"], 0) + 1
ns_cells = sum(v for k, v in byE.items() if k != "signed") * Ff
ck("C-RAND: 6 units x 9 = 54 cells", len(crand) == 6 and len(crand) * Ff == 54,
   f"{len(crand)} units -> {len(crand)*Ff} cells")
ck("all controls: 24 units x 9 = 216 cells", len(ctrl) == 24 and len(ctrl) * Ff == 216,
   f"{len(ctrl)} units -> {len(ctrl)*Ff} cells")
ck("non-signed E levels: 3 of 4", len([k for k in byE if k != "signed"]) == 3 and len(byE) == 4)
ck("810 of 1,431 = 56.6%, NOT two thirds", ns_cells == 810 and len(cells) == 1431
   and abs(ns_cells / len(cells) - 0.566) < 0.001,
   f"{ns_cells}/{len(cells)} = {ns_cells/len(cells)*100:.1f}%; two thirds would be {len(cells)*2//3}")
for f in ("DEFECTS_FOUND.md", "CORRECTION_PASS2_2026-08-25.md"):
    t = open(f"/users/3171356m/A-GCL/audit/s16/{f}").read()
    ck(f"{f}: no 'two thirds' claim", "two thirds" not in t)
ck("DEFECTS_FOUND.md: C-RAND cell count corrected to 54",
   "54 C-RAND cells" in open("/users/3171356m/A-GCL/audit/s16/DEFECTS_FOUND.md").read())

print(f"\n{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
