"""Pass 2, P11: bounded C2 must validate ALL seven sources before fitting ANY.

D42: validation and fitting shared one loop, so sources 1-6 were fitted before
     source 7 was found invalid — hours burned, and a partial source set reported.
D43: ids_ref was accepted and never used; equal labels do not prove equal subject
     order.
D44: the fold check (disjoint, total 954) does not prove a partition of 0..953 —
     a duplicate inside tr offsets a missing subject, so one subject is scored
     twice and another never.
This runs the REAL main() with probe_pipe instrumented; the counter proves no
fitting happened."""
import sys, os, json, types, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OK=[]
def check(c,m): OK.append(bool(c)); print(f"  [{'PASS' if c else 'FAIL'}] {m}")

import s16_c2_bounded_run as RUN
import s16_c2_bounded as CB
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K

CALLS={"n":0}
_real=K.probe_pipe
def counting_probe(*a, **kw):
    CALLS["n"] += 1
    return _real(*a, **kw)
K.probe_pipe = counting_probe
RUN.K.probe_pipe = counting_probe

print("=== 1. source seven invalid -> ZERO probe calls ===")
orig=list(RUN.SOURCES)
RUN.SOURCES = orig[:6] + [("S15 B1 BNT K=32 (terminated)",
                           "/nonexistent/path/that/cannot/match_*.npz",
                           "y_true","repr",4096)]
outdir=tempfile.mkdtemp(); RUN.OUT=outdir+"/"
CALLS["n"]=0
try:
    RUN.main(); rc=0
except SystemExit as e: rc=e.code
check(rc==6, f"main() halts with exit 6 (got {rc})")
check(CALLS["n"]==0, f"probe_pipe called {CALLS['n']} times — must be 0: no source "
                     f"may be fitted while any source is invalid")
res=json.load(open(outdir+"/C2_BOUNDED.json"))
check("halted" in res, f"halt is recorded: {str(res.get('halted'))[:90]}")
sts={r["status"] for r in res["results"]}
check(sts <= {"SOURCE_VALIDATION_FAILED","NOT_RUN"},
      f"no source reports OK; statuses = {sorted(sts)}")
check(sum(1 for r in res["results"] if r["status"]=="NOT_RUN")==6,
      "the six valid sources are recorded as NOT_RUN, not as results")
RUN.SOURCES=orig

print("\n=== 2. D43: ids_ref is actually used ===")
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "s16_c2_bounded_run.py")).read()
check("ids_ref" in src and "subject order differs from the frozen cohort" in src,
      "subject order is compared against the frozen cohort when ids are recorded")

print("\n=== 3. D44: exact partition of 0..953 ===")
y_ref=np.zeros(954,np.int64); ids=[f"s{i}" for i in range(954)]
def mkfold(d, tr, te, extra=None):
    """Write fold 0 as the (tr,te) under test and folds 1-4 as a valid disjoint
    cover of the remainder, so the ONLY thing being probed is fold 0.

    D47: this used to write the SAME (tr,te) five times. Pass 3 correctly rejects
    that (five copies of one fold is not a 5-fold cross-validation), so the helper
    itself had to stop encoding the defect."""
    te0=sorted(set(int(x) for x in te))
    rest=[i for i in range(954) if i not in set(te0)]
    q=[rest[j::4] for j in range(4)]                 # 4 disjoint blocks of the rest
    allte=[te0]+[sorted(b) for b in q]
    for i,t_ in enumerate(allte):
        z=dict(y=y_ref, repr=np.zeros((954,32),np.float32),
               tr=(np.asarray(tr) if i==0 else
                   np.asarray(sorted(set(range(954))-set(t_)))),
               te=np.asarray(t_))
        if extra: z.update(extra)
        np.savez(f"{d}/f{i}.npz", **z)
    return f"{d}/f*.npz"
with tempfile.TemporaryDirectory() as d:
    # D47: five DISTINCT, pairwise-disjoint folds whose union is exactly 0..953.
    # mkfold wrote the same (tr,te) five times, which Pass 3 correctly now rejects.
    good=list(range(800)), list(range(800,954))
    pat=mkfold(d, *good)
    folds,prob,_sigs=RUN.validate_source("t",pat,"y","repr",32,y_ref,ids)
    check(not prob, f"a true partition validates ({prob[:1]})")
with tempfile.TemporaryDirectory() as d:
    # duplicate inside tr offsets a missing subject: disjoint AND totals 954
    tr=list(range(799))+[0]; te=list(range(800,954))   # 800+154=954, disjoint
    check(len(tr)+len(te)==954 and not (set(tr)&set(te)),
          "the crafted split IS disjoint and DOES total 954 — the old check passes it")
    pat=mkfold(d, tr, te)
    folds,prob,_sigs=RUN.validate_source("t",pat,"y","repr",32,y_ref,ids)
    check(any("duplicate indices inside tr" in p for p in prob),
          f"duplicate inside tr is caught ({[p for p in prob if 'duplicate' in p][:1]})")
    check(any("not an exact partition" in p for p in prob),
          "the non-partition is named explicitly")
with tempfile.TemporaryDirectory() as d:
    pat=mkfold(d, list(range(800)), list(range(800,954)),
               extra={"ids": np.array(["WRONG"]+ids[1:], dtype="<U8")})
    folds,prob,_sigs=RUN.validate_source("t",pat,"y","repr",32,y_ref,ids)
    check(any("subject order differs" in p for p in prob),
          f"a permuted/renamed subject order is caught even with identical labels")

K.probe_pipe=_real
print(f"\n{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
