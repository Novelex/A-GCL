"""S16 Gate 4: intentional-failure tests. Every rejection class must make the
collector exit NONZERO and write NOTHING. Runs in the isolated 'test' namespace."""
import sys, os, json, glob, shutil, copy, subprocess, numpy as np
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_prov as P, s16_ledger as L, s16_grid as G
NS="test"; F=[]
COLLECT=["/users/3171356m/A-GCL/.venv/bin/python",
         "/users/3171356m/A-GCL/audit/s16/scripts/s16_collect.py"]
ENV=dict(os.environ, S16_NS=NS, PYTHONPYCACHEPREFIX="/users/3171356m/agcl_audit_s0/pycache")

def ck(n,ok,d=""):
    print(("PASS " if ok else "FAIL ")+n+(" | "+d if d else ""),flush=True)
    if not ok: F.append(n)

def run_collector():
    r=subprocess.run(COLLECT,capture_output=True,text=True,env=ENV)
    return r.returncode, (r.stdout+r.stderr)

def wrote_csv(): return os.path.exists(P.results_path(NS))

def reset():
    shutil.rmtree(P.root(NS), ignore_errors=True); P.ensure(NS)

def make_cell(uid, tag, status="OK", with_manifest=True, ns=NS, prov_status="OK",
              schema="s16-prov-1", malformed=False):
    jd=P.jobs_dir(ns)+uid; os.makedirs(jd, exist_ok=True)
    rec=dict(status=status, unit=uid, branch="main", arm="A1", E="signed", arch="WGIN",
             mode="plain", seed=20260818, fold=tag,
             fold_protocol=tag.rstrip("0123456789"), namespace=ns,
             head=dict(auc=0.6), probe_honest=dict(auc=0.6),
             probe_old_full=dict(auc=0.6), movement={"inp":0.2})
    p=f"{jd}/fold_{tag}.json"
    if malformed: open(p,"w").write("{not json")
    else: json.dump(dict(rec=rec,curve=[]), open(p,"w"), default=str)
    fp=P.feat_dir(ns)+f"{uid}__{tag}.npz"; np.savez_compressed(fp[:-4], x=np.zeros(2))
    if with_manifest:
        json.dump(dict(schema=schema, namespace=ns, status=prov_status, unit=uid,
                       fold=tag), open(fp+".prov.json","w"))

def make_tally(uid, attempted=9, newly=9, failed=0, reused=0, ns=NS, ok=True):
    jd=P.jobs_dir(ns)+uid; os.makedirs(jd, exist_ok=True)
    json.dump(dict(unit=uid, namespace=ns, attempted=attempted, succeeded=newly,
                   failed=failed, validated_reused=reused, newly_succeeded=newly,
                   expected_folds=9, accounting_ok=ok),
              open(f"{jd}/TALLY.json","w"))

TAGS = L.fold_tags()
UNITS = [u[0] for u in L.all_units()]

def build_complete():
    """A fully complete, clean wave in the test namespace."""
    reset()
    for uid in UNITS:
        for t in TAGS: make_cell(uid,t)
        make_tally(uid)

# ---------------- baseline: complete wave must be ACCEPTED ----------------
print("--- baseline ---")
build_complete()
rc,out = run_collector()
ck("T0_complete_wave_accepted", rc==0 and wrote_csv(),
   f"exit {rc}, csv written={wrote_csv()} | {out.strip().splitlines()[-1][:90] if out.strip() else ''}")

CASES=[]
def case(name, mutate, expect_key):
    build_complete()
    if os.path.exists(P.results_path(NS)): os.remove(P.results_path(NS))
    mutate()
    rc,out = run_collector()
    ok = (rc!=0) and (not wrote_csv()) and (expect_key in out)
    ck(name, ok, f"exit {rc}, csv={wrote_csv()}, key '{expect_key}' "
                 f"{'found' if expect_key in out else 'MISSING'}")
    CASES.append((name,rc,expect_key))

print("\n--- intentional failures ---")
case("R1_missing_unit",
     lambda: shutil.rmtree(P.jobs_dir(NS)+UNITS[0]), "missing_unit")
case("R2_missing_fold",
     lambda: os.remove(P.jobs_dir(NS)+UNITS[0]+f"/fold_{TAGS[0]}.json"), "missing_fold")
case("R3_duplicate_fold",
     lambda: make_cell(UNITS[0], TAGS[0]) or
             shutil.copy(P.jobs_dir(NS)+UNITS[0]+f"/fold_{TAGS[0]}.json",
                         P.jobs_dir(NS)+UNITS[0]+f"/fold_{TAGS[0]}_dup.json"),
     "duplicate_fold")
case("R4_unexpected_cell",
     lambda: make_cell("main_GHOST_signed_plain_s0","lab0"), "unexpected_cell")
case("R5_failed_record",
     lambda: make_cell(UNITS[0],TAGS[0],status="FAILED"), "failed_record")
case("R6_malformed_json",
     lambda: make_cell(UNITS[0],TAGS[0],malformed=True), "malformed_json")
case("R7_provenance_absent",
     lambda: os.remove(P.feat_dir(NS)+f"{UNITS[0]}__{TAGS[0]}.npz.prov.json"),
     "provenance_absent_or_incompatible")
case("R7b_provenance_incompatible",
     lambda: make_cell(UNITS[0],TAGS[0],schema="WRONG-SCHEMA"),
     "provenance_absent_or_incompatible")
case("R8_poison_marker",
     lambda: open(P.poison_path(NS),"w").write("synthetic poison"), "poison_marker")
case("R9_tally_result_disagreement",
     lambda: make_tally(UNITS[0], attempted=9, newly=3), "tally_result_disagreement")
case("R10_skipped_without_validated_reuse",
     lambda: make_tally(UNITS[0], attempted=4, newly=4, reused=0),
     "skipped_without_validated_reuse")
case("R11_wrong_namespace",
     lambda: make_cell(UNITS[0],TAGS[0],ns=NS) or
             json.dump(dict(schema="s16-prov-1",namespace="prod",status="OK"),
                       open(P.feat_dir(NS)+f"{UNITS[0]}__{TAGS[0]}.npz.prov.json","w")),
     "wrong_namespace")

# scheduler COMPLETED is not scientific completion
print("\n--- scheduler COMPLETED is not scientific completion ---")
build_complete()
if os.path.exists(P.results_path(NS)): os.remove(P.results_path(NS))
for t in TAGS: make_cell(UNITS[0],t,status="FAILED")
make_tally(UNITS[0], attempted=9, newly=0, failed=9, reused=0, ok=True)
rc,out = run_collector()
ck("T1_all_folds_failed_but_unit_exited_zero", rc!=0 and not wrote_csv()
   and "failed_record" in out,
   f"exit {rc}; a unit whose every fold FAILED still exits 0 to SLURM and would show "
   f"COMPLETED — the collector refuses anyway")

reset()
print(f"\n=== GATE 4 SUMMARY: {len(F)} failures; {len(CASES)} intentional-failure cases ===")
for n,rc,k in CASES: print(f"  {n:42s} exit {rc}  key={k}")
for f in F: print("  FAIL "+f)
sys.exit(1 if F else 0)
