"""S16 FINAL PRE-SUBMISSION discriminating tests.
Every rejection asserts its SPECIFIC reason — a generic rejection caused by an
earlier guard is NOT a valid pass. No cluster job, no scientific result."""
import sys, os, json, glob, shutil, copy, subprocess, numpy as np
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_prov as P, s16_ledger as L, s16_grid as G, s16_policy as PL, s16_feat as FT
NS="test"; F=[]
ENV=dict(os.environ,S16_NS=NS,PYTHONPYCACHEPREFIX="/users/3171356m/agcl_audit_s0/pycache")
COLLECT=["/users/3171356m/A-GCL/.venv/bin/python",
         "/users/3171356m/A-GCL/audit/s16/scripts/s16_collect.py"]
def ck(n,ok,d=""):
    print(("PASS " if ok else "FAIL ")+n+(" | "+d if d else ""),flush=True)
    if not ok: F.append(n)

# ---------------- static ----------------
print("--- static ---")
mods=[f for f in glob.glob("*.py")]
rc=subprocess.run([sys.executable,"-m","py_compile",*mods],capture_output=True).returncode
ck("H1_py_compile", rc==0, f"{len(mods)} modules")
bad=[f for f in glob.glob("sb_*.sh")
     if subprocess.run(["bash","-n",f],capture_output=True).returncode!=0]
ck("H2_bash_n", not bad, f"{len(glob.glob('sb_*.sh'))} launchers, {len(bad)} bad")

# ---------------- uniqueness + policy truthfulness ----------------
print("\n--- uniqueness & policy ---")
import _e2e_run as R
T=R.targets()
ck("H3_target_uniqueness", R.assert_targets_unique(T) and len(T)==29, f"{len(T)} unique targets")
uids=[G.unit_id({"main":G.MAIN,"ctrl":G.CTRL,"abl":G.ABL}[b][i]) for b,i,_ in T]
ck("H4_path_uniqueness", len(set(uids))==len(uids), f"{len(set(uids))} unique unit IDs")
import s16_worker as W
ck("H5_policy_truthfulness",
   W.train_consts(PL.get("e2e"))["max_epochs"]==4 and
   W.train_consts(PL.get("prod"))["max_epochs"]==400,
   "e2e record=4, prod record=400, both derived from the object that drives training")
ck("H6_no_import_time_mutation", PL.PROD.max_epochs==400 and len(L.fold_tags())==9,
   f"after importing _e2e_run: {len(L.fold_tags())} fold tags, prod max_epochs unchanged")

# ---------------- legacy refusals ----------------
print("\n--- legacy refusals ---")
for f,label in (("_e2e.py","H7_legacy_e2e_py"),):
    r=subprocess.run([sys.executable,f],capture_output=True,text=True,env=ENV)
    ck(label, r.returncode!=0 and "REFUSED" in r.stderr, f"exit {r.returncode}")
for f,label in (("sb_e2e.sh","H8_legacy_e2e_sh"),("sb_c2.sh","H9_old_c2"),
                ("sb_c2p.sh","H10_old_c2p"),("sb_main.sh","H11_overlap_main"),
                ("sb_abl.sh","H12_overlap_abl"),("sb_ctrl.sh","H13_overlap_ctrl")):
    r=subprocess.run(["bash",f],capture_output=True,text=True)
    ck(label, r.returncode!=0 and "REFUSED" in r.stderr, f"exit {r.returncode}")

# ---------------- realistic collector fixture ----------------
print("\n--- realistic fixture: complete wave, plain + fused ---")
import s16_data as DAT
d,MAN,ent = DAT.load("signed", where="testfix")
# PER-E METADATA (defect D35). The fixture previously built EVERY unit's manifest
# from the signed cache, which encoded the very bug the collector now rejects: the
# 810 abs/pos_zero/shift cells carried signed h_fc and cache_file values.
META = {"signed": (MAN, ent)}
def meta_of(E):
    if E not in META:
        _dd, _m, _e = DAT.load(E, where="testfix")
        META[E] = (_m, _e)
    return META[E]
y=d["y"].astype(np.int64); Xfc=np.zeros((954,4005))
TAGS=L.fold_tags(); UNITS=L.all_units(); pol=PL.get(NS)

def write_cell(uid,u,tag,ns=NS,status="OK",fusion=True,**over):
    jd=P.jobs_dir(ns)+uid; os.makedirs(jd,exist_ok=True)
    seed=G.SEEDS[u["seed_idx"]]
    fp=P.feat_dir(ns)+f"{uid}__{tag}.npz"; ckp=P.ckpt_dir(ns)+f"{uid}__{tag}.pt"
    predp=P.feat_dir(ns)+f"{uid}__{tag}.pred.json"; rp=f"{jd}/fold_{tag}.json"
    np.savez_compressed(fp[:-4], x=np.zeros(4)); open(ckp,"wb").write(b"ckpt")
    rng=np.random.default_rng(abs(hash(uid+tag))%2**31)
    yy=y[:40].copy(); yy[:20]=0; yy[20:]=1
    sf=rng.normal(size=40)+yy*0.8
    from sklearn.metrics import roc_auc_score
    fused_auc=float(roc_auc_score(yy,sf)); svm_enc=0.73; svm_full=0.75
    inner=[dict(alpha=float(a),auc=0.6) for a in FT.ALPHA_GRID]
    fu=dict(alpha_curve=[dict(alpha=float(a),auc=0.6) for a in FT.ALPHA_GRID],
            alpha_curve_inner=inner, alpha_selected=1.0, fused_auc=fused_auc,
            stack_auc=0.7, alpha1_equals_svm_tr_enc=True,
            alpha1_bitwise_equals_zsFC=True, delta_is_unclamped=True,
            delta_vs_svm_tr_enc=fused_auc-svm_enc,
            delta_vs_svm_tr_full=fused_auc-svm_full) if fusion and u["mode"]=="fused" else None
    json.dump(dict(schema="s16-pred-1",unit=uid,fold=tag,label_used=yy.tolist(),
        score_fused=(sf.tolist() if fu else None)), open(predp,"w"))
    rec=dict(status=status,unit=uid,namespace=ns,branch=u["branch"] if "branch" in u else "main",
        arm=u["arm"],arch=u["arch"],E=u["E"],mode=u["mode"],control=u.get("control"),
        alff_mode=u.get("alff_mode"),kh=u["kh"],seed=seed,fold=tag,
        fold_protocol=tag.rstrip("0123456789"),svm_tr_enc=svm_enc,svm_tr_full=svm_full,
        size_delta_paired=svm_full-svm_enc,head=dict(auc=0.6),head_ema=dict(auc=0.59),
        probe_honest=dict(auc=0.61),probe_old_full=dict(auc=0.62),ema_delta=-0.01,
        evaluated_state="raw=validation-best; EMA alongside",movement_max=0.2,
        clip_rate=0.05,verdict="HEALTHY",best_epoch=100,total_steps=2000,
        repr_dim_used=P.expected_repr_dim(u["arch"],u["kh"]),fusion=fu,
        movement={"inp":0.2,"enc":0.2,"head":0.2}, ckpt_sha="x",
        h_fc=meta_of(u["E"])[1]["h_fc"],
        cache_file=meta_of(u["E"])[1]["cache_file"], node="test", wall_s=1.0, n_tr=763)
    rec.update(over)
    json.dump(dict(rec=rec,curve=[]), open(rp,"w"), default=str)
    cfg=W._cfg_for_test(u) if hasattr(W,"_cfg_for_test") else dict(
        K_or_hidden=u["kh"],lr=3e-4,wd=1e-3,loss="L-BCE",
        freeze_encoder=(u.get("control")=="C-RAND"),readout="roi",dropout=0.10,H=128)
    tc=W.train_consts(pol)
    MAN_E, ent_E = meta_of(u["E"])
    man=P.build_manifest(ns,u,cfg,uid,tag,seed,tag.rstrip("0123456789"),MAN_E,ent_E,
        P.expected_repr_dim(u["arch"],u["kh"]),fp,ckp,"OK",tc,policy=pol,
        result_path=rp,pred_path=predp,effective_cfg=P.effective_config(u,cfg))
    man["worktree_clean"]=True                       # fixture: pretend a clean tree
    man["feat_sha"]=P.sha_file(fp); man["ckpt_sha"]=P.sha_file(ckp)
    man["result_sha"]=P.sha_file(rp); man["pred_sha"]=P.sha_file(predp)
    json.dump(man, open(fp+".prov.json","w"), default=str)
    return dict(uid=uid,tag=tag,fp=fp,ckp=ckp,predp=predp,rp=rp,man=fp+".prov.json")

def write_tally(uid,ns=NS,**over):
    jd=P.jobs_dir(ns)+uid; os.makedirs(jd,exist_ok=True)
    t=dict(unit=uid,namespace=ns,expected=9,newly_attempted=9,newly_successful=9,
           failed=0,validated_reused=0,remaining=0,attempted=9,succeeded=9,
           newly_succeeded=9,expected_folds=9,accounting_ok=True); t.update(over)
    json.dump(t,open(f"{jd}/TALLY.json","w"))
    open(f"{jd}/UNIT.done","w").write("done")
    json.dump(dict(state="done",folds_done=9,folds_total=9),open(f"{jd}/STATUS.json","w"))

def build(sub=None):
    shutil.rmtree(P.root(NS),ignore_errors=True); P.ensure(NS)
    hs={}
    for uid,br,u in (sub or UNITS):
        uu=dict(u); uu["branch"]=br
        for t in TAGS: hs[(uid,t)]=write_cell(uid,uu,t)
        write_tally(uid)
    return hs

def collect():
    r=subprocess.run(COLLECT,capture_output=True,text=True,env=ENV)
    return r.returncode,(r.stdout+r.stderr)

print("building 159-unit fixture (this takes a moment) ...",flush=True)
H=build()
rc,out=collect()
csv=P.results_path(NS)
ck("H14_realistic_fixture_accepted", rc==0 and os.path.exists(csv),
   f"exit {rc} | {out.strip().splitlines()[-1][:80] if out.strip() else ''}")
if os.path.exists(csv):
    import pandas as pd
    df=pd.read_csv(csv)
    n_plain=sum(1 for _,_,u in UNITS if u["mode"]=="plain")
    n_fused=sum(1 for _,_,u in UNITS if u["mode"]=="fused")
    exp_rows=(n_plain*9*4)+(n_fused*9*5)     # 4 eval points plain, +fused for fused
    ck("H15_exact_row_count", len(df)==exp_rows, f"{len(df)} rows, expected {exp_rows}")
    import s16_collect as C
    ck("H16_schema_complete", all(c in df.columns for c in C.CSV_REQUIRED),
       f"{len(C.CSV_REQUIRED)} required columns present")
    ck("H17_stale_key_absent", "delta_vs_0p7565_SECONDARY" not in df.columns,
       "delta_vs_0p7565_SECONDARY is gone from the schema")
# ---------------- intentional failures, each asserting its OWN reason -------------
print("\n--- intentional failures (specific reason asserted) ---")
SUB = UNITS[:6]        # small complete sub-wave; ledger gaps are tested separately
def small():
    shutil.rmtree(P.root(NS),ignore_errors=True); P.ensure(NS)
    h={}
    for uid,br,u in SUB:
        uu=dict(u); uu["branch"]=br
        for t in TAGS: h[(uid,t)]=write_cell(uid,uu,t)
        write_tally(uid)
    return h

def case(name, mutate, key, reason_sub):
    h=small(); mutate(h)
    rc,out=collect()
    hit = (rc!=0) and (f"[{key}]" in out) and (reason_sub in out)
    ck(name, hit, f"exit {rc} | key {'ok' if f'[{key}]' in out else 'MISSING'} | "
                  f"reason {'ok' if reason_sub in out else 'MISSING: '+reason_sub}")

u0,_,_ = SUB[0]; t0 = TAGS[0]
case("H18_missing_TALLY", lambda h: os.remove(P.jobs_dir(NS)+u0+"/TALLY.json"),
     "tally_missing", u0)
case("H19_missing_UNIT_done", lambda h: os.remove(P.jobs_dir(NS)+u0+"/UNIT.done"),
     "unit_done_missing", u0)
case("H20_missing_prediction", lambda h: os.remove(h[(u0,t0)]["predp"]),
     "bundle_invalid", "prediction JSON absent")
case("H21_corrupted_prediction",
     lambda h: open(h[(u0,t0)]["predp"],"a").write(" "),
     "bundle_invalid", "prediction JSON hash mismatch")
case("H22_corrupted_result", lambda h: open(h[(u0,t0)]["rp"],"a").write(" "),
     "bundle_invalid", "result JSON hash mismatch")
def _dir_mismatch(h):
    rp=h[(u0,t0)]["rp"]; j=json.load(open(rp)); j["rec"]["unit"]="SOMEONE_ELSE"
    json.dump(j,open(rp,"w"),default=str)
case("H23_result_dir_unit_mismatch", _dir_mismatch, "identity_mismatch",
     "record unit 'SOMEONE_ELSE' != directory")
def _fold_mismatch(h):
    rp=h[(u0,t0)]["rp"]; j=json.load(open(rp)); j["rec"]["fold"]="lab9"
    json.dump(j,open(rp,"w"),default=str)
case("H24_filename_fold_mismatch", _fold_mismatch, "identity_mismatch",
     "!= record fold 'lab9'")
def _wrong_cfg(h):
    m=json.load(open(h[(u0,t0)]["man"])); m["config_hash"]="deadbeefdeadbeef"
    json.dump(m,open(h[(u0,t0)]["man"],"w"),default=str)
case("H25_wrong_grid_config", _wrong_cfg, "bundle_invalid", "config_hash mismatch")
def _wrong_seed(h):
    m=json.load(open(h[(u0,t0)]["man"])); m["seed"]=999
    json.dump(m,open(h[(u0,t0)]["man"],"w"),default=str)
case("H26_wrong_seed", _wrong_seed, "bundle_invalid", "seed mismatch")
def _wrong_dim(h):
    m=json.load(open(h[(u0,t0)]["man"])); m["repr_dim"]=7
    json.dump(m,open(h[(u0,t0)]["man"],"w"),default=str)
case("H27_wrong_repr_dim", _wrong_dim, "bundle_invalid", "repr_dim mismatch")
def _wrong_labels(h):
    m=json.load(open(h[(u0,t0)]["man"])); m["h_labels"]="0000000000000000"
    json.dump(m,open(h[(u0,t0)]["man"],"w"),default=str)
case("H28_label_hash_mismatch", _wrong_labels, "bundle_invalid", "h_labels mismatch")
def _wrong_order(h):
    m=json.load(open(h[(u0,t0)]["man"])); m["h_subject_order"]="0000000000000000"
    json.dump(m,open(h[(u0,t0)]["man"],"w"),default=str)
case("H29_subject_order_mismatch", _wrong_order, "bundle_invalid",
     "h_subject_order mismatch")
def _wrong_epochs(h):
    m=json.load(open(h[(u0,t0)]["man"])); ep=dict(m["epoch_policy"]); ep["max_epochs"]=4
    m["epoch_policy"]=ep; json.dump(m,open(h[(u0,t0)]["man"],"w"),default=str)
case("H30_wrong_epoch_policy", _wrong_epochs, "bundle_invalid", "epoch_policy mismatch")
FU=[(uid,u) for uid,br,u in SUB if u["mode"]=="fused"]
if FU:
    fu_uid=FU[0][0]
    def _false_alpha(h):
        rp=h[(fu_uid,t0)]["rp"]; j=json.load(open(rp))
        j["rec"]["fusion"]["alpha1_equals_svm_tr_enc"]=False
        json.dump(j,open(rp,"w"),default=str)
        m=json.load(open(h[(fu_uid,t0)]["man"])); m["result_sha"]=P.sha_file(rp)
        json.dump(m,open(h[(fu_uid,t0)]["man"],"w"),default=str)
    case("H31_false_alpha_identity", _false_alpha, "fusion_invalid",
         "alpha=1 AUC identity false")
    def _no_fusion(h):
        rp=h[(fu_uid,t0)]["rp"]; j=json.load(open(rp)); j["rec"]["fusion"]=None
        json.dump(j,open(rp,"w"),default=str)
        m=json.load(open(h[(fu_uid,t0)]["man"])); m["result_sha"]=P.sha_file(rp)
        json.dump(m,open(h[(fu_uid,t0)]["man"],"w"),default=str)
    case("H32_fused_missing_fusion_block", _no_fusion, "fusion_invalid",
         "fusion is not a dict")
else: ck("H31_false_alpha_identity", False, "no fused unit in the sub-wave")

# cases that only a COMPLETE 159-unit fixture can express (ledger-level)
print("\n--- ledger-level rejections (full fixture) ---")
def full_case(name,mutate,key,reason_sub):
    H2=build(); mutate(H2)
    rc,out=collect()
    hit=(rc!=0) and (f"[{key}]" in out) and (reason_sub in out)
    ck(name,hit,f"exit {rc} | key {'ok' if f'[{key}]' in out else 'MISSING'} | "
                f"reason {'ok' if reason_sub in out else 'MISSING'}")
uA,_,_ = UNITS[0]
full_case("H41_missing_unit", lambda h: shutil.rmtree(P.jobs_dir(NS)+uA),
          "missing_unit", uA)
full_case("H42_missing_fold",
          lambda h: os.remove(P.jobs_dir(NS)+uA+f"/fold_{TAGS[0]}.json"),
          "missing_fold", f"{uA}/{TAGS[0]}")
full_case("H43_duplicate_fold",
          lambda h: shutil.copy(h[(uA,TAGS[0])]["rp"],
                                P.jobs_dir(NS)+uA+f"/fold_{TAGS[0]}_copy.json"),
          "identity_mismatch", "filename tag")
def _unexpected(h):
    jd=P.jobs_dir(NS)+"main_GHOST_signed_plain_s0"; os.makedirs(jd,exist_ok=True)
    json.dump(dict(rec=dict(status="OK",unit="main_GHOST_signed_plain_s0",fold="lab0",
        namespace=NS)), open(jd+"/fold_lab0.json","w"))
full_case("H44_unexpected_cell", _unexpected, "unexpected_cell", "GHOST")
full_case("H45_poison_marker",
          lambda h: open(P.poison_path(NS),"w").write("synthetic poison"),
          "poison_marker", "synthetic poison")
def _ns(h):
    rp=h[(uA,TAGS[0])]["rp"]; j=json.load(open(rp)); j["rec"]["namespace"]="prod"
    json.dump(j,open(rp,"w"),default=str)
full_case("H46_wrong_namespace", _ns, "wrong_namespace", "'prod'")
def _failed(h):
    rp=h[(uA,TAGS[0])]["rp"]; j=json.load(open(rp)); j["rec"]["status"]="FAILED"
    json.dump(j,open(rp,"w"),default=str)
full_case("H47_failed_record", _failed, "failed_record", "status='FAILED'")
def _malformed(h):
    open(h[(uA,TAGS[0])]["rp"],"w").write("{not json")
full_case("H48_malformed_json", _malformed, "malformed_json", "fold_"+TAGS[0])
def _tally_mismatch(h):
    write_tally(uA, newly_successful=3, newly_succeeded=3)
full_case("H49_tally_disagreement", _tally_mismatch, "accounting_mismatch",
          "reused 0 + new 3")
def _skipped(h):
    write_tally(uA, newly_attempted=4, newly_successful=4, newly_succeeded=4,
                remaining=5, accounting_ok=False)
full_case("H50_skipped_without_reuse", _skipped, "accounting_mismatch", "remaining=5")

# duplicate E2E target
print("\n--- duplicate E2E target ---")
import _e2e_run as R2
T2=R2.targets(); dupd=T2+[T2[0]]
try:
    R2.assert_targets_unique(dupd); ck("H33_duplicate_e2e_target", False, "not detected")
except AssertionError as e:
    ck("H33_duplicate_e2e_target", "duplicate E2E" in str(e), str(e)[:70])

# bounded-C2 synthetic calibration stop
print("\n--- bounded C2 calibration stop ---")
import s16_c2_bounded as CB
v_out=CB.calibration_verdict(0.0231)
ck("H34_c2_calibration_stop", (v_out["passed"] is False) and
   ("UNRESOLVED" in v_out["consequence"]),
   f"+0.0231 outside {CB.RANDOM_ENCODER_EQUIVALENCE_BAND} -> {v_out['consequence'][:60]}")
ck("H35_c2_infeasible_refuses",
   CB.matched_draw(np.array([0,1,2]), np.array([],dtype=int), np.zeros(3,int),
                   np.array(["X","X","X"]), np.random.default_rng(0)) is None,
   "matched_draw returns None on an empty pool — no replacement, no pooling")

# post-C6 protocol separation and validity gate
print("\n--- post-C6 report: separation and refusal ---")
import s16_report as RP
ck("H36_protocols_separate", RP.PROTOCOLS==("lab","site","loso"), str(RP.PROTOCOLS))
ck("H37_no_0p7565_subtraction", RP.HISTORICAL_REFERENCE_ONLY==0.7565 and
   "NEVER subtracted" in open("s16_report.py").read(), "labelled reference only")
ck("H38_validity_thresholds", (RP.MOVEMENT_MIN,RP.CLIP_MAX,RP.CRAND_MIN)==(0.10,0.30,0.03),
   f"movement>{RP.MOVEMENT_MIN}, clip<{RP.CLIP_MAX}, crand>={RP.CRAND_MIN}")
ck("H39_a7_has_no_crand", "EDGEMLP" not in RP.CRAND_REF,
   "A7 has no C-RAND reference; descriptive only")
r=subprocess.run([sys.executable,"s16_report.py"],capture_output=True,text=True,
                 env=dict(ENV,S16_NS="prod"))
ck("H40_report_refuses_without_results", r.returncode!=0 and "NO HEADLINE" in r.stderr,
   f"exit {r.returncode}")

shutil.rmtree(P.root(NS),ignore_errors=True)
print(f"\n=== FINAL SUMMARY: {len(F)} failures ===")
for f in F: print("  FAIL "+f)
sys.exit(1 if F else 0)
