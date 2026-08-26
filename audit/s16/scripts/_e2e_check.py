"""E2E assertions: read every row from disk and check SUBSTANCE."""
import sys, os, json, glob, numpy as np
sys.path.insert(0,'/users/3171356m/A-GCL/audit/s16/scripts')
import s16_data as DAT, s16_feat as FT, s16_grid as G, _e2e_run as R, s16_prov as P
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
S16=DAT.S16; F=[]
def ck(n,ok,d=""):
    print(("PASS " if ok else "FAIL ")+n+(" | "+d if d else ""),flush=True)
    if not ok: F.append(f"{n}: {d}")

d,MAN,ent = DAT.load("signed", where="e2echeck")
Xfc,_,_,_ = K.load_Xfc()
tri,_ = FT.build_X("edgetri", d["FC"], d["ALFF"], np.arange(700))
# S12A5 arm C consumed Xe = X_fc.astype(np.float32) (w_wave1.py:34), so the correct
# parity target is BITWISE AT FLOAT32. Comparing an f32 array upcast to f64 against the
# f64 original can never hold; that was an error in the assertion, not in the data.
_b32 = np.array_equal(tri, Xfc.astype(np.float32))
_d64 = float(np.abs(tri.astype(np.float64)-Xfc).max())
ck("A7_parity_f32_bitwise", tri.shape==(954,4005) and _b32 and _d64 < 3e-8,
   f"A7 input at E=signed == X_fc.astype(float32) BITWISE (what S12A5 arm C consumed); "
   f"f64 residual {_d64:.2e} is pure float32 quantisation. NOTE: this checks A7's\n"
   f"   INPUT parity only. It does NOT validate a C6<->C2 bridge — that claim is\n"
   f"   WITHDRAWN because the training recipes differ (AGGREGATION_SPEC.md section 6).")

T=R.targets(); rows=[]
R.assert_targets_unique(T)
POL=R.e2e_policy()
ck("e2e_policy_identity", POL.name=="e2e" and POL.max_epochs==4
   and (POL.n_lab,POL.n_site,POL.n_loso)==(1,0,0),
   f"policy={POL.name} max_epochs={POL.max_epochs} hash={POL.policy_hash()} "
   f"folds=({POL.n_lab},{POL.n_site},{POL.n_loso})")
EXPECTED_FOLDS = ["lab0"]            # n_lab=1, n_site=0, n_loso=0
BRANCHES={"main":G.MAIN,"ctrl":G.CTRL,"abl":G.ABL}

for b,i,label in T:
    u=BRANCHES[b][i]; uid=G.unit_id(u)
    jd=P.jobs_dir("e2e")+uid
    fs=sorted(glob.glob(jd+"/fold_*.json"))
    # FULL CONTRACT (defect D41): the checker previously read fs[0] and looked at a
    # few AUC fields. A unit could carry the wrong fold, an extra fold, a broken
    # sealed bundle, a PROD policy hash or a missing prediction file and still be
    # reported as PASS. Every target is now checked against the whole contract.
    got=[os.path.basename(f)[len("fold_"):-len(".json")] for f in fs]
    ck(f"folds_{label}", got==EXPECTED_FOLDS,
       f"folds on disk {got} expected {EXPECTED_FOLDS}")
    # UNIT-COMPLETION CONTRACT (defect D48). A sealed bundle proves one FOLD is
    # correct; it does not prove the UNIT finished. Shared with the collector so the
    # two definitions cannot diverge.
    okc, whyc = P.validate_unit_completion("e2e", uid, len(EXPECTED_FOLDS))
    ck(f"completion_{label}", okc, "; ".join(whyc) if whyc else
       "POISON absent, TALLY accounting identity holds, STATUS terminal, UNIT.done present")
    if not fs: continue
    for fpath, tag in zip(fs, got):
        rec=json.load(open(fpath))["rec"]
        if tag==EXPECTED_FOLDS[0]: rows.append((label,rec))
        ok=rec.get("status")=="OK"
        ck(f"status_{label}", ok, "OK" if ok else str(rec.get("error",""))[:130])
        if not ok: continue
        # --- identity and namespace
        ck(f"namespace_{label}", rec.get("namespace")=="e2e",
           f"record namespace {rec.get('namespace')!r} (must be an explicit 'e2e')")
        ck(f"policy_{label}", rec.get("policy_hash")==POL.policy_hash()
           and rec.get("policy_name")=="e2e",
           f"policy_name={rec.get('policy_name')!r} hash={rec.get('policy_hash')!r} "
           f"expected e2e/{POL.policy_hash()}")
        ck(f"identity_{label}", all(rec.get(k)==v for k,v in
              (("unit",uid),("fold",tag),("arm",u["arm"]),("arch",u["arch"]),
               ("E",u["E"]),("mode",u["mode"]),("control",u.get("control")),
               ("alff_mode",u.get("alff_mode")),("seed",G.SEEDS[u["seed_idx"]]))),
           f"unit/arm/arch/E/mode/control/alff_mode/seed all match the grid entry")
        # --- the 4-epoch budget was ACTUALLY honoured
        be, ts = rec.get("best_epoch"), rec.get("total_steps")
        ck(f"epochs_{label}", isinstance(be,int) and 1<=be<=POL.max_epochs,
           f"best_epoch {be} within 1..{POL.max_epochs}; total_steps {ts}")
        # --- representation width is what the architecture must produce
        exp_rd=P.expected_repr_dim(u["arch"],u["kh"],128,"roi")
        ck(f"repr_dim_{label}", rec.get("repr_dim_used")==exp_rd,
           f"repr_dim_used {rec.get('repr_dim_used')} expected {exp_rd}")
        # --- the SEALED 5-FILE BUNDLE must validate, not merely exist
        cfg=P.model_cfg(u)
        MAN_E, ent_E = (MAN, ent) if u["E"]=="signed" else DAT.load(u["E"],
                                                        where="e2echeck")[1:]
        exp=P.contract_fields("e2e", u, cfg, uid, tag, G.SEEDS[u["seed_idx"]],
                              tag.rstrip("0123456789"), MAN_E, ent_E, exp_rd,
                              POL, POL.train_consts())
        fp=P.feat_dir("e2e")+f"{uid}__{tag}.npz"
        okb,why=P.validate_bundle("e2e", uid, tag, exp, fp,
            P.ckpt_dir("e2e")+f"{uid}__{tag}.pt", fp+".prov.json", fpath,
            P.feat_dir("e2e")+f"{uid}__{tag}.pred.json")
        ck(f"bundle_{label}", okb, why)
        # --- predictions exist and are subject-resolved
        pf=P.feat_dir("e2e")+f"{uid}__{tag}.pred.json"
        try:
            pr=json.load(open(pf))
            n_ok=(len(pr.get("subject_ids",[]))==len(pr.get("label_used",[]))
                  >0 and len(pr["subject_ids"])==len(set(pr["subject_ids"])))
            ck(f"pred_{label}", n_ok and pr.get("namespace")=="e2e",
               f"{len(pr.get('subject_ids',[]))} unique test subjects, ns="
               f"{pr.get('namespace')!r}")
        except Exception as e: ck(f"pred_{label}", False, f"unreadable: {e!r}")
        # --- metrics
        for pt in ("probe_honest","probe_old_full","head","head_ema"):
            a=rec.get(pt,{}).get("auc")
            # AUC EXACTLY 0.5 IS VALID (defect D9). C-PERM on permuted labels can
            # legitimately produce it; a correct control was previously reported
            # as a gate failure. Require only: present, finite, within [0,1].
            ck(f"{pt}_{label}", a is not None and np.isfinite(a) and 0.0<=a<=1.0,
               f"AUC {a}")
        # FROZEN EVALUATED STATE. raw = validation-best checkpoint; EMA reported
        # ALONGSIDE. Selection is by VALIDATION only and is fixed in the protocol:
        # raw-versus-EMA must never be chosen after seeing test results.
        es=str(rec.get("evaluated_state",""))
        ck(f"evaluated_state_{label}",
           es.startswith("raw=validation-best checkpoint") and "EMA(0.999)" in es
           and "selection by VALIDATION only" in es,
           es[:110] if es else "evaluated_state ABSENT")
        sv=rec.get("svm_tr_enc")
        ck(f"svm_tr_enc_{label}", sv is not None and np.isfinite(sv) and 0.0<sv<1.0, f"{sv}")
        if rec.get("mode")=="fused":
            fu=rec.get("fusion")
            ck(f"fusion_{label}", isinstance(fu,dict) and len(fu.get("alpha_curve",[]))==21,
               f"{len(fu.get('alpha_curve',[])) if fu else 0} alpha points")
            if fu:
                ck(f"alpha1_exact_{label}", fu["alpha1_equals_svm_tr_enc"] is True,
                   f"|{fu['alpha1_auc']:.10f} - {sv:.10f}| = {abs(fu['alpha1_auc']-sv):.2e} (<1e-12)")
                ck(f"alpha1_bitwise_{label}", fu["alpha1_bitwise_equals_zsFC"] is True, "z(s_FC) on te")
                ck(f"unclamped_{label}", fu.get("delta_is_unclamped") is True,
                   "the selected-alpha delta is reported unclamped, negative or not")
                ck(f"fused_{label}", np.isfinite(fu["fused_auc"]) and 0<fu["fused_auc"]<1,
                   f"AUC {fu['fused_auc']:.4f} alpha={fu['alpha_selected']} "
                   f"delta_vs_svm_tr_enc={fu['delta_vs_svm_tr_enc']:+.4f}")

ck("all_targets_present", len(rows)==len(T),
   f"{len(rows)} of {len(T)} targets produced the expected fold")
if rows:
    lbl,rec=next(((l,r) for l,r in rows if r.get("mode")=="fused"), rows[0])
    print(f"\n=== ONE FULL RESULTS ROW, VERBATIM ({lbl}) ===")
    print(json.dumps(rec,indent=1,default=str)[:6500])
print(f"\n=== E2E SUMMARY: {len(T)} arms, {len(F)} assertion failures ===")
for f in F: print("  FAIL "+f)
sys.exit(1 if F else 0)
