"""S16 C6 worker. One unit = (config, mode, seed) x 9 folds.
J1 no dynamics assert kills a job. J3 per-fold try/except. J4 resume-safe."""
import sys, os, json, time, socket, hashlib, traceback, threading, signal
import numpy as np, torch
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_data as DAT, s16_models as MO, s16_train as TR, s16_feat as FT, s16_grid as G
import s16_prov as P
import s16_policy as PL
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, brier_score_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as _ras

S16 = DAT.S16; _STOP={"f":False}
def train_consts(policy):
    """Derived from the POLICY OBJECT. Never from module globals — that is exactly
    how the 4-vs-400 epoch lie arose."""
    return dict(max_epochs=policy.max_epochs, min_epochs=policy.min_epochs,
                patience=policy.patience, min_delta=policy.min_delta,
                warmup_frac=policy.warmup_frac, cosine_floor=policy.cosine_floor,
                label_smooth=policy.label_smooth, batch=policy.batch,
                ema_decay=policy.ema_decay, policy_name=policy.name,
                policy_hash=policy.policy_hash())
signal.signal(signal.SIGUSR1, lambda s,f: _STOP.update(f=True))

def _boot(y,s,B=2000,seed=DAT.BASE):
    from scipy.stats import rankdata
    rng=np.random.default_rng(seed); n=len(y); idx=rng.integers(0,n,(B,n))
    Y=y[idx].astype(np.float64); S=s[idx]; npos=Y.sum(1); nneg=n-npos
    ok=(npos>0)&(nneg>0); r=rankdata(S,method="average",axis=1)
    return (((r*Y).sum(1)-npos*(npos+1)/2)/np.maximum(npos*nneg,1))[ok]

def metrics(y,score,boot=2000):
    y=np.asarray(y); score=np.asarray(score,dtype=np.float64)
    if len(np.unique(y))<2: return dict(auc=float("nan"),n=int(len(y)))
    yh=(score>0).astype(int); p=1/(1+np.exp(-np.clip(score,-30,30)))
    tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel(); bs=_boot(y,score,boot)
    lg=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    cal=LogisticRegression(C=1e10,max_iter=5000).fit(lg[:,None],y)
    return dict(auc=float(roc_auc_score(y,score)),
        auc_ci_lo=float(np.percentile(bs,2.5)),auc_ci_hi=float(np.percentile(bs,97.5)),
        auprc=float(average_precision_score(y,score)),acc=float(accuracy_score(y,yh)),
        bal_acc=float(balanced_accuracy_score(y,yh)),
        sens=float(tp/max(tp+fn,1)),spec=float(tn/max(tn+fp,1)),
        ppv=float(tp/max(tp+fp,1)),npv=float(tn/max(tn+fn,1)),
        f1=float(f1_score(y,yh)),mcc=float(matthews_corrcoef(y,yh)),
        tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn),brier=float(brier_score_loss(y,p)),
        calib_slope=float(cal.coef_[0,0]),calib_intercept=float(cal.intercept_[0]),n=int(len(y)))

def status(jd,st,done,tot,extra=None):
    r=dict(state=st,folds_done=done,folds_total=tot,host=socket.gethostname(),
           updated=time.strftime("%F %T"))
    if extra: r.update(extra)
    json.dump(r,open(f"{jd}/STATUS.json.tmp","w"),indent=1)
    os.replace(f"{jd}/STATUS.json.tmp",f"{jd}/STATUS.json")

def heartbeat(jd,ev):
    while not ev.is_set():
        open(f"{jd}/HEARTBEAT","w").write(time.strftime("%F %T")); ev.wait(60)

def aj(o,p):
    json.dump(o,open(p+".tmp","w"),indent=1,default=str); json.load(open(p+".tmp"))
    os.replace(p+".tmp",p)

def run(branch, idx, ns=None):
    """ns: 'prod' or 'e2e'. Namespaces are FULLY disjoint — a poison marker or
    artifact in one can never affect or satisfy the other."""
    ns = ns or os.environ.get("S16_NS","prod")
    policy = PL.get(ns)                      # namespace SELECTS the immutable policy
    assert policy.namespace == ns, f"policy/namespace mismatch {policy.namespace}!={ns}"
    TRAIN_CONSTS = train_consts(policy)
    P.ensure(ns)
    u = {"main":G.MAIN,"ctrl":G.CTRL,"abl":G.ABL,
         "bnt":G.BNTU,"wgin":G.WGINU,"ctrlu":G.CTRLU}[branch][idx]
    uid = G.unit_id(u); jd=P.jobs_dir(ns)+uid; os.makedirs(jd,exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","4")))
    torch.use_deterministic_algorithms(True)
    t_unit=time.time()
    d,MAN,ent = DAT.load(u["E"], where=uid)              # GATE-C every job
    FC,ALFF,y_true = d["FC"],d["ALFF"],d["y"].astype(np.int64)
    sparse = bool(ent["sparse"])
    Xfc,_y,_i,_m = K.load_Xfc()
    folds = (DAT.folds(d,"lab")[:policy.n_lab] + DAT.folds(d,"site")[:policy.n_site]
             + DAT.folds(d,"loso")[:policy.n_loso])       # POLICY decides the folds
    ev=threading.Event(); threading.Thread(target=heartbeat,args=(jd,ev),daemon=True).start()
    status(jd,"running",0,len(folds),dict(unit=uid,config=u))
    arch=u["arch"]; spec=FT.ARMS[u["arm"]][1]; ctrl=u.get("control")
    y_use = np.random.default_rng(DAT.BASE).permutation(y_true) if ctrl=="C-PERM" else y_true
    seed=G.SEEDS[u["seed_idx"]]
    cfg=dict(K_or_hidden=u["kh"],lr=3e-4,wd=1e-3,loss="L-BCE",
             freeze_encoder=(ctrl=="C-RAND"),readout="roi",dropout=0.10,H=128)
    done=0; n_fail=0; n_attempt=0; n_reused=0; n_new=0
    POISON_FRAC=0.05                 # >5% of folds attempted -> the wave is broken
    for tag,tr,te in folds:
        fp=P.feat_dir(ns)+f"{uid}__{tag}.npz"
        ckp=P.ckpt_dir(ns)+f"{uid}__{tag}.pt"
        mfp=fp+".prov.json"
        # PROVENANCE-SAFE RESUME. Filename existence is NEVER sufficient.
        exp=dict(schema="s16-prov-1", namespace=ns, git_sha=P.git_sha(),
                 worker_version=P.WORKER_VERSION,
                 config_hash=P.cfg_hash(u,cfg,TRAIN_CONSTS),
                 h_fc=ent["h_fc"], h_alff=MAN["h_alff"],
                 h_folds_lab=MAN["h_folds_lab"], h_folds_site=MAN["h_folds_site"],
                 h_folds_loso=MAN["h_folds_loso"], cache_file=ent["cache_file"],
                 unit=uid, arm=u["arm"], arch=u["arch"], E=u["E"], mode=u["mode"],
                 control=u.get("control"), alff_mode=u.get("alff_mode"),
                 seed=int(seed), fold=tag, protocol=tag.rstrip("0123456789"),
                 epoch_policy=dict(max_epochs=TRAIN_CONSTS["max_epochs"],
                     min_epochs=TRAIN_CONSTS["min_epochs"],
                     patience=TRAIN_CONSTS["patience"],
                     min_delta=TRAIN_CONSTS["min_delta"]),
                 optimizer_recipe=dict(opt="AdamW", lr=cfg["lr"], wd=cfg["wd"],
                     betas=[0.9,0.999], eps=1e-8,
                     warmup_frac=TRAIN_CONSTS["warmup_frac"],
                     cosine_floor=TRAIN_CONSTS["cosine_floor"],
                     clip="adaptive p90 of last 200, no clip for first 50 steps",
                     label_smooth=TRAIN_CONSTS["label_smooth"], loss=cfg["loss"]),
                 model_state_rule=("raw = validation-best checkpoint; EMA(0.999) "
                     "evaluated alongside and reported with the delta; selection by "
                     "VALIDATION only (S15 PROTOCOL.md:186)"),
                 repr_dim=None)
        # B1: expected repr dim computed INDEPENDENTLY from architecture + config.
        # It must NEVER be read back out of the manifest being validated.
        exp["repr_dim"] = P.expected_repr_dim(u["arch"], u["kh"], cfg.get("H",128),
                                              cfg.get("readout","roi"))
        exp["policy_hash"] = policy.policy_hash()
        exp["h_labels"] = MAN["h_labels"]; exp["h_subject_order"] = MAN["h_subject_order"]
        okr, why = P.validate_bundle(ns, uid, tag, exp, fp, ckp, mfp,
                                     jd+f"/fold_{tag}.json",
                                     P.feat_dir(ns)+f"{uid}__{tag}.pred.json")
        if okr:
            n_reused+=1; done+=1
            print(f"REUSE {uid} {tag} (provenance validated)",flush=True)
            status(jd,"running",done,len(folds),dict(reused=n_reused)); continue
        elif os.path.exists(fp):
            print(f"RECOMPUTE {uid} {tag}: {why}",flush=True)
        t0=time.time()
        try:
            # SPLIT FIRST, then preprocess. Cohort statistics must never see
            # tr_prb or te (defect D5).
            tr_enc,tr_prb = FT.honest_split(tr,y_use)     # encoder sees tr_enc ONLY
            X,FCu = FT.build_X(spec,FC,ALFF,tr_enc,control=ctrl,
                               alff_mode=u["alff_mode"])
            Xin = X            # for A7 build_X already returns the E-transformed triangle
            D_in = Xin.shape[-1]
            model,ema_sd,curve,info = TR.train_fold(arch,Xin,FCu,y_use,tr_enc,cfg,seed,
                                                    log=f"{uid}/{tag}",sparse=sparse,
                                                    policy=policy)
            R,S = TR.extract(model,Xin,FCu,np.arange(954),arch=="WGIN",sparse=sparse)
            # FROZEN RULE (S15 PROTOCOL.md:186,200): "EMA of weights decay 0.999, EMA
            # and raw both evaluated, selection by VALIDATION only, both reported with
            # the delta." EMA was being computed every step and DISCARDED (defect D10).
            # The raw state loaded above is already the VALIDATION-BEST checkpoint;
            # selection is unchanged. EMA is evaluated ALONGSIDE it, never instead.
            ema_model = MO.build_model(arch, Xin.shape[-1], seed, cfg["K_or_hidden"],
                                       freeze_encoder=cfg["freeze_encoder"],
                                       readout=cfg["readout"], p=cfg["dropout"],
                                       H=cfg["H"])
            ema_model.load_state_dict(ema_sd)
            _,S_ema = TR.extract(ema_model,Xin,FCu,np.arange(954),arch=="WGIN",
                                 sparse=sparse)
            # ---- learned score, honest probe (both sides out-of-sample) ----
            dh,ph_oof = FT.probe_honest(R,y_use,tr_prb,te)
            do,po = K.probe_pipe(np.asarray(R,dtype=np.float64),y_use,
                                 [(np.asarray(tr),np.asarray(te))],[])   # old_full
            # ---- FOLD-SPECIFIC FC COMPARATOR (NOT a floor): SVM trained on tr_enc,
            #      the SAME data the encoder
            #      saw. 0.7565 is a full-tr HISTORICAL reference, kept separate.
            # ONE expensive FC probe per fold: fitted on tr_enc, scored on tr_prb+te.
            # svm_tr_enc is derived from it, so fused arms pay no second FC fit.
            s_fc,s_le = FT.scores_for_fusion(R,Xfc,y_use,tr_enc,tr_prb,te)
            svm_tr_enc = float(_ras(y_use[te], s_fc[te]))
            # FOLD-SPECIFIC baselines on IDENTICAL test subjects. 0.7319 is NOT a
            # universal constant — it was one fold's reading.
            _d_full,_o_full = K.probe_pipe(Xfc.astype(np.float64),y_use,
                                           [(np.asarray(tr),np.asarray(te))],[])
            svm_tr_full = float(_d_full["auc"])
            size_delta_paired = float(svm_tr_full - svm_tr_enc)
            fusion=None
            if u["mode"]=="fused":
                curve_a=[dict(alpha=float(a),
                    auc=float(_ras(y_use[te],FT.fuse_scores(s_fc,s_le,a,tr_prb)[te])))
                    for a in FT.ALPHA_GRID]
                inner=[dict(alpha=float(a),
                    auc=float(_ras(y_use[tr_prb],FT.fuse_scores(s_fc,s_le,a,tr_prb)[tr_prb])))
                    for a in FT.ALPHA_GRID]
                # CONSERVATIVE TIE-BREAKING: highest inner AUC, and among ties the
                # LARGEST alpha, i.e. the most FC-favouring choice (defect D7).
                a_sel=max(inner,key=lambda r:(r["auc"],r["alpha"]))["alpha"]
                f_sel=FT.fuse_scores(s_fc,s_le,a_sel,tr_prb)
                # alpha=1.0 is the FC FALLBACK ENDPOINT — NOT a guaranteed floor.
                # What is guaranteed: the endpoint exists; it equals standardised FC;
                # it preserves the FC ranking; and its AUC equals this fold's
                # svm_tr_enc exactly. NOTHING guarantees the SELECTED alpha beats it
                # on the outer test set: delta_vs_svm_tr_enc MAY BE NEGATIVE and is
                # never clamped, floored or replaced after test evaluation.
                f1=FT.fuse_scores(s_fc,s_le,1.0,tr_prb)
                mu,sd=FT.zfit(s_fc,tr_prb)
                a1_bitwise=bool(np.array_equal(f1[te],FT.zapply(s_fc,mu,sd)[te]))
                a1_auc=float(_ras(y_use[te],f1[te]))
                a1_exact=bool(abs(a1_auc-svm_tr_enc)<1e-12)
                st_sc,st_coef=FT.stack_scores(s_fc,s_le,y_use,tr_prb,te)
                fused_auc=float(_ras(y_use[te],f_sel[te]))
                fusion=dict(alpha_curve=curve_a,alpha_curve_inner=inner,
                    alpha_selected=a_sel,fused_auc=fused_auc,
                    stack_auc=float(_ras(y_use[te],st_sc)),stack_coef=st_coef,
                    alpha1_bitwise_equals_zsFC=a1_bitwise,alpha1_auc=a1_auc,
                    alpha1_equals_svm_tr_enc=a1_exact,
                    delta_vs_svm_tr_enc=float(fused_auc-svm_tr_enc),
                    delta_vs_svm_tr_full=float(fused_auc-svm_tr_full),
                    delta_is_unclamped=True,
                    endpoint_semantics=("alpha=1 is the FC FALLBACK ENDPOINT, not a "
                        "guaranteed floor; the outer-test delta may be negative and "
                        "is reported as measured"))
            ph = ph_oof
            ck=ckp
            torch.save(model.state_dict(),ck+".tmp"); os.replace(ck+".tmp",ck)
            rec={**u, **dict(status="OK",unit=uid,branch=branch,fold=tag,
                 fold_protocol=tag.rstrip("0123456789"),seed=seed,
                 head=metrics(y_use[te],S[te]),
                 head_ema=metrics(y_use[te],S_ema[te]),
                 ema_delta=(float(_ras(y_use[te],S_ema[te])-_ras(y_use[te],S[te]))
                            if len(np.unique(y_use[te]))>1 else float("nan")),
                 evaluated_state="raw=validation-best checkpoint; EMA(0.999) reported "
                                 "alongside; selection by VALIDATION only "
                                 "(S15 PROTOCOL.md:186)",
                 probe_honest=metrics(y_use[te],ph[te]),
                 probe_old_full=metrics(y_use[te],po[te]),
                 probe_bias_uncorrected=float(do["auc"]-dh["auc"]),
                 svm_tr_enc=svm_tr_enc, svm_tr_full=svm_tr_full,
                 size_delta_paired=size_delta_paired, fusion=fusion,
                 n_tr=int(len(tr)), n_tr_enc=int(len(tr_enc)), n_tr_probe=int(len(tr_prb)),
                 repr_dim_used=int(R.shape[1]), sparse=sparse,
                 **{k:v for k,v in info.items() if k!="movement"},
                 movement=info["movement"],
                 label_convention="ASD=1 NC=0 (A-GCL uses ASD=0/HC=1: AUC same, SENS/SPEC SWAPPED)",
                 h_fc=ent["h_fc"],h_labels=MAN["h_labels"],h_folds_lab=MAN["h_folds_lab"],
                 cache_file=ent["cache_file"],node=socket.gethostname(),
                 ckpt_sha=hashlib.sha256(open(ck,"rb").read()).hexdigest()[:16],
                 wall_s=round(time.time()-t0,1),
                 peak_rss_mb=round(__import__("resource").getrusage(
                     __import__("resource").RUSAGE_SELF).ru_maxrss/1024.0,1))}
            aj(dict(rec=rec,curve=curve), f"{jd}/fold_{tag}.json")     # JSON FIRST
            tmp=fp+".tmp.npz"
            np.savez_compressed(tmp[:-4],repr=R.astype(np.float32),head=S.astype(np.float32),
                probe_honest=ph,probe_old=po,head_ema=S_ema.astype(np.float32),
                y_true=y_true,y_used=y_use,
                tr=np.asarray(tr),tr_enc=tr_enc,tr_prb=tr_prb,te=np.asarray(te))
            zz=np.load(tmp); assert np.isfinite(zz["repr"]).all(); os.replace(tmp,fp)
            # ---- D. subject-level predictions for this cell ----
            teA=np.asarray(te)
            pred=dict(schema="s16-pred-1", unit=uid, namespace=ns, branch=branch,
                arm=u["arm"], E=u["E"], arch=u["arch"], mode=u["mode"],
                control=u.get("control"), alff_mode=u.get("alff_mode"),
                seed=int(seed), fold=tag, protocol=tag.rstrip("0123456789"),
                subject_ids=[d["ids"][i] for i in teA],
                subject_index=teA.tolist(),
                label_true=y_true[teA].tolist(), label_used=y_use[teA].tolist(),
                label_convention="ASD=1 NC=0",
                score_fc=s_fc[teA].tolist(), score_learned=s_le[teA].tolist(),
                score_head=S[teA].tolist(), score_head_ema=S_ema[teA].tolist(),
                score_fused=(f_sel[teA].tolist() if fusion else None),
                alpha_selected=(fusion["alpha_selected"] if fusion else None),
                alpha_curve_inner=(fusion["alpha_curve_inner"] if fusion else None),
                svm_tr_enc=svm_tr_enc, svm_tr_full=svm_tr_full,
                git_sha=P.git_sha(), config_hash=P.cfg_hash(u,cfg,TRAIN_CONSTS),
                h_fc=ent["h_fc"], h_alff=MAN["h_alff"], cache_file=ent["cache_file"],
                ckpt_sha=rec["ckpt_sha"], feat_sha=P.sha_file(fp))
            P.atomic_json(pred, P.feat_dir(ns)+f"{uid}__{tag}.pred.json")
            # manifest LAST: it records hashes of the result and prediction files,
            # so the whole bundle is sealed together (B2).
            P.atomic_json(P.build_manifest(ns,u,cfg,uid,tag,seed,
                tag.rstrip("0123456789"),MAN,ent,int(R.shape[1]),fp,ck,"OK",
                TRAIN_CONSTS, policy=policy,
                result_path=f"{jd}/fold_{tag}.json",
                pred_path=P.feat_dir(ns)+f"{uid}__{tag}.pred.json",
                effective_cfg=P.effective_config(u,cfg)), mfp)
            n_new+=1
            print(f"DONE {uid} {tag} honest {rec['probe_honest']['auc']:.4f} "
                  f"old {rec['probe_old_full']['auc']:.4f} head {rec['head']['auc']:.4f} "
                  f"mv {info['movement_max']:.3f} clip {info['clip_rate']:.2f} "
                  f"{info['verdict']} {rec['wall_s']}s",flush=True)
        except Exception as e:
            n_fail+=1
            rec={**u, **dict(status="FAILED",unit=uid,branch=branch,fold=tag,
                 error=repr(e),traceback=traceback.format_exc(),
                 node=socket.gethostname(),wall_s=round(time.time()-t0,1))}
            aj(dict(rec=rec,curve=[]), f"{jd}/fold_{tag}.json")
            print(f"FAILED {uid} {tag}: {e}",flush=True)
        done+=1; n_attempt+=1
        status(jd,"running",done,len(folds),dict(failed=n_fail,attempted=n_attempt))
        # MASS FAILURE ABORTS THE WHOLE ARRAY. The per-fold try/except exists to
        # survive ONE anomalous fold, not to let a systematic bug report COMPLETED.
        if n_attempt>=4 and n_fail > POISON_FRAC*n_attempt:
            aid=os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
            msg=(f"POISON {uid}: {n_fail}/{n_attempt} folds failed "
                 f"(>{POISON_FRAC:.0%}). Cancelling array {aid}.")
            open(P.poison_path(ns),"a").write(msg+"\n")
            open(f"{jd}/POISON","w").write(msg+"\n")
            status(jd,"poisoned",done,len(folds),dict(failed=n_fail,attempted=n_attempt))
            print("ERROR "+msg, file=sys.stderr, flush=True); print("ERROR "+msg, flush=True)
            ev.set()
            if aid: os.system(f"scancel {aid}")
            sys.exit(2)
        if _STOP["f"]: status(jd,"requeued",done,len(folds)); ev.set(); sys.exit(0)
    ev.set()
    n_ok=n_attempt-n_fail
    status(jd,"done",done,len(folds),dict(wall_s=round(time.time()-t_unit,1),
           attempted=n_attempt,succeeded=n_ok,failed=n_fail))
    json.dump(dict(unit=uid, namespace=ns,
                   expected=len(folds),                 # 1 expected
                   newly_attempted=n_attempt,           # 2 newly attempted
                   newly_successful=n_new,              # 3 newly successful
                   failed=n_fail,                       # 4 failed
                   validated_reused=n_reused,           # 5 validated reused
                   remaining=len(folds)-(n_reused+n_new),  # 6 remaining
                   # legacy aliases kept so nothing downstream silently reads None
                   attempted=n_attempt, succeeded=n_ok, newly_succeeded=n_new,
                   expected_folds=len(folds),
                   accounting_ok=bool(n_reused+n_new==len(folds)-n_fail)),
              open(f"{jd}/TALLY.json","w"),indent=1)
    open(f"{jd}/UNIT.done","w").write("done")
    line=(f"UNIT_COMPLETE {uid} [{ns}]: {n_attempt} attempted, {n_ok} succeeded, "
          f"{n_fail} failed | validated_reused {n_reused} + newly_succeeded {n_new} "
          f"= {n_reused+n_new} of expected {len(folds)}")
    print(line+f" ({time.time()-t_unit:.0f}s)",flush=True)
    if n_fail>0: print("ERROR "+line, file=sys.stderr, flush=True)

if __name__=="__main__":
    run(sys.argv[1], int(sys.argv[2]),
        ns=(sys.argv[3] if len(sys.argv)>3 else None))
