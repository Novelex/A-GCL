"""S16 C6 worker. One unit = (config, mode, seed) x 9 folds.
J1 no dynamics assert kills a job. J3 per-fold try/except. J4 resume-safe."""
import sys, os, json, time, socket, hashlib, traceback, threading, signal
import numpy as np, torch
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_data as DAT, s16_models as MO, s16_train as TR, s16_feat as FT, s16_grid as G
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, brier_score_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as _ras

S16 = DAT.S16; _STOP={"f":False}
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

def run(branch, idx):
    u = {"main":G.MAIN,"ctrl":G.CTRL,"abl":G.ABL,
         "bnt":G.BNTU,"wgin":G.WGINU,"ctrlu":G.CTRLU}[branch][idx]
    uid = G.unit_id(u); jd=f"{S16}jobs/{uid}"; os.makedirs(jd,exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","4")))
    torch.use_deterministic_algorithms(True)
    t_unit=time.time()
    d,MAN,ent = DAT.load(u["E"], where=uid)              # GATE-C every job
    FC,ALFF,y_true = d["FC"],d["ALFF"],d["y"].astype(np.int64)
    sparse = bool(ent["sparse"])
    Xfc,_y,_i,_m = K.load_Xfc()
    folds = DAT.folds(d,"lab")[:3]+DAT.folds(d,"site")[:3]+DAT.folds(d,"loso")[:3]
    ev=threading.Event(); threading.Thread(target=heartbeat,args=(jd,ev),daemon=True).start()
    status(jd,"running",0,len(folds),dict(unit=uid,config=u))
    arch=u["arch"]; spec=FT.ARMS[u["arm"]][1]; ctrl=u.get("control")
    y_use = np.random.default_rng(DAT.BASE).permutation(y_true) if ctrl=="C-PERM" else y_true
    seed=G.SEEDS[u["seed_idx"]]
    cfg=dict(K_or_hidden=u["kh"],lr=3e-4,wd=1e-3,loss="L-BCE",
             freeze_encoder=(ctrl=="C-RAND"),readout="roi",dropout=0.10,H=128)
    done=0
    for tag,tr,te in folds:
        fp=f"{S16}feat/{uid}__{tag}.npz"
        if os.path.exists(fp): done+=1; status(jd,"running",done,len(folds)); continue
        t0=time.time()
        try:
            X,FCu = FT.build_X(spec,FC,ALFF,tr,control=ctrl,alff_mode=u["alff_mode"])
            tr_enc,tr_prb = FT.honest_split(tr,y_use)     # encoder sees tr_enc ONLY
            Xin = X            # for A7 build_X already returns the E-transformed triangle
            D_in = Xin.shape[-1]
            model,ema_sd,curve,info = TR.train_fold(arch,Xin,FCu,y_use,tr_enc,cfg,seed,
                                                    log=f"{uid}/{tag}",sparse=sparse)
            R,S = TR.extract(model,Xin,FCu,np.arange(954),arch=="WGIN",sparse=sparse)
            # ---- learned score, honest probe (both sides out-of-sample) ----
            dh,ph_oof = FT.probe_honest(R,y_use,tr_prb,te)
            do,po = K.probe_pipe(np.asarray(R,dtype=np.float64),y_use,
                                 [(np.asarray(tr),np.asarray(te))],[])   # old_full
            # ---- C6 FLOOR ANCHOR: SVM trained on tr_enc, the SAME data the encoder
            #      saw. 0.7565 is a full-tr HISTORICAL reference, kept separate.
            # ONE expensive FC probe per fold: fitted on tr_enc, scored on tr_prb+te.
            # svm_tr_enc is derived from it, so fused arms pay no second FC fit.
            s_fc,s_le = FT.scores_for_fusion(R,Xfc,y_use,tr_enc,tr_prb,te)
            svm_tr_enc = float(_ras(y_use[te], s_fc[te]))
            fusion=None
            if u["mode"]=="fused":
                curve_a=[dict(alpha=float(a),
                    auc=float(_ras(y_use[te],FT.fuse_scores(s_fc,s_le,a,tr_prb)[te])))
                    for a in FT.ALPHA_GRID]
                inner=[dict(alpha=float(a),
                    auc=float(_ras(y_use[tr_prb],FT.fuse_scores(s_fc,s_le,a,tr_prb)[tr_prb])))
                    for a in FT.ALPHA_GRID]
                a_sel=max(inner,key=lambda r:r["auc"])["alpha"]     # selected on INNER only
                f_sel=FT.fuse_scores(s_fc,s_le,a_sel,tr_prb)
                # alpha=1.0 IDENTITY: fused must be z(s_FC) bitwise on the scored
                # indices, and its AUC must equal svm_tr_enc EXACTLY.
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
                    delta_vs_0p7565_SECONDARY=float(fused_auc-0.7565))
            ph = ph_oof
            ck=f"{S16}ckpt/{uid}__{tag}.pt"
            torch.save(model.state_dict(),ck+".tmp"); os.replace(ck+".tmp",ck)
            rec={**u, **dict(status="OK",unit=uid,branch=branch,fold=tag,
                 fold_protocol=tag.rstrip("0123456789"),seed=seed,
                 head=metrics(y_use[te],S[te]),
                 probe_honest=metrics(y_use[te],ph[te]),
                 probe_old_full=metrics(y_use[te],po[te]),
                 probe_bias_uncorrected=float(do["auc"]-dh["auc"]),
                 svm_tr_enc=svm_tr_enc, fusion=fusion,
                 n_tr=int(len(tr)), n_tr_enc=int(len(tr_enc)), n_tr_probe=int(len(tr_prb)),
                 repr_dim_used=int(Rp.shape[1]), sparse=sparse,
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
                probe_honest=ph,probe_old=po,y_true=y_true,y_used=y_use,
                tr=np.asarray(tr),tr_enc=tr_enc,tr_prb=tr_prb,te=np.asarray(te))
            zz=np.load(tmp); assert np.isfinite(zz["repr"]).all(); os.replace(tmp,fp)
            print(f"DONE {uid} {tag} honest {rec['probe_honest']['auc']:.4f} "
                  f"old {rec['probe_old']['auc']:.4f} head {rec['head']['auc']:.4f} "
                  f"mv {info['movement_max']:.3f} clip {info['clip_rate']:.2f} "
                  f"{info['verdict']} {rec['wall_s']}s",flush=True)
        except Exception as e:
            rec={**u, **dict(status="FAILED",unit=uid,branch=branch,fold=tag,
                 error=repr(e),traceback=traceback.format_exc(),
                 node=socket.gethostname(),wall_s=round(time.time()-t0,1))}
            aj(dict(rec=rec,curve=[]), f"{jd}/fold_{tag}.json")
            print(f"FAILED {uid} {tag}: {e}",flush=True)
        done+=1; status(jd,"running",done,len(folds))
        if _STOP["f"]: status(jd,"requeued",done,len(folds)); ev.set(); sys.exit(0)
    ev.set(); status(jd,"done",done,len(folds),dict(wall_s=round(time.time()-t_unit,1)))
    open(f"{jd}/UNIT.done","w").write("done")
    print(f"UNIT_COMPLETE {uid} {time.time()-t_unit:.0f}s",flush=True)

if __name__=="__main__": run(sys.argv[1], int(sys.argv[2]))
