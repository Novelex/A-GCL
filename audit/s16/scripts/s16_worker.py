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
    u = {"main":G.MAIN,"ctrl":G.CTRL,"abl":G.ABL}[branch][idx]
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
            tr_enc,tr_prb = FT.honest_split(tr,y_use)     # C2c: encoder sees tr_enc ONLY
            model,ema_sd,curve,info = TR.train_fold(arch,X,FCu,y_use,tr_enc,cfg,seed,
                                                    log=f"{uid}/{tag}",sparse=sparse)
            R,S = TR.extract(model,X,FCu,np.arange(954),arch=="WGIN",sparse=sparse)
            Rp = FT.fuse(R,Xfc) if u["mode"]=="fused" else R
            dh,ph = FT.probe_honest(Rp,y_use,tr_prb,te)   # honest: both sides OOS
            do,po = K.probe_pipe(np.asarray(Rp,dtype=np.float64),y_use,
                                 [(np.asarray(tr),np.asarray(te))],[])  # old, for delta
            ck=f"{S16}ckpt/{uid}__{tag}.pt"
            torch.save(model.state_dict(),ck+".tmp"); os.replace(ck+".tmp",ck)
            rec={**u, **dict(status="OK",unit=uid,branch=branch,fold=tag,
                 fold_protocol=tag.rstrip("0123456789"),seed=seed,
                 head=metrics(y_use[te],S[te]),
                 probe_honest=metrics(y_use[te],ph[te]),
                 probe_old=metrics(y_use[te],po[te]),
                 probe_bias=float(do["auc"]-dh["auc"]),
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
