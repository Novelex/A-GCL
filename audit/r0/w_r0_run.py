"""R0 Phase 5: ONE full run. seed 123, 100 epochs (README), pre-eval + eval every 5.
Training = frozen upstream_batch_step (bitwise-parity-verified). Eval = repaired released
kf evaluator on UNWEIGHTED h. Global NumPy RNG stream, never reseeded after start."""
import sys, os, json, time, copy, hashlib, csv, numpy as np, torch, random
torch.set_num_threads(int(os.environ.get("R0_TORCH_THREADS","1")))
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/r0"); import r0_core as R, upstream_step as U
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s8"); import s8_core as S
import s7_core as C7
from torch_geometric.loader import DataLoader
OUT=R.R0+"out"; CK=R.R0+"R0_CHECKPOINTS"; os.makedirs(OUT,exist_ok=True); os.makedirs(CK,exist_ok=True)
SEED=123; EPOCHS=100; EI=5
log=open(R.R0+"R0_FULL_LOG.txt","a")
def P(*a):
    m=" ".join(str(x) for x in a); print(m,flush=True); log.write(m+"\n"); log.flush()
# seed ONCE (upstream setup_seed semantics); never reseeded again
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True; np.random.seed(SEED); random.seed(SEED)
dl=S.load_dataset(); y=np.array([int(g.y) for g in dl])
model,view,_,_,_,_=S.build("O",0)                 # arch identical to 08339b7 defaults
# re-seed once to SEED after build's internal seeding, mirroring upstream order:
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
mopt=torch.optim.Adam(model.parameters(),lr=5e-4); vopt=torch.optim.Adam(view.parameters(),lr=5e-4)
bank=U.MemoryBank_Q_upstream(256,32,"cpu")
ARGS=dict(batch_size=32,reg_lambda=2.0,cr_lambda=0.4)
loader=DataLoader(dl,batch_size=32,shuffle=True,drop_last=True)   # global-RNG shuffle, as upstream
ev_rows=[]; curves={k:[] for k in ("acc","f1","sen","spe")}
def do_eval(tag,epoch,flag):
    model.eval(); view.eval()
    h,_,_=S.extract(model,dl,weighted=False)      # released_bug_unweighted_eval=True
    r=R.upstream_kf_eval(h,y,folds=10,flag=flag)
    te=r["test"]
    ev_rows.append(dict(event=len(ev_rows),source_epoch=epoch,tag=tag,
        acc=te[0],acc_sd=te[1],f1=te[2],f1_sd=te[3],sen=te[4],sen_sd=te[5],spe=te[6],spe_sd=te[7],
        val_acc=r["val"][0],train_acc=r["train"][0],
        auc_diag=r["auc_diagnostic_not_upstream_parity"],
        rng_hash=r["rng_hash_after"],fold_hashes=";".join(r["fold_hashes"])))
    P(f"[eval {len(ev_rows)-1}] ep{epoch} test acc={te[0]:.4f}+-{te[1]:.4f} f1={te[2]:.4f} "
      f"sen={te[4]:.4f} spe={te[6]:.4f} | AUCdiag={r['auc_diagnostic_not_upstream_parity']:.4f} rng={r['rng_hash_after']}")
    torch.save({"model":model.state_dict(),"view":view.state_dict()},CK+f"/ckpt_ep{epoch:03d}.pt")
    return te
cur=do_eval("pretraining",0,flag=False)           # initial eval, flag=False (repaired path)
t0=time.time()
for epoch in range(1,EPOCHS+1):
    for batch in loader:
        U.upstream_batch_step(ARGS,model,view,mopt,vopt,bank,batch,"cpu",{})
    if epoch%EI==0:
        cur=do_eval("train",epoch,flag=True)
    for k,i in (("acc",0),("f1",2),("sen",4),("spe",6)):   # append EVERY epoch (stale between evals)
        curves[k].append(dict(curve_epoch=epoch,source_epoch=ev_rows[-1]["source_epoch"],value=cur[i]))
    if epoch%10==0: P(f"ep{epoch} done ({time.time()-t0:.0f}s)")
with open(R.R0+"R0_EVAL_EVENTS_21.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(ev_rows[0])); w.writeheader(); w.writerows(ev_rows)
with open(R.R0+"R0_UPSTREAM_CURVE_100.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["curve_epoch","source_epoch","acc","f1","sen","spe"])
    for i in range(EPOCHS):
        w.writerow([curves["acc"][i]["curve_epoch"],curves["acc"][i]["source_epoch"],
                    curves["acc"][i]["value"],curves["f1"][i]["value"],
                    curves["sen"][i]["value"],curves["spe"][i]["value"]])
json.dump(dict(config=R.CONFIG_NAME,seed=SEED,epochs=EPOCHS,eval_interval=EI,
    released_bug_unweighted_eval=R.released_bug_unweighted_eval,
    n_eval_events=len(ev_rows),runtime_s=round(time.time()-t0,1),
    provenance=C7.provenance({"unit":"R0_full_run"})),open(R.R0+"R0_RUN_CONFIG.json","w"),indent=1,default=str)
P(f"R0 FULL RUN COMPLETE: {len(ev_rows)} eval events, {time.time()-t0:.0f}s training+eval")
