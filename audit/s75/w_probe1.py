"""S7.5 ONE probe = one (path,seed,stage). Fully resumable at probe granularity."""
import sys, os, argparse, numpy as np
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s75"); import s75_core as S
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C7
ap=argparse.ArgumentParser(); ap.add_argument("--task",type=int); ap.add_argument("--ntask",type=int)
A=ap.parse_args()
OUT=S.S75+"out/probes"; os.makedirs(OUT,exist_ok=True)
STAGES=[("wgin1","WGIN1_preBN"),("bn1","BN1"),("postbn1","postBN1_toWGIN2"),
        ("wgin2","WGIN2_preBN"),("bn2","BN2_final_nodes_preNorm"),
        ("final_nodes_postnorm","final_nodes_postNorm"),("h","h"),("z","z")]
W=[(p,s,k,l) for p in ["P","O"] for s in range(10) for k,l in STAGES]
D=C7.load_all(); y=D["y"]; FOLDS,_=C7.splits(); idx=np.arange(954)
mine=W[A.task::A.ntask]
cache={}
for (path,sd,key,lab) in mine:
    name=f"pr_{path}_s{sd:02d}_{lab}"
    if C7.is_done(OUT,name): print("skip",name,flush=True); continue
    if (path,sd) not in cache:
        cache.clear(); cache[(path,sd)]=S.stages(path,sd,"B",idx)
    st=cache[(path,sd)]
    m,o=S.probe(st[key],y,FOLDS,boot=1000)
    C7.write_unit(OUT,name,payload_npz=dict(oof=o.astype(np.float64)),
                  payload_json=dict(path=path,seed=sd,stage=lab,**m))
    print("done",name,round(m["auc"],4),flush=True)
print("PROBE TASK COMPLETE",flush=True)
