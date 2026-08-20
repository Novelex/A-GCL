"""S7.5 sections 1-2: ONE (path,seed) stage-wise probe unit. Resumable, own DONE."""
import sys, os, argparse, numpy as np
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s75"); import s75_core as S
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C7
ap=argparse.ArgumentParser(); ap.add_argument("--task",type=int); ap.add_argument("--ntask",type=int,default=20)
A=ap.parse_args()
OUT=S.S75+"out/stages"; os.makedirs(OUT,exist_ok=True)
W=[(p,s) for p in ["P","O"] for s in range(10)]
D=C7.load_all(); y=D["y"]; FOLDS,_=C7.splits(); idx=np.arange(954)
STAGES=[("wgin1","WGIN1_preBN"),("bn1","BN1"),("postbn1","postBN1_toWGIN2"),
        ("wgin2","WGIN2_preBN"),("bn2","BN2_final_nodes_preNorm"),
        ("final_nodes_postnorm","final_nodes_postNorm"),("h","h"),("z","z")]
for (path,sd) in W[A.task::A.ntask]:
    name=f"stage_{path}_s{sd:02d}"
    if C7.is_done(OUT,name): print("skip",name,flush=True); continue
    st=S.stages(path,sd,"B",idx); rec={}; oo={}
    for key,lab in STAGES:
        m,o=S.probe(st[key],y,FOLDS,boot=1000); rec[lab]=m; oo[lab]=o
        print(f"  {path} s{sd} {lab} AUC={m['auc']:.4f}",flush=True)
    rec["_paired"]=dict(nodes_minus_h=S.paired_delta(oo["final_nodes_postNorm"],oo["h"],y,2000),
                        h_minus_z=S.paired_delta(oo["h"],oo["z"],y,2000))
    C7.write_unit(OUT,name,payload_json=dict(path=path,seed=sd,stages=rec,
                  **C7.provenance({"unit":"S75_stage"})))
print("STAGE TASK COMPLETE",flush=True)
