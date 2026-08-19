"""J3a - RANDOM (untrained) encoder embedding extraction. NO OPTIMIZER, NO TRAINING."""
import sys, os, json, argparse, numpy as np, torch, time
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C
ap=argparse.ArgumentParser()
ap.add_argument("--task",type=int,default=0); ap.add_argument("--ntask",type=int,default=1)
ap.add_argument("--device",default="cpu"); ap.add_argument("--smoke",action="store_true")
ap.add_argument("--bs",type=int,default=64)
A=ap.parse_args()
TAG=os.environ.get("S7_SMOKE_TAG","smoke")
OUT=C.S7+(TAG+"/J3a" if A.smoke else "J3a"); os.makedirs(OUT,exist_ok=True)
dev=torch.device(A.device if (A.device=="cpu" or torch.cuda.is_available()) else "cpu")
D=C.load_all(); y=D["y"]; ids=D["ids"]
idx=np.array([0,1,2,477,478,479,900,901]) if A.smoke else np.arange(954)

def worklist(smoke):
    W=[]
    S1,S2=(2,3) if smoke else (30,50)
    for s in range(S1):                       # seeds 0-29 -> ALL 9 configs (complete matrix first)
        for p in ["P","O","C"]:
            for b in C.BRANCHES: W.append((s,p,b))
    for s in range(S1,S2):                    # seeds 30-49 -> primary B only
        for p in ["P","O","C"]: W.append((s,p,"B"))
    return W
W=worklist(A.smoke); mine=W[A.task::A.ntask]
print(f"total units {len(W)}, this task {len(mine)}, device {dev}",flush=True)
for (seed,path,br) in mine:
    name=f"emb_s{seed:03d}_{path}_{br}"
    if C.is_done(OUT,name): print("skip",name,flush=True); continue
    t0=time.time(); m=C.build_model(path,seed,device=dev); m.eval()
    H=[];Z=[]
    with torch.no_grad():
        for s0 in range(0,len(idx),A.bs):
            ch=idx[s0:s0+A.bs]
            x,ei,ew,bt=C.batch_graphs(ch,br,device=dev)
            h,z,_=m.encode(bt,x,ei,None,ew)
            H.append(h.float().cpu()); Z.append(z.float().cpu())
    h=torch.cat(H).numpy().astype(np.float32); z=torch.cat(Z).numpy().astype(np.float32)
    assert h.shape[0]==len(idx) and z.shape[0]==len(idx)
    C.write_unit(OUT,name,payload_npz=dict(h=h,z=z,y=y[idx].astype(np.int64),
                 subject_index=idx.astype(np.int64)),
                 payload_json=dict(**C.provenance({"unit":"J3a","seed":seed,"path":path,"branch":br,
                    "device":str(dev),"secs":round(time.time()-t0,2),"h_dim":h.shape[1],"z_dim":z.shape[1]})))
    print("done",name,f"{time.time()-t0:.1f}s",flush=True)
print("J3a TASK COMPLETE",flush=True)
