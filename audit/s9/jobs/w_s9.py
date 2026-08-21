"""S9 - ONE decisive diagnostic. Corrected-C only, seed 20260818, 200 epochs.
Captures pre_norm_nodes / post_norm_nodes / h / z at epoch 0 (pre-update) and epoch 200."""
import sys, os, json, time, argparse, hashlib, numpy as np, torch
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s8"); import s8_core as S
import s7_core as C7
ap=argparse.ArgumentParser()
ap.add_argument("--smoke",action="store_true"); ap.add_argument("--device",default="cpu")
ap.add_argument("--epochs",type=int,default=200)
A=ap.parse_args()
S9="/users/3171356m/agcl_audit_s0/s9/"; OUT=S9+("smoke_out" if A.smoke else "out")
os.makedirs(OUT,exist_ok=True)
dev=torch.device(A.device if (A.device=="cpu" or torch.cuda.is_available()) else "cpu")
if A.smoke: torch.set_num_threads(2)
# pre-registered 20 subjects for deterministic mu: first 10 of ASD block + first 10 of NC block
MU_SUBJECTS=list(range(0,10))+list(range(455,465))

dl=S.load_dataset(range(8) if A.smoke else None)
N=len(dl)

def extract_all(model, view):
    """pre_norm (bn2 hook, eval->dropout inert), post_norm (encoder output), h, z."""
    model.eval(); view.eval()
    pre=[]
    hk=model.encoder.bns[1].register_forward_hook(lambda m,i,o: pre.append(o.detach().float().cpu()))
    H=[];Z=[];PN=[];Y=[];SID=[]
    from torch_geometric.loader import DataLoader
    with torch.no_grad():
        for b in DataLoader(dl,batch_size=64,shuffle=False):
            b=b.to(dev)
            assert b.edge_weight is not None and torch.isfinite(b.edge_weight).all()
            h,z,node=model.encode(b.batch,b.x,b.edge_index,None,b.edge_weight)
            H.append(h.float().cpu()); Z.append(z.float().cpu()); PN.append(node.float().cpu())
            Y.append(b.y.cpu()); SID.append(b.subject_id.cpu())
    hk.remove()
    out=dict(pre_norm_nodes=torch.cat(pre).numpy().reshape(N,90,32),
             post_norm_nodes=torch.cat(PN).numpy().reshape(N,90,32),
             h=torch.cat(H).numpy(), z=torch.cat(Z).numpy(),
             labels=torch.cat(Y).numpy().astype(np.int64),
             subject_ids=torch.cat(SID).numpy().astype(np.int64))
    exp={"pre_norm_nodes":(N,90,32),"post_norm_nodes":(N,90,32),"h":(N,32),"z":(N,32),
         "labels":(N,),"subject_ids":(N,)}
    for k,v in out.items():
        assert v.shape==exp[k], f"{k} shape {v.shape} != {exp[k]}"
        assert np.isfinite(v).all(), f"{k} non-finite"
    return out

def det_mu(view, model):
    """Deterministic mu (no Concrete noise) for the 20 pre-registered subjects, C contract."""
    view.eval(); res={}
    subs=[i for i in MU_SUBJECTS if i<N]
    from torch_geometric.loader import DataLoader
    with torch.no_grad():
        for i in subs:
            b=next(iter(DataLoader([dl[i]],batch_size=1)));  b=b.to(dev)
            lg=view(b.batch,b.x,b.edge_index,None,b.edge_weight)
            rev=S.compute_reverse_index(b.edge_index)
            mu=torch.sigmoid(S.symmetrize_edge_logits(b.edge_index,lg,rev))
            res[f"mu_{i:03d}"]=mu.float().cpu().numpy().reshape(90,90)
    return res

t0=time.time()
model,view,mopt,vopt,bank,cfg=S.build("C",0,device=str(dev))     # seed = BASE+0 = 20260818
# ---- EPOCH-0 extraction BEFORE any optimizer update ----
n_updates=0
e0=extract_all(model,view)
assert n_updates==0
C7.write_unit(OUT,"embeddings_epoch000",payload_npz=e0)
torch.save({"model":model.state_dict(),"view":view.state_dict()},OUT+"/ckpt_epoch000.pt")
print("epoch-0 extraction + checkpoint saved (pre-update)",flush=True)

# ---- training with per-epoch mask/loss logging ----
from torch_geometric.loader import DataLoader
g=torch.Generator(); g.manual_seed(S.BASE)
loader=DataLoader(dl,batch_size=(8 if A.smoke else 32),shuffle=True,generator=g,drop_last=True)
curves=[]
for ep in range(1,A.epochs+1):
    st=[]; grad_ok=True; mus=[]
    for batch in loader:
        batch=batch.to(dev)
        S.train_step("C",model,view,mopt,vopt,bank,batch,str(dev),stats=st)
        n_updates+=1
        for p in list(model.parameters())+list(view.parameters()):
            if p.grad is not None and not torch.isfinite(p.grad).all(): grad_ok=False
        mus.append(st[-1]["keep_mu"])
    agg={k:float(np.mean([s[k] for s in st])) for k in st[0]}
    agg.update(epoch=ep, grad_finite=grad_ok,
               mu_mean=float(np.mean(mus)), mu_std=float(np.std(mus)),
               mu_min=float(min(s["keep_mu"] for s in st)), mu_max=float(max(s["keep_mu"] for s in st)),
               expected_keep=float(np.mean(mus)), sampled_keep=agg["keep_sampled"])
    curves.append(agg)
    if ep%10==0 or ep==1: print(f"ep{ep} view={agg['view_loss']:.4f} model={agg['model_loss']:.4f} "
        f"nce_v={agg['infonce_view']:.4f} cr={agg['cr']:.4f} mu={agg['mu_mean']:.4f} "
        f"keep={agg['sampled_keep']:.4f} grads_finite={grad_ok}",flush=True)

# ---- FINAL extraction, checkpoints, deterministic mu ----
eF=extract_all(model,view)
C7.write_unit(OUT,f"embeddings_epoch{A.epochs:03d}",payload_npz=eF)
torch.save({"model":model.state_dict(),"view":view.state_dict()},OUT+f"/ckpt_epoch{A.epochs:03d}.pt")
C7.write_unit(OUT,"mu_deterministic_final",payload_npz=det_mu(view,model))
prov=C7.provenance({"unit":"S9_train","device":str(dev),"epochs":A.epochs,"seed":S.BASE,
    "cmd":" ".join(sys.argv),"runtime_s":round(time.time()-t0,1),
    "dataset_sha":S.M1B_SHA,"n_optimizer_update_pairs":n_updates,
    "mu_subjects_preregistered":MU_SUBJECTS,
    "cuda":torch.version.cuda,"gpu":(torch.cuda.get_device_name(0) if dev.type=="cuda" else None),
    "out_sha":{f:hashlib.sha256(open(os.path.join(OUT,f),'rb').read()).hexdigest()[:16]
               for f in sorted(os.listdir(OUT)) if f.endswith(('.npz','.pt'))}})
C7.write_unit(OUT,"S9_TRAIN_META",payload_json=dict(provenance=prov,curves=curves))
# ---- checkpoint reload verification ----
m2,v2,_,_,_,_=S.build("C",0,device=str(dev))
ck=torch.load(OUT+f"/ckpt_epoch{A.epochs:03d}.pt",weights_only=False)
m2.load_state_dict(ck["model"]); v2.load_state_dict(ck["view"])
# (1) checkpoint INTEGRITY: reloaded weights must be BITWISE identical
for (ka,va),(kb,vb) in zip(model.state_dict().items(),m2.state_dict().items()):
    assert ka==kb and torch.equal(va.cpu(),vb.cpu()), f"state_dict mismatch {ka}"
# (2) functional check: forward equivalence. Tolerance is device-dependent — CUDA
# scatter_add is atomically non-deterministic (S6/S8 measured run-to-run ~2e-3),
# CPU is bitwise. atol chosen accordingly and RECORDED.
tol = 5e-3 if dev.type=="cuda" else 1e-6
e2=extract_all(m2,v2)
for k in ("pre_norm_nodes","h","z"):
    d=float(np.abs(eF[k]-e2[k]).max())
    assert d<tol, f"reload forward mismatch {k}: {d} >= {tol}"
    print(f"  reload check {k}: max_abs={d:.2e} (tol {tol})",flush=True)
print("checkpoint reload verified (bitwise state_dict + forward within device tolerance)",flush=True)
open(OUT+"/S9_TRAIN_DONE","w").write(C7.git_head()+"\n")
print(f"S9 TRAIN COMPLETE {round(time.time()-t0,1)}s",flush=True)
