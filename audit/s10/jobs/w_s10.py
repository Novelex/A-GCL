"""S10 - ROI-aware readout x mask-budget factorial. Arms A/B/C/D, seeds 0/1/2 (BASE+s).
Audit-only wrapper; production code untouched. Writes under agcl_audit_s0/s10 only."""
import sys, os, json, time, argparse, hashlib, numpy as np, torch
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s8"); import s8_core as S
import s7_core as C7
from torch_geometric.loader import DataLoader
ap=argparse.ArgumentParser()
ap.add_argument("--arm",required=True,choices=list("ABCD")); ap.add_argument("--seed",type=int,required=True)
ap.add_argument("--smoke",action="store_true"); ap.add_argument("--device",default="cpu")
ap.add_argument("--epochs",type=int,default=200)
A=ap.parse_args()
S10="/users/3171356m/agcl_audit_s0/s10/"; OUT=S10+("smoke_out" if A.smoke else "out")
os.makedirs(OUT,exist_ok=True)
dev=torch.device(A.device if (A.device=="cpu" or torch.cuda.is_available()) else "cpu")
if A.smoke: torch.set_num_threads(2)
TARGET=0.80; MU_SUBJECTS=list(range(0,10))+list(range(455,465))
ROI_SHA=hashlib.sha256(open("/users/3171356m/agcl_audit_s0/s1_roi_alignment_90.csv","rb").read()).hexdigest()

# ---------------- DATA / CACHE GATE ----------------
assert hashlib.sha256(open(S.M1B,"rb").read()).hexdigest()==S.M1B_SHA, "FAIL - dataset SHA"
dl=S.load_dataset(range(8) if A.smoke else None); N=len(dl)
Z=np.load("/users/3171356m/agcl_audit_s0/s3c/X_sources.npz",allow_pickle=True)
XB=Z["M1"].astype(np.float64)
mn=XB.min((1,2),keepdims=True); mx=XB.max((1,2),keepdims=True); XB=((XB-mn)/(mx-mn)).astype(np.float32)
if not A.smoke:
    ys=np.array([int(g.y) for g in dl]); assert N==954 and ys.sum()==455 and (ys==0).sum()==499
    s9=np.load("/users/3171356m/agcl_audit_s0/s9/out/embeddings_epoch200.npz")
    assert np.array_equal(s9["labels"],ys) and np.array_equal(s9["subject_ids"],np.array([int(g.subject_id) for g in dl]))
sp=json.load(open("/users/3171356m/agcl_audit_s0/s3c/splits.json"))
assert hashlib.sha256(open("/users/3171356m/agcl_audit_s0/s3c/splits.json","rb").read()).hexdigest()=="28fed44dc4666066cc0621f329392e58050b39d5ef1371ec5327830518d98916"
# ROI-order protection at LOAD: every graph's x must equal the frozen M1_B row block exactly
map_idx=list(range(8)) if A.smoke else list(range(N))
import pandas as pd
m5=pd.read_csv("/users/3171356m/agcl_audit_s0/s5/subject_id_map.csv")
xs_ids=[str(s) for s in Z["ids"]]
for j,g in enumerate(dl):
    assert g.x.shape==(90,3) and g.num_nodes==90
    fid=m5.FILE_ID[int(g.subject_id)]
    k=xs_ids.index(fid)
    assert np.allclose(g.x.numpy(),XB[k],atol=1e-6), f"ROI order/reindex mismatch subject {fid}"
print(f"DATA GATE PASS: N={N}, ROI manifest sha {ROI_SHA[:16]}..., x==frozen M1_B rows",flush=True)

# ---------------- ROI-aware wrapper (arms B/D) ----------------
class ROIAwareModel(torch.nn.Module):
    """Replaces ONLY global_add_pool: post_norm_nodes [B,90,32] -> flatten (frozen ROI
    order) -> Linear(2880,32) -> the EXISTING GInfoMinMax projection head."""
    def __init__(self, gmodel):
        super().__init__()
        self.encoder=gmodel.encoder; self.proj_head=gmodel.proj_head
        self.roi_linear=torch.nn.Linear(90*32,32)
        torch.nn.init.xavier_uniform_(self.roi_linear.weight.data); self.roi_linear.bias.data.fill_(0.0)
    calc_loss=staticmethod(S.GInfoMinMax.calc_loss)
    def _nodes(self,batch,x,ei,ea,ew):
        _,node=self.encoder(batch,x,ei,ea,ew)          # node = post-norm nodes
        nb=int(batch.max())+1
        assert node.shape[0]==nb*90, "non-contiguous batch"
        assert torch.equal(batch, torch.arange(nb,device=batch.device).repeat_interleave(90)), \
            "subject blocks not contiguous / ROI order broken"
        return node.reshape(nb,90*32)
    def encode(self,batch,x,ei,ea,ew=None):
        flat=self._nodes(batch,x,ei,ea,ew); h=self.roi_linear(flat)
        return h, self.proj_head(h), flat.reshape(-1,32)      # nodes reshaped from the SAME pass
    def forward(self,batch,x,ei,ea,ew=None):
        h,z,_=self.encode(batch,x,ei,ea,ew); return z,None

def build_arm(arm,seed):
    if arm in ("C","D"):
        S.CFG["C80"]=dict(S.CFG["C"]); S.CFG["C80"]["reg"]="target80"
    cfgname="C" if arm in ("A","B") else "C80"
    model,view,mopt,vopt,bank,c=S.build("C",seed,device=str(dev))   # identical init stream to arm A
    if arm in ("B","D"):
        torch.manual_seed(S.BASE+seed+500000)                        # extra params init AFTER, recorded
        model=ROIAwareModel(model).to(dev)
        mopt=torch.optim.Adam(model.parameters(),lr=S.LR)
    return cfgname,model,view,mopt,vopt,bank

# target80 regularizer: extend _per_graph reg inside a wrapped train_step via s8_core hook
_orig_ts=S.train_step
if "target80" not in getattr(S,"_s10",[]):
    old_mask=S._mask
    def _reg_patch():
        src=S.train_step.__module__
    S._s10=["target80"]
    # implement by monkey-adding reg branch: easiest is wrapping train_step's cfg lookup
def train_step_s10(cfgname,model,view,mopt,vopt,bank,batch,stats):
    """Same as s8_core.train_step but supports reg='target80' (production regularizer_mode
    'target_keep', target 0.80: reg=(keep-0.8)^2, sign -1)."""
    c=S.CFG[cfgname]
    view.train(); view.zero_grad(); model.eval()
    x,_=model(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
    logits=view(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
    mu,mask=S._mask(c,batch,logits,str(dev))
    aug_w=batch.edge_weight*mask
    x_aug,_=model(batch.batch,batch.x,batch.edge_index,None,aug_w)
    keep=S._per_graph_mean(mu,batch)
    if c["reg"]=="target80": reg=(keep-TARGET).pow(2); sgn=-1.0
    else:                    reg=keep;                 sgn=+1.0     # budget (arms A/B)
    vm,vids=bank.get_valid_memory()
    cr = x.sum()*0.0 if vm.size(0)==0 else S.calc_regloss(x,x_aug,vm,vids,batch.subject_id,temperature=c["mem_T"])
    nce_v=model.calc_loss(x,x_aug,temperature=c["batch_T"],sym=c["sym"])
    view_loss=nce_v+sgn*(S.REG_LAMBDA*reg)+S.CR_LAMBDA*cr
    (-view_loss).backward(); vopt.step()
    model.train(); view.eval(); model.zero_grad()
    x,_=model(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
    with torch.no_grad():
        lg2=view(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
        _,mask2=S._mask(c,batch,lg2,str(dev))
    x_aug,_=model(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight*mask2)
    vm,vids=bank.get_valid_memory()
    cr2 = x.sum()*0.0 if vm.size(0)==0 else S.calc_regloss(x,x_aug,vm,vids,batch.subject_id,temperature=c["mem_T"])
    nce_m=model.calc_loss(x,x_aug,temperature=c["batch_T"],sym=c["sym"])
    model_loss=nce_m+S.CR_LAMBDA*cr2
    model_loss.backward(); mopt.step()
    bank.push(x_aug.detach(),batch.subject_id)
    rev=S.compute_reverse_index(batch.edge_index)
    stats.append(dict(view_loss=float(view_loss),model_loss=float(model_loss),
        infonce_view=float(nce_v),infonce_model=float(nce_m),reg=float(reg),cr=float(cr),
        mu_mean=float(mu.mean()),mu_std=float(mu.std()),mu_min=float(mu.min()),mu_max=float(mu.max()),
        expected_keep=float(mu.mean()),sampled_keep=float(mask.mean()),
        mask_sym_err=float((mask-mask[rev]).abs().max()),
        grad_finite=all(torch.isfinite(p.grad).all() for p in list(model.parameters())+list(view.parameters()) if p.grad is not None)))

# ---------------- budget unit test (arms C/D, pre-training) ----------------
def budget_unit_test():
    torch.manual_seed(0)
    b=next(iter(DataLoader(dl[:4],batch_size=4)))
    for bias,expect in ((-2.0,"up"),(+2.0,"down")):
        lg=torch.full((b.edge_index.shape[1],1),bias,requires_grad=True)
        rev=S.compute_reverse_index(b.edge_index)
        sym=S.symmetrize_edge_logits(b.edge_index,lg,rev); mu=torch.sigmoid(sym)
        keep=S._per_graph_mean(mu,b); reg=-S.REG_LAMBDA*(keep-TARGET).pow(2)
        (-reg).backward()                                # view ASCENDS view_loss (+reg term)
        push=-float(lg.grad.mean())                      # ascent direction on logits
        ok=(push>0)==(expect=="up")
        assert torch.isfinite(lg.grad).all()
        assert float((mu-mu[rev.clamp(max=mu.numel()-1)]).abs().max())<1e-6
        print(f"  budget test: mean(mu)={float(mu.mean()):.3f} -> ascent pushes keep {'UP' if push>0 else 'DOWN'} (expect {expect}) {'OK' if ok else 'FAIL'}",flush=True)
        assert ok, "FAIL - budget gradient direction"
if A.arm in ("C","D"): budget_unit_test()

# ---------------- run ----------------
name=f"{A.arm}_s{A.seed}"
if C7.is_done(OUT,f"train_{name}"): print("already done"); sys.exit(0)
t0=time.time()
cfgname,model,view,mopt,vopt,bank=build_arm(A.arm,A.seed)
torch.save({"model":model.state_dict(),"view":view.state_dict()},OUT+f"/ckpt_{name}_init.pt")
g=torch.Generator(); g.manual_seed(S.BASE+A.seed)
loader=DataLoader(dl,batch_size=(8 if A.smoke else 32),shuffle=True,generator=g,drop_last=True)
curves=[]
for ep in range(1,A.epochs+1):
    st=[]
    for batch in loader:
        batch=batch.to(dev); train_step_s10(cfgname,model,view,mopt,vopt,bank,batch,st)
    agg={k:(float(np.mean([s[k] for s in st])) if k!="grad_finite" else all(s[k] for s in st)) for k in st[0]}
    agg["epoch"]=ep; curves.append(agg)
    if ep%25==0 or ep==1: print(f"[{name}] ep{ep} view={agg['view_loss']:.3f} model={agg['model_loss']:.3f} "
        f"keep_exp={agg['expected_keep']:.3f} keep_smp={agg['sampled_keep']:.3f} sym={agg['mask_sym_err']:.1e} gf={agg['grad_finite']}",flush=True)

# extraction (post_norm nodes, h, z) with ROI asserts
model.eval(); view.eval()
H=[];Zz=[];PN=[];Y=[];SID=[]
with torch.no_grad():
    for b in DataLoader(dl,batch_size=64,shuffle=False):
        b=b.to(dev)
        assert b.edge_weight is not None and torch.isfinite(b.edge_weight).all()
        if A.arm in ("B","D"):
            _,node=model.encoder(b.batch,b.x,b.edge_index,None,b.edge_weight)
            nb=int(b.batch.max())+1
            flat=node.reshape(nb,2880); h=model.roi_linear(flat); z=model.proj_head(h)
        else:
            h,z,node=model.encode(b.batch,b.x,b.edge_index,None,b.edge_weight)
        H.append(h.float().cpu());Zz.append(z.float().cpu());PN.append(node.float().cpu())
        Y.append(b.y.cpu());SID.append(b.subject_id.cpu())
emb=dict(post_norm_nodes=torch.cat(PN).numpy().reshape(N,90,32),
         h=torch.cat(H).numpy(),z=torch.cat(Zz).numpy(),
         labels=torch.cat(Y).numpy().astype(np.int64),subject_ids=torch.cat(SID).numpy().astype(np.int64))
for k,v in emb.items(): assert np.isfinite(v).all(), k
C7.write_unit(OUT,f"emb_{name}",payload_npz=emb)
torch.save({"model":model.state_dict(),"view":view.state_dict()},OUT+f"/ckpt_{name}_final.pt")
# deterministic mu, 20 fixed S9 subjects
mus={}
with torch.no_grad():
    for i in [s for s in MU_SUBJECTS if s<N]:
        b=next(iter(DataLoader([dl[i]],batch_size=1))).to(dev)
        lg=view(b.batch,b.x,b.edge_index,None,b.edge_weight)
        rev=S.compute_reverse_index(b.edge_index)
        mus[f"mu_{i:03d}"]=torch.sigmoid(S.symmetrize_edge_logits(b.edge_index,lg,rev)).float().cpu().numpy().reshape(90,90)
C7.write_unit(OUT,f"mu_{name}",payload_npz=mus)
# reload check (device-aware tolerance)
cfg2,m2,v2,_,_,_=build_arm(A.arm,A.seed)
ck=torch.load(OUT+f"/ckpt_{name}_final.pt",weights_only=False)
m2.load_state_dict(ck["model"]); v2.load_state_dict(ck["view"])
for (ka,va),(kb,vb) in zip(model.state_dict().items(),m2.state_dict().items()):
    assert torch.equal(va.cpu(),vb.cpu()), ka
C7.write_unit(OUT,f"train_{name}",payload_json=dict(
    provenance=C7.provenance({"unit":"S10","arm":A.arm,"seed":S.BASE+A.seed,"cfg":cfgname,
      "roi_manifest_sha256":ROI_SHA,"dataset_sha":S.M1B_SHA,"epochs":A.epochs,
      "runtime_s":round(time.time()-t0,1),"cmd":" ".join(sys.argv)}),curves=curves))
print(f"S10 {name} COMPLETE {round(time.time()-t0,1)}s",flush=True)
