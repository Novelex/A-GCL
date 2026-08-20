"""S8 core: faithful mirrors of the THREE production training contracts (P/O/C).
Uses production classes/functions via import wherever possible. NO production file touched.
P = fork paper_exact | O = original 08339b7 agcl_ABIDE_queue.py | C = fork corrected defaults."""
import sys, os, json, copy, hashlib, numpy as np, torch
os.environ.setdefault("MPLBACKEND","Agg")
sys.path.insert(0,"/users/3171356m/A-GCL")
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7")
import s7_core as C7
from torch_scatter import scatter
from unsupervised.encoder.tu_encoder import TUEncoder
from unsupervised.learning.ginfominmax import GInfoMinMax
from unsupervised.view_learner import (ViewLearner, compute_reverse_index,
    symmetrize_edge_logits, sample_symmetric_logistic_noise, sample_ordered_concrete_mask)
from agcl_ABIDE_queue import (MemoryBank_Q, PaperMemoryBank_Q, calc_regloss,
                              calc_regloss_paper, setup_seed)
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset

S8="/users/3171356m/agcl_audit_s0/s8/"; BASE=20260818
M1B="/users/3171356m/agcl_audit_s0/s5/M1_B/processed/M1_B_v1.pt"
M1B_SHA="312266b23ecf1348ce083cb25d9c5e5a51d5595dab9ce5639875a51c12f1f844"

CFG={
 "P":dict(nn=False,mr=False,pbr=False,drop=0.0,batch_T=1.0,mem_T=1.0,sym=False,
          mask="ordered",reg="paper_keep",bank="early",model_cr_sign=+1.0,eval_repr="z"),
 "O":dict(nn=True, mr=True, pbr=True, drop=0.3,batch_T=0.2,mem_T=0.1,sym=True,
          mask="original",reg="orig_drop",bank="early",model_cr_sign=-1.0,eval_repr="h",
          aug="replace"),   # 08339b7: x_aug forward receives the MASK ALONE as edge_weight
 "C":dict(nn=True, mr=True, pbr=True, drop=0.3,batch_T=0.2,mem_T=0.1,sym=True,
          mask="symmetric",reg="budget",bank="late",model_cr_sign=+1.0,eval_repr="z"),
}
REG_LAMBDA=2.0; CR_LAMBDA=0.4; LR=5e-4; BATCH=32; EMB=32; MAXLEN=256; MLP_EDGE=64

class Cache(InMemoryDataset):
    def __init__(s, root, fn):
        s._fn=fn; super().__init__(root)
        s.data, s.slices = torch.load(s.processed_paths[0], weights_only=False)
    @property
    def processed_file_names(s): return s._fn
    @property
    def raw_file_names(s): return []
    def download(s): pass
    def process(s): raise RuntimeError("audit cache must never rebuild")

def load_dataset(subset=None):
    assert hashlib.sha256(open(M1B,"rb").read()).hexdigest()==M1B_SHA, "M1_B cache drift"
    ds=Cache("/users/3171356m/agcl_audit_s0/s5/M1_B","M1_B_v1.pt")
    dl=[ds[i] for i in (range(len(ds)) if subset is None else subset)]
    return dl

def build(cfg_name, seed, device="cpu"):
    c=CFG[cfg_name]; setup_seed(BASE+seed)
    enc=lambda: TUEncoder(num_dataset_features=3, emb_dim=EMB, num_gc_layers=2,
        drop_ratio=c["drop"], pooling_type="standard", is_infograph=False,
        normalize_nodes=c["nn"], message_relu=c["mr"], post_bn_relu=c["pbr"])
    model=GInfoMinMax(enc(), EMB).to(device)                       # positional, real call site
    view =ViewLearner(enc(), mlp_edge_model_dim=MLP_EDGE).to(device)
    mopt=torch.optim.Adam(model.parameters(), lr=LR)
    vopt=torch.optim.Adam(view.parameters(),  lr=LR)
    bank = MemoryBank_Q(MAXLEN,EMB,device) if c["bank"]=="late" else PaperMemoryBank_Q(MAXLEN,EMB,device)
    return model, view, mopt, vopt, bank, c

def _mask(c, batch, edge_logits, device):
    """Returns (mu_or_None, edge_mask) exactly as each production path computes them."""
    if c["mask"]=="ordered":                                   # P: fork paper_exact
        mu, m = sample_ordered_concrete_mask(edge_logits, temperature=1.0)
        return mu, m
    if c["mask"]=="original":                                  # O: 08339b7 verbatim math
        bias=0.0+0.0001
        eps=(bias-(1-bias))*torch.rand(edge_logits.size(),device=device)+(1-bias)
        gate=(torch.log(eps)-torch.log(1-eps)+edge_logits)/1.0
        return None, torch.sigmoid(gate).squeeze()
    rev=compute_reverse_index(batch.edge_index)                # C: fork corrected
    sym=symmetrize_edge_logits(batch.edge_index, edge_logits, rev)
    mu=torch.sigmoid(sym)
    noise=sample_symmetric_logistic_noise(batch.edge_index, rev, bias=1e-4, device=device)
    return mu, torch.sigmoid((noise+sym)/1.0)

def _per_graph_mean(vals, batch):
    edge_batch=batch.batch[batch.edge_index[0]]
    uni,cnt=edge_batch.unique(return_counts=True)
    sm=scatter(vals, edge_batch, reduce="sum")
    out=[]
    for b_id in range(BATCH):
        if b_id in uni: out.append(sm[b_id]/cnt[uni.tolist().index(b_id)])
    return torch.stack(out).mean()

def train_step(cfg_name, model, view, mopt, vopt, bank, batch, device, stats=None,
               skip_model=False, skip_view=False):
    c=CFG[cfg_name]
    # ---------- VIEW half-step (ascent) ----------
    view.train(); view.zero_grad(); model.eval()
    x,_=model(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
    logits=view(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
    mu,mask=_mask(c,batch,logits,device)
    aug_w = mask if c.get("aug")=="replace" else batch.edge_weight*mask   # O replaces; P/C scale
    x_aug,_=model(batch.batch,batch.x,batch.edge_index,None,aug_w)
    if c["reg"]=="orig_drop":  reg=_per_graph_mean(1.0-mask,batch); sgn=-1.0
    elif c["reg"]=="budget":   reg=_per_graph_mean(mu,batch);       sgn=+1.0
    else:                      reg=_per_graph_mean(mu,batch);       sgn=-1.0   # paper_keep
    if c["bank"]=="early":     # P and O: push BEFORE cr (paper-literal bug 5 / original order)
        bank.push(x_aug.detach())
        cr=calc_regloss_paper(x,x_aug,bank.get_memory(),temperature=c["mem_T"])
    else:                      # C: read pre-batch state, same-subject exclusion
        vm,vids=bank.get_valid_memory()
        cr = x.sum()*0.0 if vm.size(0)==0 else calc_regloss(x,x_aug,vm,vids,batch.subject_id,temperature=c["mem_T"])
    infonce_v=model.calc_loss(x,x_aug,temperature=c["batch_T"],sym=c["sym"])
    view_loss=infonce_v + sgn*(REG_LAMBDA*reg) + CR_LAMBDA*cr
    if not skip_view:
        (-view_loss).backward(); vopt.step()
    if skip_model=="omit":                       # diagnostics: no model half AT ALL
        return view_loss, None
    # ---------- MODEL half-step (descent) ----------
    model.train(); view.eval(); model.zero_grad()
    x,_=model(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
    with torch.no_grad():
        logits2=view(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
        _,mask2=_mask(c,batch,logits2,device)
    aug_w2 = mask2 if c.get("aug")=="replace" else batch.edge_weight*mask2
    x_aug,_=model(batch.batch,batch.x,batch.edge_index,None,aug_w2)
    if c["bank"]=="early":
        cr2=calc_regloss_paper(x,x_aug,bank.get_memory(),temperature=c["mem_T"])
    else:
        vm,vids=bank.get_valid_memory()
        cr2 = x.sum()*0.0 if vm.size(0)==0 else calc_regloss(x,x_aug,vm,vids,batch.subject_id,temperature=c["mem_T"])
    infonce_m=model.calc_loss(x,x_aug,temperature=c["batch_T"],sym=c["sym"])
    model_loss=infonce_m + c["model_cr_sign"]*CR_LAMBDA*cr2
    if not skip_model:
        model_loss.backward(); mopt.step()
    if c["bank"]=="late":
        bank.push(x_aug.detach(), batch.subject_id)
    if stats is not None:
        stats.append(dict(view_loss=float(view_loss), model_loss=float(model_loss),
            infonce_view=float(infonce_v), infonce_model=float(infonce_m),
            reg=float(reg), cr=float(cr), keep_mu=float(mu.mean()) if mu is not None else float('nan'),
            keep_sampled=float(mask.mean())))
    return view_loss, model_loss

def extract(model, dl, device="cpu", weighted=True):
    model.eval(); H=[];Z=[];Y=[]
    loader=DataLoader(dl,batch_size=64,shuffle=False)
    with torch.no_grad():
        for b in loader:
            b=b.to(device)
            h,z,_=model.encode(b.batch,b.x,b.edge_index,None,b.edge_weight if weighted else None)
            H.append(h.cpu());Z.append(z.cpu());Y.append(b.y.cpu())
    return torch.cat(H).numpy(), torch.cat(Z).numpy(), torch.cat(Y).numpy()

def train(cfg_name, seed, epochs, dl, device="cpu", collect_every=None, bs=None):
    torch.set_num_threads(1)          # bitwise CPU determinism requires single-thread
    model,view,mopt,vopt,bank,c=build(cfg_name,seed,device)
    g=torch.Generator(); g.manual_seed(BASE+seed)
    loader=DataLoader(dl,batch_size=bs or BATCH,shuffle=True,generator=g,drop_last=True)  # production sets drop_last=True; bs override is SMOKE-ONLY  # production agcl_ABIDE_queue.py:227 sets drop_last=True
    curves=[]
    for ep in range(1,epochs+1):
        st=[]
        for batch in loader:
            batch=batch.to(device)
            train_step(cfg_name,model,view,mopt,vopt,bank,batch,device,stats=st)
        agg={k:float(np.mean([s[k] for s in st])) for k in st[0]}
        agg["epoch"]=ep; curves.append(agg)
        if collect_every and ep%collect_every==0:
            print(f"[{cfg_name} s{seed}] ep{ep} view={agg['view_loss']:.4f} model={agg['model_loss']:.4f} keep={agg['keep_sampled']:.4f}",flush=True)
    return model,view,curves
