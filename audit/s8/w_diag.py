"""S8 Step 2: view-learner diagnostics A-F. CPU. NO full training (max 50 steps, one batch, diagnostic)."""
import sys, os, json, copy, numpy as np, torch
torch.set_num_threads(int(os.environ.get("S8_THREADS","4")))
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s8"); import s8_core as S
import s7_core as C7
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr, spearmanr
OUT=S.S8+"out"; os.makedirs(OUT,exist_ok=True)
dl=S.load_dataset(); N=len(dl)
rep={"provenance":C7.provenance({"unit":"S8_diag"})}

def all_mu(cfg,seed):
    """Collect per-directed-edge mu (P/C) or logits (raw) across ALL 954 graphs."""
    torch.manual_seed(S.BASE+seed)
    model,view,_,_,_,c=S.build(cfg,seed); view.eval()
    mus=[]; logits_all=[]
    with torch.no_grad():
        for b in DataLoader(dl,batch_size=64,shuffle=False):
            lg=view(b.batch,b.x,b.edge_index,None,b.edge_weight)
            if c["mask"]=="symmetric":
                rev=S.compute_reverse_index(b.edge_index)
                mu=torch.sigmoid(S.symmetrize_edge_logits(b.edge_index,lg,rev))
            else:
                mu=torch.sigmoid(lg.squeeze(-1))
            mus.append(mu.cpu()); logits_all.append(lg.squeeze(-1).cpu())
    return torch.cat(mus).numpy().reshape(N,8100), torch.cat(logits_all).numpy().reshape(N,8100)

# ---------- A. initial mask behaviour (3 seeds) ----------
A={}
for cfg in ["P","C"]:
    per_seed=[]
    for sd in range(3):
        mu,_=all_mu(cfg,sd)
        h,edges=np.histogram(mu,bins=10,range=(0,1))
        per_seed.append(dict(seed=sd,mean=float(mu.mean()),std=float(mu.std()),
            min=float(mu.min()),max=float(mu.max()),hist=h.tolist()))
    M=np.stack([all_mu(cfg,sd)[0] for sd in range(3)])          # [3,954,8100]
    ev=M.var(axis=0)                                             # per-edge variance across seeds
    A[cfg]=dict(per_seed=per_seed, per_edge_var_across_seeds=dict(
        mean=float(ev.mean()),p95=float(np.percentile(ev,95)),max=float(ev.max())))
    print("A done",cfg,flush=True)
rep["A_initial_mask"]=A

# ---------- B. what does the VL prefer at init? ----------
mu0,_=all_mu("C",0)
EW=np.stack([g.edge_weight.numpy() for g in dl]).reshape(N,8100)
FCm=EW.reshape(N,90,90)
s_out=FCm.sum(1); s_in=FCm.sum(2)                                # column/row sums (diag incl., const 1 offset)
src=np.tile(np.repeat(np.arange(90),90),(N,1)); dst=np.tile(np.tile(np.arange(90),90),(N,1))
end_str=(np.take_along_axis(s_out,src,1)+np.take_along_axis(s_out,dst,1))
B={}
for nm,v in (("FC_value",EW),("abs_FC",np.abs(EW)),
             ("endpoint_strength_sum",end_str),
             ("src_node_strength",np.take_along_axis(s_out,src,1)),
             ("dst_node_strength",np.take_along_axis(s_out,dst,1))):
    B[nm]=dict(pearson=float(pearsonr(mu0.ravel(),v.ravel())[0]),
               spearman=float(spearmanr(mu0.ravel()[::37],v.ravel()[::37])[0]))
rep["B_mask_vs_structure"]=B; print("B done",flush=True)

# ---------- fixed batch for C/D ----------
g=torch.Generator(); g.manual_seed(S.BASE)
fixed=next(iter(DataLoader(dl,batch_size=32,shuffle=True,generator=g)))

def infonce_probe(cfg,model,view,batch,noise_seed=777):
    """Common-random-numbers InfoNCE: same noise before/after an update."""
    c=S.CFG[cfg]; model.eval(); view.eval()
    with torch.no_grad():
        x,_=model(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
        torch.manual_seed(noise_seed)
        lg=view(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
        _,m=S._mask(c,batch,lg,"cpu")
        xa,_=model(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight*m)
        return float(model.calc_loss(x,xa,temperature=c["batch_T"],sym=c["sym"]))

# ---------- C. gradient direction (one controlled step) ----------
Cres={}
for cfg in ["P","O","C"]:
    r={}
    model,view,mopt,vopt,bank,_=S.build(cfg,0)
    before=infonce_probe(cfg,model,view,fixed)
    S.train_step(cfg,model,view,mopt,vopt,bank,fixed,"cpu",skip_model=True)   # VL step only
    r["case1_freeze_encoder_update_VL"]=dict(infonce_before=before,
        infonce_after=infonce_probe(cfg,model,view,fixed),
        expected="VL does gradient ASCENT on view_loss -> InfoNCE should INCREASE")
    model,view,mopt,vopt,bank,_=S.build(cfg,0)
    before=infonce_probe(cfg,model,view,fixed)
    S.train_step(cfg,model,view,mopt,vopt,bank,fixed,"cpu",skip_view=True)    # model step only
    r["case2_freeze_VL_update_encoder"]=dict(infonce_before=before,
        infonce_after=infonce_probe(cfg,model,view,fixed),
        expected="encoder does DESCENT on model_loss -> InfoNCE should DECREASE")
    Cres[cfg]=r; print("C done",cfg,flush=True)
rep["C_gradient_direction"]=Cres

# ---------- D. mask evolution: 50 steps, one fixed batch (diagnostic, not training) ----------
Dres={}
for cfg in ["P","O","C"]:
    model,view,mopt,vopt,bank,c=S.build(cfg,0)
    def mu_now():
        with torch.no_grad():
            lg=view(fixed.batch,fixed.x,fixed.edge_index,None,fixed.edge_weight)
            if c["mask"]=="symmetric":
                rev=S.compute_reverse_index(fixed.edge_index)
                return torch.sigmoid(S.symmetrize_edge_logits(fixed.edge_index,lg,rev)),lg.squeeze(-1)
            return torch.sigmoid(lg.squeeze(-1)),lg.squeeze(-1)
    mu_a,lg_a=mu_now(); rev=S.compute_reverse_index(fixed.edge_index)
    traj=[]
    for step in range(50):
        S.train_step(cfg,model,view,mopt,vopt,bank,fixed,"cpu")
        if step%5==4 or step==0:
            mu,lg=mu_now()
            raw=torch.sigmoid(lg)
            traj.append(dict(step=step+1,mu_mean=float(mu.mean()),mu_var=float(mu.var()),
                kept_pct=float((mu>0.5).float().mean()),
                asym_gap=float((raw-raw[rev]).abs().mean())))
    mu_b,_=mu_now(); d=(mu_b-mu_a).abs()
    top=torch.topk(d,10)
    ei=fixed.edge_index
    Dres[cfg]=dict(trajectory=traj,
        mu_change_mean=float(d.mean()),mu_change_max=float(d.max()),
        top10_changed_edges=[dict(src=int(ei[0,i])%90,dst=int(ei[1,i])%90,
            delta=float(d[i]),fc=float(fixed.edge_weight[i])) for i in top.indices])
    print("D done",cfg,flush=True)
rep["D_mask_evolution"]=Dres

# ---------- E. edge orientation with ASYMMETRIC mask ----------
torch.manual_seed(0)
n=4; X=torch.randn(n,2,dtype=torch.float64)
E=torch.randn(n,n,dtype=torch.float64)
Bmask=torch.rand(n,n,dtype=torch.float64)                        # deliberately B[i,j] != B[j,i]
assert (Bmask-Bmask.T).abs().max()>0.1
nodes=torch.arange(n); ei=torch.stack([nodes.repeat_interleave(n),nodes.repeat(n)],0)
from unsupervised.convs.wgin_conv import WGINConv
conv=WGINConv(torch.nn.Identity(),eps=0.,train_eps=False,message_relu=False).double()
conv.eps.data=conv.eps.data.double()
out=conv(X,ei,(E*Bmask).reshape(-1))
hand=(E*Bmask).T@X+X                                             # target j receives sum_i E[i,j]B[i,j] x_i
err=float((out-hand).abs().max())
rep["E_orientation"]=dict(max_abs_err_vs_hand=err,
    conclusion="edge k=(i*N+j): edge_index[0]=i SOURCE, edge_index[1]=j TARGET; a mask value on "
               "directed edge (i,j) scales the message INTO TARGET j. Implemented operator is "
               "(I+(E*B)^T)X — consistent with S6. ViewLearner builds logits from "
               "[emb(edge_index[0]) || emb(edge_index[1])] = [src||dst]. No silent transpose "
               "BUG, but an asymmetric mask IS transposed relative to writing Q=(I+(E*B))X.")
assert err<1e-9; print("E done err",err,flush=True)

# ---------- F. symmetric vs asymmetric mask ----------
model,view,_,_,_,c=S.build("C",0)
with torch.no_grad():
    lg=view(fixed.batch,fixed.x,fixed.edge_index,None,fixed.edge_weight).squeeze(-1)
rev=S.compute_reverse_index(fixed.edge_index)
mu_asym=torch.sigmoid(lg); mu_sym=torch.sigmoid((lg+lg[rev])/2)
torch.manual_seed(1); _,m_asym=S._mask(S.CFG["P"],fixed,lg.unsqueeze(-1),"cpu")
torch.manual_seed(1); _,m_sym =S._mask(S.CFG["C"],fixed,lg.unsqueeze(-1),"cpu")
dis_asym=float(((m_asym>0.5)!=(m_asym[rev]>0.5)).float().mean())
dis_sym =float(((m_sym >0.5)!=(m_sym [rev]>0.5)).float().mean())
rep["F_sym_vs_asym"]=dict(mu_corr=float(pearsonr(mu_asym.numpy(),mu_sym.numpy())[0]),
    mu_mean_abs_diff=float((mu_asym-mu_sym).abs().mean()),
    keep_disagreement_rate_asym=dis_asym, keep_disagreement_rate_sym=dis_sym,
    note="disagreement = (i,j) kept while (j,i) dropped; must be 0 under the symmetric path")
print("F done",flush=True)

C7.write_unit(OUT,"S8_DIAGNOSTICS",payload_json=rep)
open(OUT+"/S8_DIAG_DONE","w").write(C7.git_head()+"\n")
print("S8 DIAGNOSTICS COMPLETE",flush=True)
