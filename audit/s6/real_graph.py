"""S6 Parts 6-11: real-graph magnitude / cancellation / rank / FC-strength analysis.
Uses ALL 954 frozen graphs. No training, no cache modification."""
import sys, numpy as np, pandas as pd, scipy.io as sio, torch, json
sys.path.insert(0,"/users/3171356m/A-GCL")
from unsupervised.convs.wgin_conv import WGINConv
from scipy.stats import pearsonr, spearmanr
A="/users/3171356m/A-GCL/"; OUT="/users/3171356m/agcl_audit_s0/s6/"
coh=pd.read_csv("/users/3171356m/agcl_audit_s0/s1_audit_table.csv")
meta=pd.read_csv("/users/3171356m/agcl_audit_s0/s3c/meta.csv")
ids=coh.subject_id.tolist(); dx=dict(zip(coh.subject_id,coh.dx_storage))
Z=np.load("/users/3171356m/agcl_audit_s0/s3c/X_sources.npz",allow_pickle=True)
assert [str(s) for s in Z["ids"]]==ids
RAW=Z["M1"].astype(np.float64)
def nb(X,k):
    if k=="B": mn=X.min((1,2),keepdims=True); mx=X.max((1,2),keepdims=True); return (X-mn)/(mx-mn)
    if k=="C": mn=X.min(1,keepdims=True); mx=X.max(1,keepdims=True); return (X-mn)/(mx-mn)
    if k=="D": return (X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)
NODE={k:nb(RAW,k) for k in "BCD"}
FC=np.stack([sio.loadmat(f"{A}data/raw/{'ASD' if dx[s]=='ASD' else 'NC'}_ADJ/{s}_adj.mat")["cropped_matrix"]
             for s in ids]).astype(np.float64)
print("loaded", FC.shape, {k:v.shape for k,v in NODE.items()}, flush=True)

# ---- deterministic seeded first-layer MLP (identical weights for every condition) ----
torch.manual_seed(20260819)
MLP=torch.nn.Sequential(torch.nn.Linear(3,32), torch.nn.ReLU(), torch.nn.Linear(32,32)).double()
for p in MLP.parameters(): p.requires_grad_(False)
CONV=WGINConv(MLP, eps=0., train_eps=False, message_relu=False).double()
CONV.eps.data=CONV.eps.data.double()
n=90; nodes=torch.arange(n); EI=torch.stack([nodes.repeat_interleave(n), nodes.repeat(n)],0)

def eff_rank(M):
    s=np.linalg.svd(M-M.mean(0,keepdims=True), compute_uv=False)
    s=s[s>0]
    if len(s)==0: return 0.0,np.array([0.0]),0.0
    p=s/s.sum(); er=float(np.exp(-(p*np.log(p)).sum()))
    return er, s, float((s[0]**2)/(s**2).sum())
def cos_mean(M):
    Nn=M/np.clip(np.linalg.norm(M,axis=1,keepdims=True),1e-12,None)
    C=Nn@Nn.T; iu=np.triu_indices(len(M),1); return float(C[iu].mean())
def stats(M):
    er,s,f1=eff_rank(M)
    d=np.linalg.norm(M[:,None,:]-M[None,:,:],axis=2); iu=np.triu_indices(len(M),1)
    return dict(across_node_var=float(M.var(0).mean()), mean_cos=cos_mean(M),
                eff_rank=er, frac_var_sv1=f1, mean_pair_dist=float(d[iu].mean()),
                mean_norm=float(np.linalg.norm(M,axis=1).mean()))

ABL={"1_raw_signed": lambda E: E,
     "2_abs":        lambda E: np.abs(E),
     "3_pos_only":   lambda E: np.maximum(E,0),
     "4_all_ones":   lambda E: np.ones_like(E),
     "5_offdiag_only":lambda E: E-np.diag(np.diag(E))}

rows=[]; nodrows=[]; graphrows=[]; ablrows=[]
for i,s in enumerate(ids):
    E=FC[i]; Eoff=E-np.diag(np.diag(E)); dg=np.diag(E)
    for k in "BCD":
        X=NODE[k][i]
        resid=X; expl=dg[:,None]*X; offd=Eoff.T@X; Q=resid+expl+offd
        Qc=(np.eye(n)+E).T@X
        assert np.abs(Q-Qc).max()<1e-9
        nr=np.linalg.norm(resid,axis=1); ne=np.linalg.norm(expl,axis=1)
        nt=np.linalg.norm(resid+expl,axis=1); no=np.linalg.norm(offd,axis=1)
        nq=np.linalg.norm(Q,axis=1)
        ub=np.abs(Eoff).T@np.linalg.norm(X,axis=1)          # sum_u |e_uv| ||x_u||
        canc=1.0-no/np.clip(ub,1e-12,None)
        W1=CONV(torch.from_numpy(X), EI, torch.from_numpy(E.reshape(-1).copy())).numpy()
        rows.append(dict(subject=s,branch=k,
            resid_norm=nr.mean(), expl_norm=ne.mean(), total_self_norm=nt.mean(),
            offdiag_norm=no.mean(), q_norm=nq.mean(),
            ratio_off_over_self=float(np.mean(no/np.clip(nt,1e-12,None))),
            ratio_self_over_q=float(np.mean(nt/np.clip(nq,1e-12,None))),
            ub_mean=ub.mean(), cancellation=float(canc.mean()),
            **{f"X_{a}":b for a,b in stats(X).items()},
            **{f"Q_{a}":b for a,b in stats(Q).items()},
            **{f"W1_{a}":b for a,b in stats(W1).items()}))
        if k=="B":
            sv=Eoff.sum(0); av=np.abs(Eoff).sum(0)
            pv=np.maximum(Eoff,0).sum(0); ngv=np.minimum(Eoff,0).sum(0)
            for v in range(n):
                nodrows.append(dict(subject=s,node=v,s_v=sv[v],a_v=av[v],pos_v=pv[v],neg_v=ngv[v],
                    q_norm=nq[v],w1_norm=float(np.linalg.norm(W1[v])),x_norm=float(np.linalg.norm(X[v])),
                    q0=Q[v,0],q1=Q[v,1],q2=Q[v,2],canc=canc[v],offd_norm=no[v]))
            graphrows.append(dict(subject=s, **{f"Qsum{j}":Q.sum(0)[j] for j in range(3)},
                **{f"W1sum{j}":W1.sum(0)[j] for j in range(6)},
                W1sum_norm=float(np.linalg.norm(W1.sum(0))), Qsum_norm=float(np.linalg.norm(Q.sum(0))),
                fc_mean=Eoff[np.triu_indices(n,1)].mean(), fc_absmean=np.abs(Eoff[np.triu_indices(n,1)]).mean(),
                fc_pos=np.maximum(Eoff,0)[np.triu_indices(n,1)].sum(), fc_neg=np.minimum(Eoff,0)[np.triu_indices(n,1)].sum(),
                fc_sd=Eoff[np.triu_indices(n,1)].std(), frac_neg=(Eoff[np.triu_indices(n,1)]<0).mean(),
                alff_level=float(X.sum())))
    if i%200==0: print("subj",i,flush=True)
    if i<200:                                   # ablations on a representative subset
        for an,fn in ABL.items():
            Ea=fn(E); X=NODE[["B"][0]][i]
            Q=(np.eye(n)+Ea).T@X
            W1=CONV(torch.from_numpy(X),EI,torch.from_numpy(Ea.reshape(-1).copy())).numpy()
            ablrows.append(dict(subject=s,ablation=an,
                **{f"Q_{a}":b for a,b in stats(Q).items()},
                **{f"W1_{a}":b for a,b in stats(W1).items()},
                q_norm=float(np.linalg.norm(Q,axis=1).mean())))
pd.DataFrame(rows).to_csv(OUT+"s6_magnitude.csv",index=False)
pd.DataFrame(nodrows).to_csv(OUT+"s6_nodes.csv",index=False)
pd.DataFrame(graphrows).to_csv(OUT+"s6_graph.csv",index=False)
pd.DataFrame(ablrows).to_csv(OUT+"s6_ablation.csv",index=False)
print("SAVED")
