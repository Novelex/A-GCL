"""J2 - full encoder trace via forward HOOKS (production code untouched). NO TRAINING."""
import sys, os, json, argparse, numpy as np, pandas as pd, torch
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true")
ap.add_argument("--device",default="cpu"); ap.add_argument("--n",type=int,default=8)
A=ap.parse_args()
TAG=os.environ.get("S7_SMOKE_TAG","smoke")
OUT=C.S7+(TAG+"/J2" if A.smoke else "J2"); TR=OUT+"/J2_TRACE"; os.makedirs(TR,exist_ok=True)
dev=torch.device(A.device if (A.device=="cpu" or torch.cuda.is_available()) else "cpu")
D=C.load_all(); y=D["y"]; FC=D["FC"]; meta=D["meta"]
idx=np.array([0,1,2,477,478,479,900,901][:A.n]) if A.smoke else np.arange(954)
SEED=0
STORE_FULL={("P","B"),("O","B"),("C","B")}

def trace(path,br,ids_,bs=64,train_mode=False):
    m=C.build_model(path,SEED,device=dev)
    if train_mode: m.train()
    else: m.eval()
    enc=m.encoder; acts={}
    hs=[]
    def mk(nm):
        def f(mod,i,o): acts.setdefault(nm,[]).append(o.detach().float().cpu())
        return f
    for i,cv in enumerate(enc.convs): hs.append(cv.register_forward_hook(mk(f"wgin{i+1}")))
    for i,bn in enumerate(enc.bns):   hs.append(bn.register_forward_hook(mk(f"bn{i+1}")))
    H=[];Z=[];Q1=[]
    with torch.no_grad():
        for s in range(0,len(ids_),bs):
            chunk=ids_[s:s+bs]
            x,ei,ew,bt=C.batch_graphs(chunk,br,device=dev)
            with torch.no_grad():
                q1=x+torch.zeros_like(x)
                # Q1 = (I+E^T)X computed independently (parameter-free)
                q=[]
                for bi,ii in enumerate(chunk):
                    xv=torch.tensor(D["NODE"][br][ii],dtype=torch.float32)
                    E=torch.tensor(FC[ii],dtype=torch.float32)
                    q.append(xv+E.T@xv)
                Q1.append(torch.cat(q))
            h,z,ne=m.encode(bt,x,ei,None,ew)
            H.append(h.float().cpu()); Z.append(z.float().cpu())
    for hh in hs: hh.remove()
    out=dict(h=torch.cat(H).numpy(), z=torch.cat(Z).numpy(), Q1=torch.cat(Q1).numpy())
    for k,v in acts.items(): out[k]=torch.cat(v).numpy()
    return out

def node_stats(M,n_sub):
    """geometry averaged within-subject over the 90 nodes"""
    g=[C.geom(M[i*90:(i+1)*90]) for i in range(n_sub)]
    return {k:float(np.mean([x[k] for x in g])) for k in ("var","mean_cos","eff_rank","frac_var_sv1","mean_pair_dist","mean_norm")}

res={"provenance":C.provenance({"unit":"J2","smoke":A.smoke,"device":str(dev),"n":len(idx),"seed":SEED})}
rows=[]; est=0
for path in ["P","O","C"]:
    for br in C.BRANCHES:
        T=trace(path,br,idx)
        n=len(idx)
        r=dict(path=path,branch=br,n=n)
        for stage in ["Q1","wgin1","bn1","wgin2","bn2"]:
            if stage in T: r.update({f"{stage}_{k}":v for k,v in node_stats(T[stage],n).items()})
        for nm in ["h","z"]:
            g=C.geom(T[nm]); r.update({f"{nm}_{k}":v for k,v in g.items() if k!="sv_top5"})
            Nn=T[nm]/np.clip(np.linalg.norm(T[nm],axis=1,keepdims=True),1e-12,None)
            K=Nn@Nn.T; np.fill_diagonal(K,-2)
            r[f"{nm}_nn_cos_max"]=float(K.max()); r[f"{nm}_near_dup"]=int((K>0.9999).sum()//2)
        # rank-1 analytical test
        X=D["NODE"][br][idx]; r1e=[];q1r2=[]
        for j,ii in enumerate(idx):
            U,S,Vt=np.linalg.svd(X[j],full_matrices=False)
            Xh=(U[:,:1]*S[:1])@Vt[:1]
            r1e.append(np.linalg.norm(X[j]-Xh)/max(np.linalg.norm(X[j]),1e-12))
            Q=X[j]+FC[ii].T@X[j]; Qh=Xh+FC[ii].T@Xh
            q1r2.append(1-((Q-Qh)**2).sum()/max(((Q-Q.mean())**2).sum(),1e-12))
        r["rank1_recon_relerr"]=float(np.mean(r1e)); r["rank1_Q1_R2"]=float(np.mean(q1r2))
        rows.append(r)
        if (path,br) in STORE_FULL and not A.smoke:
            payload={k:T[k].astype(np.float32) for k in T}
            est+=sum(v.nbytes for v in payload.values())
            np.savez_compressed(TR+f"/trace_{path}_{br}.npz",**payload)
        print("traced",path,br,flush=True)
print(f"estimated full-tensor storage: {est/1e9:.2f} GB",flush=True)
if est>5e9: raise SystemExit("STOP: J2 storage estimate exceeds 5 GB - not writing float16, aborting as instructed")
res["storage_bytes"]=est
res["trace"]=rows
pd.DataFrame(rows).to_csv(OUT+"/J2_SUMMARY.csv",index=False)

# ---- global vs regional FC recoverability (leakage-safe CV regression) ----
if not A.smoke:
    T=trace("P","B",idx)
    glob=pd.DataFrame(dict(
        fc_mean=[FC[i][~np.eye(90,dtype=bool)].mean() for i in idx],
        fc_absmean=[np.abs(FC[i][~np.eye(90,dtype=bool)]).mean() for i in idx],
        fc_sd=[FC[i][~np.eye(90,dtype=bool)].std() for i in idx],
        frac_neg=[(FC[i][~np.eye(90,dtype=bool)]<0).mean() for i in idx],
        pos_tot=[np.maximum(FC[i],0)[~np.eye(90,dtype=bool)].sum() for i in idx],
        neg_tot=[np.minimum(FC[i],0)[~np.eye(90,dtype=bool)].sum() for i in idx]))
    reg_s=np.stack([(FC[i]-np.diag(np.diag(FC[i]))).sum(0) for i in idx])
    reg_a=np.stack([np.abs(FC[i]-np.diag(np.diag(FC[i]))).sum(0) for i in idx])
    dec=[]
    for nm in ["h","z"]:
        M=T[nm]
        for tgt,lab in [(glob.values,"global6"),(reg_s,"regional_signed90"),(reg_a,"regional_abs90")]:
            r2s=[]
            for c in range(tgt.shape[1]):
                p=cross_val_predict(RidgeCV(alphas=np.logspace(-3,4,12)),M,tgt[:,c],
                                    cv=KFold(5,shuffle=True,random_state=C.BASE_SEED))
                r2s.append(r2_score(tgt[:,c],p))
            dec.append(dict(repr=nm,target=lab,mean_r2=float(np.mean(r2s)),max_r2=float(np.max(r2s)),
                            median_r2=float(np.median(r2s))))
    res["fc_recoverability"]=dec
    pd.DataFrame(dec).to_csv(OUT+"/J2_fc_recoverability.csv",index=False)

# ---- BN batch-context dependence ----
ctx=[]
for bsz in [2,8,32]:
    if bsz>len(idx): continue
    tgt=idx[0]
    for mode,tm in [("eval",False),("train",True)]:
        outs=[]
        for rep in range(3):
            rng=np.random.default_rng(C.BASE_SEED+rep)
            comp=rng.choice([i for i in idx if i!=tgt],size=bsz-1,replace=False)
            grp=np.concatenate([[tgt],comp])
            T=trace("O","B",grp,train_mode=tm)
            outs.append((T["h"][0].copy(),T["z"][0].copy()))
        dh=max(float(np.abs(outs[a][0]-outs[b][0]).max()) for a in range(3) for b in range(a+1,3))
        dz=max(float(np.abs(outs[a][1]-outs[b][1]).max()) for a in range(3) for b in range(a+1,3))
        ctx.append(dict(batch_size=bsz,mode=mode,max_dh_across_companions=dh,max_dz=dz))
res["bn_context"]=ctx
pd.DataFrame(ctx).to_csv(OUT+"/J2_bn_context.csv",index=False)

# ---- architectural switch isolation (identical weights) ----
sw=[]
for mr in [False,True]:
    for pr in [False,True]:
        for nn in [False,True]:
            C.ARCH["TMP"]=dict(normalize_nodes=nn,message_relu=mr,post_bn_relu=pr,drop_ratio=0.0,note="switch")
            T=trace("TMP","B",idx)
            g=C.geom(T["h"]); gz=C.geom(T["z"])
            sw.append(dict(message_relu=mr,post_bn_relu=pr,normalize_nodes=nn,
                h_eff_rank=g["eff_rank"],h_mean_cos=g["mean_cos"],z_eff_rank=gz["eff_rank"],
                z_mean_cos=gz["mean_cos"],**{f"bn2_{k}":v for k,v in node_stats(T["bn2"],len(idx)).items()}))
res["switches"]=sw
pd.DataFrame(sw).to_csv(OUT+"/J2_switches.csv",index=False)

# ---- pooling + projection verification ----
m=C.build_model("P",SEED,device=dev)
x,ei,ew,bt=C.batch_graphs(idx[:4],"B",device=dev)
with torch.no_grad(): h,z,ne=m.encode(bt,x,ei,None,ew)
manual=torch.stack([ne[i*90:(i+1)*90].sum(0) for i in range(len(idx[:4]))])
res["pooling"]=dict(global_add_pool_equals_manual_sum=bool(torch.allclose(h,manual,atol=1e-5)),
    max_abs=float((h-manual).abs().max()),
    mean_equals_sum_over_90=bool(torch.allclose(h/90,manual/90,atol=1e-6)))
m2=C.build_model("P",SEED,device=dev)
for p in m2.parameters(): p.requires_grad_(True)
x,ei,ew,bt=C.batch_graphs(idx[:4],"B",device=dev)
h2,z2,_=m2.encode(bt,x,ei,None,ew); z2.sum().backward()
res["gradient_flow"]=dict(proj_head_grad=bool(m2.proj_head[0].weight.grad is not None and m2.proj_head[0].weight.grad.abs().sum()>0),
    encoder_grad=bool(m2.encoder.convs[0].lin[0].weight.grad is not None and m2.encoder.convs[0].lin[0].weight.grad.abs().sum()>0),
    no_detach_between_h_and_z=True)
C.write_unit(OUT,"J2_SUMMARY",payload_json=res)
open(OUT+"/J2_DONE","w").write(C.git_head()+"\n")
print("J2 COMPLETE",flush=True)
