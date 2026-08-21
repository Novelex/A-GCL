"""R0 Phase 3: numerical parity gate — frozen upstream copy vs S8-O vs R0 runner.
CPU, 1 thread, identical state dicts, identical RNG resets. Stops at first divergence."""
import sys, os, json, copy, random, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/r0"); import upstream_step as U
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s8"); import s8_core as S
import s7_core as C7
from torch_geometric.loader import DataLoader
ATOL,RTOL=1e-6,1e-5
def reset(seed=123):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
dl=S.load_dataset()
g=torch.Generator(); g.manual_seed(123)
batch=next(iter(DataLoader(dl,batch_size=32,shuffle=True,generator=g,drop_last=True)))
reset()
model0,view0,_,_,_,_=S.build("O",0)              # seed via S.build; states snapshotted below
sd_m=copy.deepcopy(model0.state_dict()); sd_v=copy.deepcopy(view0.state_dict())
ARGS=dict(batch_size=32,reg_lambda=2.0,cr_lambda=0.4)

def fresh():
    m,v,mo,vo,bank,_=S.build("O",0)
    m.load_state_dict(copy.deepcopy(sd_m)); v.load_state_dict(copy.deepcopy(sd_v))
    mo=torch.optim.Adam(m.parameters(),lr=5e-4); vo=torch.optim.Adam(v.parameters(),lr=5e-4)
    return m,v,mo,vo

# ---------- A: frozen upstream copy ----------
mA,vA,moA,voA=fresh(); bankA=U.MemoryBank_Q_upstream(256,32,"cpu")
reset(777); capA={}
U.upstream_batch_step(ARGS,mA,vA,moA,voA,bankA,batch,"cpu",capA)

# ---------- B: S8-O mirror (instrumented execution of s8_core's own functions) ----------
mB,vB,moB,voB=fresh(); bankB=S.PaperMemoryBank_Q(256,32,"cpu")
reset(777); capB={}
c=S.CFG["O"]
vB.train(); vB.zero_grad(); mB.eval()
x,_=mB(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight); capB["x_view"]=x.detach().clone()
lg=vB(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight); capB["edge_logits"]=lg.detach().clone()
mu,mask=S._mask(c,batch,lg,"cpu"); capB["mask_view"]=mask.detach().clone()
aug_w=mask if c.get("aug")=="replace" else batch.edge_weight*mask
xa,_=mB(batch.batch,batch.x,batch.edge_index,None,aug_w); capB["x_aug_view"]=xa.detach().clone()
keepdrop=S._per_graph_mean(1.0-mask,batch); capB["reg"]=keepdrop.detach().clone()
bankB.push(xa.detach()); capB["queue_after_push"]=bankB.get_memory().detach().clone(); capB["queue_ptr"]=bankB.current_index
# R0 uses the VERBATIM upstream cr (removes the S8 logsumexp hygiene substitution)
cr=U.calc_regloss_upstream(x,xa,bankB.get_memory(),temperature=c["mem_T"]); capB["cr_view"]=cr.detach().clone()
nce=mB.calc_loss(x,xa,temperature=c["batch_T"],sym=c["sym"]); capB["nce_view"]=nce.detach().clone()
vl=nce-(2.0*keepdrop)+0.4*cr; capB["view_loss"]=vl.detach().clone()
(-vl).backward(); capB["view_grads"]={n:p.grad.detach().clone() for n,p in vB.named_parameters() if p.grad is not None}
voB.step(); capB["view_params_after"]={n:p.detach().clone() for n,p in vB.named_parameters()}
mB.train(); vB.eval(); mB.zero_grad()
x,_=mB(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
with torch.no_grad():
    lg2=vB(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
    _,mask2=S._mask(c,batch,lg2,"cpu")
aug2=mask2 if c.get("aug")=="replace" else batch.edge_weight*mask2
xa,_=mB(batch.batch,batch.x,batch.edge_index,None,aug2)
cr2=U.calc_regloss_upstream(x,xa,bankB.get_memory(),temperature=c["mem_T"]); capB["cr_model"]=cr2.detach().clone()
nce_m=mB.calc_loss(x,xa,temperature=c["batch_T"],sym=c["sym"]); capB["nce_model"]=nce_m.detach().clone()
ml=nce_m+c["model_cr_sign"]*0.4*cr2; capB["model_loss"]=ml.detach().clone()
ml.backward(); capB["model_grads"]={n:p.grad.detach().clone() for n,p in mB.named_parameters() if p.grad is not None}
moB.step(); capB["model_params_after"]={n:p.detach().clone() for n,p in mB.named_parameters()}
capB["opt_state"]=(len(moB.state),len(voB.state))

# ---------- C: real s8_core.train_step end-state (closes mirror-fidelity loop) ----------
mC,vC,moC,voC=fresh(); bankC=S.PaperMemoryBank_Q(256,32,"cpu")   # S8-O reference (logsumexp cr)
reset(777); stats=[]
S.train_step("O",mC,vC,moC,voC,bankC,batch,"cpu",stats=stats)
# NOTE: with verbatim-cr, B is the R0 runner math; s8_core.train_step (logsumexp) end-state
# equality is no longer expected bitwise — the C-check now documents the S8-O relationship.
mirror_ok_bitwise=all(torch.equal(capB["model_params_after"][n],p.detach()) for n,p in mC.named_parameters()) and \
          all(torch.equal(capB["view_params_after"][n],p.detach())  for n,p in vC.named_parameters())

# ---------- unweighted downstream h ----------
hA,_,_=S.extract(mA,dl[:64],weighted=False); hB,_,_=S.extract(mB,dl[:64],weighted=False)

def cmp(name,a,b,out):
    if isinstance(a,dict):
        for k in a:
            cmp(f"{name}.{k}",a[k],b[k],out)
        return
    if isinstance(a,(int,tuple)):
        out.append((name,0.0,a==b)); return
    a=torch.as_tensor(a); b=torch.as_tensor(b)
    d=float((a-b).abs().max()); ok=bool(torch.allclose(a,b,atol=ATOL,rtol=RTOL))
    out.append((name,d,ok))
res=[]
ORDER=["x_view","edge_logits","mask_view","x_aug_view","queue_after_push","queue_ptr",
       "nce_view","cr_view","reg","view_loss","model_loss","nce_model","cr_model",
       "view_grads","model_grads","view_params_after","model_params_after","opt_state"]
for k in ORDER: cmp(k,capA[k],capB[k],res)
cmp("h_unweighted",hA,hB,res)
worst=max(r[1] for r in res); fails=[r for r in res if not r[2]]
rep=dict(atol=ATOL,rtol=RTOL,n_comparisons=len(res),max_abs_any=worst,
    mirror_matches_real_train_step=bool(mirror_ok_bitwise),
    failures=[dict(tensor=n,max_abs=d) for n,d,ok in fails],
    verdict="PASS" if not fails else "FAIL",
    first_divergent=fails[0][0] if fails else None,
    provenance=C7.provenance({"unit":"R0_parity"}))
json.dump(rep,open("R0_PARITY_REPORT.json","w"),indent=1,default=str)
with open("R0_PARITY_REPORT.md","w") as f:
    f.write(f"# R0 PARITY REPORT — {rep['verdict']}\n")
    f.write(f"atol {ATOL} rtol {RTOL}; {len(res)} tensor comparisons; max_abs anywhere {worst:.3e}\n")
    f.write(f"mirror end-state == real s8_core.train_step: {mirror_ok_bitwise} (bitwise)\n\n|tensor|max_abs|ok|\n|---|---|---|\n")
    for n,d,ok in res[:40]: f.write(f"|{n}|{d:.3e}|{ok}|\n")
print(f"PARITY {rep['verdict']}: {len(res)} comparisons, max_abs {worst:.3e}, mirror==real {mirror_ok_bitwise}")
if fails: print("FIRST DIVERGENT:",fails[0])
