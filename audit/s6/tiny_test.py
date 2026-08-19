"""S6 Parts 2-5, 13: tiny asymmetric float64 hand test of WGINConv."""
import sys, torch, numpy as np
sys.path.insert(0,"/users/3171356m/A-GCL")
torch.set_printoptions(precision=10, sci_mode=False)
from unsupervised.convs.wgin_conv import WGINConv
DEV=sys.argv[1] if len(sys.argv)>1 else "cpu"
dt=torch.float64
dev=torch.device(DEV)
print(f"### DEVICE={DEV} torch={torch.__version__} dtype={dt}")

# ---- 4-node, 2-feature graph; x has positive AND negative components ----
X=torch.tensor([[ 1.5, -2.0],
                [-0.5,  3.0],
                [ 2.0,  0.25],
                [-1.25,-0.75]], dtype=dt, device=dev)
# ---- intentionally ASYMMETRIC E, with positive / negative / fractional /
#      zero / explicit self edges ----
E=torch.tensor([[ 1.0,  0.5, -0.75,  0.0],     # self=1, pos, neg, ZERO
                [ 0.25, 1.0,  0.0,  -1.5],     # asymmetric vs [0,1]=0.5
                [ 2.0,  0.0,  1.0,   0.3],     # asymmetric vs [0,2]=-0.75
                [ 0.0,  0.6, -0.2,   1.0]], dtype=dt, device=dev)
N=4
nodes=torch.arange(N,device=dev)
src=nodes.repeat_interleave(N); dst=nodes.repeat(N)      # loader-identical layout
edge_index=torch.stack([src,dst],0)
edge_weight=E.reshape(-1).clone()
print("E (asymmetric):\n",E)
print("E - E^T max abs:", float((E-E.T).abs().max()), "-> genuinely asymmetric")

def run(mrelu, ew=None, ei=None):
    c=WGINConv(torch.nn.Identity(), eps=0., train_eps=False, message_relu=mrelu).to(dev).to(dt)
    c.eps.data=c.eps.data.to(dt)
    return c(X, edge_index if ei is None else ei, edge_weight if ew is None else ew)

print("\n" + "="*76); print("PART 2 — HAND CALCULATION vs CODE (message_relu=False, eps=0)"); print("="*76)
out=run(False)
hand_E  = E @ X + X          # if operator were  E X
hand_ET = E.T @ X + X        # if operator were  E^T X
print("code output q:\n", out)
print("hand (E X + X):\n", hand_E)
print("hand (E^T X + X):\n", hand_ET)
e1=float((out-hand_E).abs().max()); e2=float((out-hand_ET).abs().max())
print(f"\nmax_abs_error vs  E X + X : {e1:.3e}")
print(f"max_abs_error vs E^T X + X : {e2:.3e}")
print(f"=> implemented operator is {'E^T (transpose)' if e2<e1 else 'E'};  passes 1e-6: {min(e1,e2)<1e-6}")
# explicit per-node scalar check on node 0
m0=sum(E[u,0]*X[u] for u in range(N))
print(f"\nnode 0 hand m_0 = sum_u E[u,0]*x_u = {m0.tolist()}   q_0 = m_0 + x_0 = {(m0+X[0]).tolist()}")
print(f"node 0 code                                                      = {out[0].tolist()}")
print("\nORIENTATION PROOF: edge k=i*N+j has edge_index[0]=i (source), edge_index[1]=j (target),")
print("weight E[i,j]. PyG flow='source_to_target' gathers x_j:=x[edge_index[0]] and scatters")
print("into out[edge_index[1]]. Hence out[j] = sum_i E[i,j] x_i = (E^T X)[j].")
print("For SYMMETRIC real FC E^T==E so this is invisible; for a learned asymmetric Bernoulli")
print("mask it is NOT. RECORD FOR S8.")

print("\n" + "="*76); print("PART 3 — SELF-LOOP DOUBLE COUNT"); print("="*76)
Xz=X.clone()
E_nodiag=E.clone(); E_nodiag.fill_diagonal_(0.0)
out_with = run(False)
out_wo   = run(False, ew=E_nodiag.reshape(-1).clone())
resid = X
explicit = torch.diag(E).unsqueeze(1)*X
print("residual self (1+eps)*x_v      :", resid[0].tolist(), " (eps=0 -> coefficient 1.0)")
print("explicit FC self e_vv*x_v      :", explicit[0].tolist(), " (e_00 = 1.0)")
print("TOTAL SELF = residual+explicit :", (resid+explicit)[0].tolist(), " = 2*x_0:", (2*X[0]).tolist())
print(f"  with diagonal, q - offdiag  = {(out_with - (E_nodiag.T@X))[0].tolist()}")
print(f"  => equals 2*x_0            : {torch.allclose(out_with-(E_nodiag.T@X), 2*X, atol=1e-12)}")
print(f"  diagonal REMOVED, q - offdiag = {(out_wo - (E_nodiag.T@X))[0].tolist()}  == x_0: "
      f"{torch.allclose(out_wo-(E_nodiag.T@X), X, atol=1e-12)}")
print(f"  max|q_with - q_without - x| = {float((out_with-out_wo-X).abs().max()):.3e}  "
      f"(exactly the extra x_v)")

print("\n" + "="*76); print("PART 4 — message_relu AND NEGATIVE EDGES"); print("="*76)
oF=run(False); oT=run(True)
print("message_relu=False q:\n",oF)
print("message_relu=True  q:\n",oT)
print("hand False: E^T X + X          max err:", float((oF-(E.T@X+X)).abs().max()))
print("hand True : E^T relu(X) + X    max err:", float((oT-(E.T@torch.relu(X)+X)).abs().max()))
print(f"max|True - False| = {float((oT-oF).abs().max()):.6f}  -> ReLU is NOT a no-op here")
u,v=1,0
print(f"\nworked example: negative x times negative FC, edge u={u} -> v={v}, E[{u},{v}]={float(E[u,v])}")
print(f"  x_{u} = {X[u].tolist()}  (component 0 is negative)")
print(f"  A (relu=False): e*x_u        = {(E[u,v]*X[u]).tolist()}")
print(f"  B (relu=True) : e*relu(x_u)  = {(E[u,v]*torch.relu(X[u])).tolist()}")
print("  => with relu=False a negative feature times a positive/negative edge keeps its sign")
print("     and can ADD positively; with relu=True the negative component is zeroed BEFORE")
print("     weighting, so it contributes nothing regardless of edge sign.")

print("\n" + "="*76); print("PART 5 — GRADIENT CHECK (float64, autograd vs central differences)"); print("="*76)
torch.manual_seed(0)
lin=torch.nn.Linear(2,2).to(dev).to(dt)
def f(x_,ew_,W,b):
    c=WGINConv(torch.nn.Identity(), eps=0., train_eps=False, message_relu=False).to(dev).to(dt)
    c.eps.data=c.eps.data.to(dt)
    q=c(x_, edge_index, ew_)
    return ((q@W.T+b)**2).sum()
x_=X.clone().requires_grad_(True); ew_=edge_weight.clone().requires_grad_(True)
W=lin.weight.detach().clone().requires_grad_(True); b=lin.bias.detach().clone().requires_grad_(True)
loss=f(x_,ew_,W,b); loss.backward()
def fd(t, idx, h=1e-6):
    tp=t.detach().clone(); tm=t.detach().clone()
    tp.view(-1)[idx]+=h; tm.view(-1)[idx]-=h
    args=[x_.detach(),ew_.detach(),W.detach(),b.detach()]
    pos=[i for i,a in enumerate([x_,ew_,W,b]) if a is t][0]
    a1=list(args); a1[pos]=tp; a2=list(args); a2[pos]=tm
    return (f(*a1).item()-f(*a2).item())/(2*h)
for name,t in (("x",x_),("edge_weight",ew_),("Linear.weight",W),("Linear.bias",b)):
    n=t.numel(); ana=t.grad.detach().view(-1); num=torch.tensor([fd(t,i) for i in range(n)],dtype=dt,device=dev)
    ae=float((ana-num).abs().max()); re=float(((ana-num).abs()/num.abs().clamp(min=1e-8)).max())
    print(f"  {name:<14} n={n:2d}  max_abs_err={ae:.3e}  max_rel_err={re:.3e}  "
          f"finite={bool(torch.isfinite(ana).all())}")

print("\n" + "="*76); print("PART 13 — SANITY"); print("="*76)
c=WGINConv(torch.nn.Identity(), eps=0., train_eps=False, message_relu=False).to(dev).to(dt); c.eps.data=c.eps.data.to(dt)
o=c(X,edge_index,edge_weight)
print(f"  output shape {tuple(o.shape)}  finite={bool(torch.isfinite(o).all())}")
z=c(X,edge_index,torch.zeros_like(edge_weight))
print(f"  edge_weight=0 -> q == x ? {torch.allclose(z,X,atol=1e-14)}  (max err {float((z-X).abs().max()):.1e})")
I=torch.eye(N,dtype=dt,device=dev)
oi=c(X,edge_index,I.reshape(-1).clone())
print(f"  E=Identity   -> q == 2x ? {torch.allclose(oi,2*X,atol=1e-14)}")
o2=c(X,edge_index,edge_weight)
print(f"  deterministic repeat: bitwise identical = {torch.equal(o,o2)}")
print(f"  negative edges consumed (any negative message contributes): "
      f"{not torch.allclose(c(X,edge_index,E.clamp(min=0).reshape(-1).clone()),o)}")
torch.save({"q_false":oF.cpu(),"q_true":oT.cpu(),"X":X.cpu(),"E":E.cpu()},
           f"/users/3171356m/agcl_audit_s0/s6/tiny_{DEV}.pt")
