"""S16 architectures: WGIN-R and BNT-R. Contract: forward(batch, edge_vec)->(repr,logits).
WGINConv message passing UNCHANGED from production (comparability with S12A5).
Supports DENSE (identical topology) and SPARSE (subject-specific, E=pos_zero) graphs."""
import sys, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/users/3171356m/A-GCL")
from unsupervised.convs.wgin_conv import WGINConv

def gram_schmidt(C):
    E = torch.zeros_like(C)
    for i in range(C.shape[0]):
        v = C[i].clone()
        for j in range(i): v = v - torch.dot(E[j], v) * E[j]
        n = torch.linalg.norm(v); assert float(n) > 1e-8, "degenerate cluster centre"
        E[i] = v / n
    return E

def xavier_init(mod):
    for m in mod.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

class Batch:
    __slots__ = ("X","edge_index","edge_weight","batch","n")
    def __init__(self, X, ei=None, ew=None, b=None):
        self.X, self.edge_index, self.edge_weight, self.batch = X, ei, ew, b
        self.n = X.shape[0]

# ---------------------------------------------------------------- BNT-R
class MHSA(nn.Module):
    def __init__(self, H, heads=4, p=0.10):
        super().__init__(); assert H % heads == 0
        self.h, self.dk = heads, H // heads
        self.q = nn.Linear(H,H,bias=False); self.k = nn.Linear(H,H,bias=False)
        self.v = nn.Linear(H,H,bias=False); self.o = nn.Linear(H,H)
        self.drop = nn.Dropout(p); self.last_attn = None
    def forward(self, Z, keep=False):
        B,N,H = Z.shape
        sh = lambda t: t.view(B,N,self.h,self.dk).transpose(1,2)
        q,k,v = sh(self.q(Z)), sh(self.k(Z)), sh(self.v(Z))
        att = torch.softmax(q @ k.transpose(-2,-1) / math.sqrt(self.dk), dim=-1)
        if keep: self.last_attn = att.detach()
        return self.o((self.drop(att) @ v).transpose(1,2).reshape(B,N,H))

class Block(nn.Module):
    def __init__(self, H, heads=4, p=0.10):
        super().__init__()
        self.n1=nn.LayerNorm(H); self.attn=MHSA(H,heads,p); self.d1=nn.Dropout(p)
        self.n2=nn.LayerNorm(H)
        self.ffn=nn.Sequential(nn.Linear(H,2*H),nn.GELU(),nn.Dropout(p),nn.Linear(2*H,H))
        self.d2=nn.Dropout(p)
    def forward(self, Z, keep=False):
        Z = Z + self.d1(self.attn(self.n1(Z), keep))
        return Z + self.d2(self.ffn(self.n2(Z)))

class BNTR(nn.Module):
    ARCH = "BNT"
    def __init__(self, D, K_clusters=32, H=128, n_layers=2, heads=4, p=0.10,
                 seed=20260818, scaled=True, freeze_encoder=False):
        super().__init__(); assert H >= D, f"H({H}) must be >= D({D})"
        torch.manual_seed(seed); np.random.seed(seed % 2**32)
        self.K,self.H,self.D,self.scaled = K_clusters,H,D,scaled
        self.inp = nn.Linear(D,H)
        self.blocks = nn.ModuleList([Block(H,heads,p) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(H)
        self.head = nn.Sequential(nn.LayerNorm(K_clusters*H), nn.Dropout(p),
                                  nn.Linear(K_clusters*H, 1))
        xavier_init(self)
        C = torch.empty(K_clusters,H); nn.init.xavier_uniform_(C)
        self.register_buffer("E", gram_schmidt(C))
        self.repr_dim = K_clusters*H; self._entropy = float("nan")
        if freeze_encoder:
            for n,q in self.named_parameters():
                if not n.startswith("head."): q.requires_grad_(False)
    def encode(self, X, keep=False):
        Z = self.inp(X)
        for b in self.blocks: Z = b(Z, keep)
        return self.norm_f(Z)
    def ocread(self, Z):
        lg = Z @ self.E.t()
        if self.scaled: lg = lg / math.sqrt(self.H)
        P = torch.softmax(lg, dim=-1)
        ZG = P.transpose(1,2) @ Z
        with torch.no_grad():
            self._entropy = float(-(P.clamp_min(1e-12).log()*P).sum(-1).mean())
        return ZG, P
    def repr_of(self, b, edge_vec=None):
        ZG,_ = self.ocread(self.encode(b.X)); return ZG.reshape(ZG.shape[0],-1)
    def forward(self, b, edge_vec=None):
        r = self.repr_of(b); return r, self.head(r).squeeze(-1)

# ---------------------------------------------------------------- WGIN-R
class WGINR(nn.Module):
    ARCH = "WGIN"
    def __init__(self, D, hidden=128, n_layers=2, p=0.10, seed=20260818,
                 readout="roi", freeze_encoder=False):
        super().__init__(); assert hidden >= D, f"hidden({hidden}) must be >= D({D})"
        torch.manual_seed(seed); np.random.seed(seed % 2**32)
        self.hidden,self.readout,self.n_layers = hidden,readout,n_layers
        self.inp = nn.Linear(D,hidden)
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
            self.convs.append(WGINConv(mlp, message_relu=True))   # UNCHANGED
            self.norms.append(nn.LayerNorm(hidden))               # LayerNorm, not BN
        self.drop = nn.Dropout(p)
        self.repr_dim = (90*hidden) if readout=="roi" else hidden
        self.head = nn.Sequential(nn.LayerNorm(self.repr_dim), nn.Dropout(p),
                                  nn.Linear(self.repr_dim,1))
        xavier_init(self)
        if freeze_encoder:
            for n,q in self.named_parameters():
                if not n.startswith("head."): q.requires_grad_(False)
    def encode(self, b):
        B,N,_ = b.X.shape
        x = self.inp(b.X).reshape(B*N, self.hidden)
        for i in range(self.n_layers):
            x = self.convs[i](x, b.edge_index, b.edge_weight)
            x = self.norms[i](x)                     # NO F.normalize anywhere
            x = self.drop(F.relu(x)) if i < self.n_layers-1 else self.drop(x)
        return x.reshape(B,N,self.hidden)
    def repr_of(self, b, edge_vec=None):
        node = self.encode(b)
        return node.reshape(node.shape[0],-1) if self.readout=="roi" else node.sum(1)
    def forward(self, b, edge_vec=None):
        r = self.repr_of(b); return r, self.head(r).squeeze(-1)

class EdgeMLP(nn.Module):
    """EdgeMLP: 4005 FC upper triangle -> hidden -> 32. Architecturally identical to
    S12A5 arm C, but an ORDINARY C6 arm — the bridge role is withdrawn because the
    training recipes differ (see AGGREGATION_SPEC.md section 6)."""
    ARCH="EDGEMLP"
    ARCH_PARITY = ("BITWISE-IDENTICAL to s12a5_core.EdgeMLP + ArmModel(arm='C'): "
                   "Linear(4005,256)-ReLU-Dropout(0.3)-Linear(256,32), head = plain "
                   "Linear(32,1). Dropout is HARDCODED 0.3 (NOT S16's 0.10) and the "
                   "head has NO LayerNorm, so the only difference from S12A5 arm C is "
                   "the training-set size. See C6_SHORT.md for the residual recipe "
                   "confound, which architecture parity does NOT remove.")
    def __init__(self, D_edges=4005, hidden=256, p=None, seed=20260818,
                 freeze_encoder=False, **kw):
        super().__init__(); torch.manual_seed(seed); np.random.seed(seed%2**32)
        # p is IGNORED: parity with S12A5 arm C requires dropout 0.3 exactly.
        self.net=nn.Sequential(nn.Linear(D_edges,hidden),nn.ReLU(),nn.Dropout(0.3),
                               nn.Linear(hidden,32))
        self.repr_dim=32
        self.head=nn.Linear(32,1)                    # plain Linear, as S12A5 arm C
        xavier_init(self)
        if freeze_encoder:
            for n,q in self.named_parameters():
                if not n.startswith("head."): q.requires_grad_(False)
    def repr_of(self,b,edge_vec=None): return self.net(b.X)
    def forward(self,b,edge_vec=None):
        r=self.repr_of(b); return r, self.head(r).squeeze(-1)

# ---------------------------------------------------------------- batching
def n_trainable(m): return int(sum(p.numel() for p in m.parameters() if p.requires_grad))

_DENSE_EI = {}
def _dense_ei(B):
    if B not in _DENSE_EI:
        n = torch.arange(90)
        e0 = torch.stack([n.repeat_interleave(90), n.repeat(90)], 0)
        _DENSE_EI[B] = torch.cat([e0 + j*90 for j in range(B)], 1)
    return _DENSE_EI[B]

def make_batch(X, FC, idxs, need_graph, sparse=False):
    """Self-loops ARE included: (i,i) with FC[i,i]=1.0 AND WGINConv adds (1+eps)x_r,
    so a node's own features count TWICE. LOGGED FORK (S12A5 ran with it)."""
    idxs = np.asarray(idxs); B = len(idxs)
    Xt = torch.tensor(X[idxs], dtype=torch.float32)
    if not need_graph: return Batch(Xt)
    if not sparse:
        ew = torch.tensor(FC[idxs].reshape(-1), dtype=torch.float32)
        return Batch(Xt, _dense_ei(B), ew, torch.arange(B).repeat_interleave(90))
    eis, ews = [], []
    for j, i in enumerate(idxs):
        f = FC[i]; r, c = np.nonzero(f)                      # subject-specific
        eis.append(torch.from_numpy(np.stack([r, c]).astype(np.int64)) + j*90)
        ews.append(torch.from_numpy(f[r, c].astype(np.float32)))
    return Batch(Xt, torch.cat(eis,1), torch.cat(ews),
                 torch.arange(B).repeat_interleave(90))

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k,v in model.state_dict().items()
                       if torch.is_floating_point(v)}
    def update(self, model):
        for k,v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1-self.decay)
    def state_dict(self, model):
        sd = {k: v.detach().clone() for k,v in model.state_dict().items()}
        for k in self.shadow: sd[k] = self.shadow[k].to(sd[k].dtype)
        return sd

def build_model(arch, D, seed, kh, freeze_encoder=False, readout="roi", p=0.10,
                H=128, scaled_softmax=True):
    if arch=="EDGEMLP": return EdgeMLP(D_edges=D, hidden=kh, p=p, seed=seed,
                                       freeze_encoder=freeze_encoder)
    if arch=="BNT":  return BNTR(D, K_clusters=kh, H=H, seed=seed, p=p,
                                 scaled=scaled_softmax, freeze_encoder=freeze_encoder)
    if arch=="WGIN": return WGINR(D, hidden=kh, seed=seed, p=p, readout=readout,
                                  freeze_encoder=freeze_encoder)
    raise ValueError(arch)
