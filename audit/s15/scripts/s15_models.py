"""S15 architectures: BNT-R and WGIN-R. Same contract:
    forward(batch, edge_vec) -> (repr, logits[B])
`repr` is what the frozen probe scores. WGINConv message passing is imported
UNCHANGED from production so comparability with S12A5 holds."""
import sys, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/users/3171356m/A-GCL")
from unsupervised.convs.wgin_conv import WGINConv

# ------------------------------------------------------------------ shared
def gram_schmidt(C):
    """Modified Gram-Schmidt; orthonormal rows (unit L2)."""
    Kc, Hh = C.shape
    E = torch.zeros_like(C)
    for i in range(Kc):
        v = C[i].clone()
        for j in range(i):
            v = v - torch.dot(E[j], v) * E[j]
        n = torch.linalg.norm(v)
        assert float(n) > 1e-8, "degenerate cluster centre"
        E[i] = v / n
    return E

def xavier_init(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

class Batch:
    """Carries everything both architectures need."""
    __slots__ = ("X", "edge_index", "edge_weight", "batch", "n")
    def __init__(self, X, edge_index=None, edge_weight=None, batch=None):
        self.X, self.edge_index, self.edge_weight, self.batch = X, edge_index, edge_weight, batch
        self.n = X.shape[0]

# ------------------------------------------------------------------ BNT-R
class MHSA(nn.Module):
    def __init__(self, H, heads=4, p=0.10):
        super().__init__()
        assert H % heads == 0
        self.h, self.dk = heads, H // heads
        self.q = nn.Linear(H, H, bias=False)
        self.k = nn.Linear(H, H, bias=False)
        self.v = nn.Linear(H, H, bias=False)
        self.o = nn.Linear(H, H)
        self.drop = nn.Dropout(p)
        self.last_attn = None
    def forward(self, Z, keep_attn=False):
        B, N, H = Z.shape
        sh = lambda t: t.view(B, N, self.h, self.dk).transpose(1, 2)
        q, k, v = sh(self.q(Z)), sh(self.k(Z)), sh(self.v(Z))
        # FC edge weights are NEVER injected here. Scores are QK^T/sqrt(dk) only.
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dk), dim=-1)
        if keep_attn: self.last_attn = att.detach()
        out = (self.drop(att) @ v).transpose(1, 2).reshape(B, N, H)
        return self.o(out)

class Block(nn.Module):
    def __init__(self, H, heads=4, p=0.10):
        super().__init__()
        self.n1 = nn.LayerNorm(H); self.attn = MHSA(H, heads, p); self.d1 = nn.Dropout(p)
        self.n2 = nn.LayerNorm(H)
        self.ffn = nn.Sequential(nn.Linear(H, 2 * H), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(2 * H, H))
        self.d2 = nn.Dropout(p)
    def forward(self, Z, keep_attn=False):
        Z = Z + self.d1(self.attn(self.n1(Z), keep_attn))
        Z = Z + self.d2(self.ffn(self.n2(Z)))
        return Z

class BNTR(nn.Module):
    """BNT-R: wide OCREAD, SCALED softmax, SINGLE linear head (no 32-d bottleneck)."""
    ARCH = "BNT"
    def __init__(self, D, K_clusters=32, H=128, n_layers=2, heads=4, p=0.10,
                 seed=20260818, scaled_softmax=True, freeze_encoder=False):
        super().__init__()
        assert H >= D, f"H({H}) must be >= D({D}): never a compression"
        torch.manual_seed(seed); np.random.seed(seed % 2**32)
        self.K, self.H, self.D, self.scaled = K_clusters, H, D, scaled_softmax
        self.inp = nn.Linear(D, H)
        self.blocks = nn.ModuleList([Block(H, heads, p) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(H)
        self.head = nn.Sequential(nn.LayerNorm(K_clusters * H), nn.Dropout(p),
                                  nn.Linear(K_clusters * H, 1))
        xavier_init(self)
        C = torch.empty(K_clusters, H); nn.init.xavier_uniform_(C)
        self.register_buffer("E", gram_schmidt(C))       # BUFFER, never a Parameter
        self.repr_dim = K_clusters * H
        self._last_entropy = float("nan")
        if freeze_encoder:                                # C-RAND: head only trains
            for n, p_ in self.named_parameters():
                if not n.startswith("head."): p_.requires_grad_(False)
    def encode(self, X, keep_attn=False):
        Z = self.inp(X)
        for b in self.blocks: Z = b(Z, keep_attn)
        return self.norm_f(Z)
    def ocread(self, Z_L):
        logits = Z_L @ self.E.t()
        if self.scaled: logits = logits / math.sqrt(self.H)
        P = torch.softmax(logits, dim=-1)
        Z_G = P.transpose(1, 2) @ Z_L
        with torch.no_grad():
            ent = -(P.clamp_min(1e-12).log() * P).sum(-1).mean()
            self._last_entropy = float(ent)
        return Z_G, P
    def repr_of(self, batch, edge_vec=None):
        Z_G, _ = self.ocread(self.encode(batch.X))
        return Z_G.reshape(Z_G.shape[0], -1)
    def forward(self, batch, edge_vec=None):
        r = self.repr_of(batch, edge_vec)
        return r, self.head(r).squeeze(-1)

# ------------------------------------------------------------------ WGIN-R
class WGINR(nn.Module):
    """WGINConv UNCHANGED ((I+A.E)H, same eps, same relu(x_j) in message()).
    Repaired around it: expansion inp, LayerNorm, no F.normalize, ROI-concat."""
    ARCH = "WGIN"
    def __init__(self, D, hidden=128, n_layers=2, p=0.10, seed=20260818,
                 readout="roi", freeze_encoder=False):
        super().__init__()
        assert hidden >= D, f"hidden({hidden}) must be >= D({D}): fixes B2"
        torch.manual_seed(seed); np.random.seed(seed % 2**32)
        self.hidden, self.readout, self.n_layers = hidden, readout, n_layers
        self.inp = nn.Linear(D, hidden)                              # expansion (B2)
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden))
            self.convs.append(WGINConv(mlp, message_relu=True))       # UNCHANGED
            self.norms.append(nn.LayerNorm(hidden))                   # LayerNorm (B3)
        self.drop = nn.Dropout(p)
        self.repr_dim = (90 * hidden) if readout == "roi" else hidden # ROI-concat (B5)
        self.head = nn.Sequential(nn.LayerNorm(self.repr_dim), nn.Dropout(p),
                                  nn.Linear(self.repr_dim, 1))
        xavier_init(self)
        if freeze_encoder:
            for n, p_ in self.named_parameters():
                if not n.startswith("head."): p_.requires_grad_(False)
    def encode(self, batch):
        B, N, _ = batch.X.shape
        x = self.inp(batch.X).reshape(B * N, self.hidden)
        for i in range(self.n_layers):
            x = self.convs[i](x, batch.edge_index, batch.edge_weight)
            x = self.norms[i](x)                       # NO F.normalize anywhere (B4)
            if i < self.n_layers - 1: x = self.drop(F.relu(x))
            else: x = self.drop(x)
        return x.reshape(B, N, self.hidden)
    def repr_of(self, batch, edge_vec=None):
        node = self.encode(batch)
        if self.readout == "roi": return node.reshape(node.shape[0], -1)
        return node.sum(1)                             # logged control arm only
    def forward(self, batch, edge_vec=None):
        r = self.repr_of(batch, edge_vec)
        return r, self.head(r).squeeze(-1)

# ------------------------------------------------------------------ helpers
def n_trainable(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))

def edge_struct(n=90):
    idx = torch.arange(n)
    return torch.stack([idx.repeat_interleave(n), idx.repeat(n)], 0)

_EI = {}
def make_batch(X, FC, idxs, need_graph):
    """X [N,90,D] numpy, FC [N,90,90] numpy. Self-loops ARE included: edge_index
    contains (i,i) with FC[i,i]=1.0 AND WGINConv adds (1+eps)*x_r, so a node's own
    features are counted TWICE. LOGGED FORK, not silently fixed (S12A5 ran with it)."""
    idxs = np.asarray(idxs); B = len(idxs)
    Xt = torch.tensor(X[idxs], dtype=torch.float32)
    if not need_graph: return Batch(Xt)
    if B not in _EI:
        e0 = edge_struct()
        _EI[B] = torch.cat([e0 + j * 90 for j in range(B)], 1)
    ei = _EI[B]
    ew = torch.tensor(FC[idxs].reshape(-1), dtype=torch.float32)
    bt = torch.arange(B).repeat_interleave(90)
    return Batch(Xt, ei, ew, bt)

class EMA:
    """Exponential moving average of weights, decay 0.999."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if torch.is_floating_point(v)}
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
    def state_dict(self, model):
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k in self.shadow: sd[k] = self.shadow[k].to(sd[k].dtype)
        return sd

def build_model(arch, D, seed, K_or_hidden, freeze_encoder=False, readout="roi",
                scaled_softmax=True, p=0.10, H=128):
    if arch == "BNT":
        return BNTR(D, K_clusters=K_or_hidden, H=H, seed=seed, p=p,
                    scaled_softmax=scaled_softmax, freeze_encoder=freeze_encoder)
    if arch == "WGIN":
        return WGINR(D, hidden=K_or_hidden, seed=seed, p=p, readout=readout,
                     freeze_encoder=freeze_encoder)
    raise ValueError(arch)
