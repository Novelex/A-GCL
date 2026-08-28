"""S17 Wave 1: RowMLPR — a shared per-ROI-row MLP with learned ROI identity.

Everything else is imported from S16. No S16 code is copied or forked.

THE MODEL CONTRACT. The task referred to "all 7 items of the model contract" but did
not list them, so they are derived here from the three existing S16 architectures
(EdgeMLP, BNTR, WGINR), which share one interface exactly:

  1. `ARCH` class attribute naming the architecture.
  2. `__init__(D, ..., p, seed, freeze_encoder, **kw)` that seeds torch AND numpy
     before constructing any parameter.
  3. `self.repr_dim` set to the representation width the probe will consume.
  4. `self.head`, named `head.` so s16_train.GROUPS can partition parameters.
  5. `xavier_init(self)` applied to every Linear.
  6. `freeze_encoder=True` sets requires_grad_(False) on every non-head parameter
     and records `self._frozen_encoder`.
  7. A `train(mode)` override that forces the frozen encoder submodules into eval,
     because requires_grad=False does NOT disable dropout (S16 defect D33).

  Plus the call interface every S16 trainer assumes:
     repr_of(b, edge_vec=None) -> repr
     forward(b, edge_vec=None) -> (repr, logits)
"""
import numpy as np, torch, torch.nn as nn
import sys
sys.path.insert(0, "/users/3171356m/A-GCL/audit/s16/scripts")
from s16_models import xavier_init          # imported, not copied

N_ROI = 90
ID_DIM = 16
OUT_DIM = 32


class RowMLPR(nn.Module):
    """One shared MLP applied to each of the 90 ROI rows INDEPENDENTLY.

    Each ROI's row (its FC profile) is concatenated with a learned 16-d embedding of
    that ROI's identity, so the shared MLP can tell which ROI it is looking at while
    still sharing weights across all 90.

    All 90 outputs are kept IN ORDER and flattened: repr_dim = 90 * 32 = 2880. There
    is deliberately NO sum, mean or any other pooling — S12B localised the FC signal
    loss to `global_add_pool`, which collapsed the representation to chance. Keeping
    the rows ordered and separate is the whole point of this arm.

    need_graph is False for ROWMLP (s16_train sets it only for WGIN), so the batch
    carries b.X of shape [B, 90, D] and no edge index.
    """
    ARCH = "ROWMLP"

    def __init__(self, D, hidden=64, p=0.10, seed=20260818, freeze_encoder=False, **kw):
        super().__init__()
        torch.manual_seed(seed); np.random.seed(seed % 2**32)      # contract item 2
        self.D, self.hidden = int(D), int(hidden)
        self.roi_emb = nn.Embedding(N_ROI, ID_DIM)                 # ROI identity
        self.mlp = nn.Sequential(
            nn.Linear(int(D) + ID_DIM, int(hidden)),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(int(hidden), OUT_DIM),
        )
        self.repr_dim = N_ROI * OUT_DIM                            # 2880, item 3
        self.head = nn.Linear(self.repr_dim, 1)                    # item 4
        xavier_init(self)                                          # item 5
        nn.init.normal_(self.roi_emb.weight, std=0.02)             # Embedding: xavier
                                                                   # skips it (Linear only)
        self._frozen_encoder = bool(freeze_encoder)                # item 6
        if freeze_encoder:
            for n, q in self.named_parameters():
                if not n.startswith("head."):
                    q.requires_grad_(False)

    def train(self, mode=True):                                    # item 7
        """A frozen encoder must also be in EVAL mode: requires_grad=False does not
        disable dropout, and a 'fixed random encoder' whose dropout keeps resampling
        is not fixed (S16 defect D33). The head still trains."""
        super().train(mode)
        if getattr(self, "_frozen_encoder", False):
            self.mlp.eval(); self.roi_emb.eval()
        return self

    def repr_of(self, b, edge_vec=None):
        X = b.X                                    # [B, 90, D]
        B = X.shape[0]
        ids = torch.arange(N_ROI, device=X.device)
        e = self.roi_emb(ids).unsqueeze(0).expand(B, N_ROI, ID_DIM)
        Z = self.mlp(torch.cat([X, e], dim=-1))    # [B, 90, 32] — shared, per row
        return Z.reshape(B, self.repr_dim)         # ordered flatten, NEVER pooled

    def forward(self, b, edge_vec=None):
        r = self.repr_of(b)
        return r, self.head(r).squeeze(-1)
