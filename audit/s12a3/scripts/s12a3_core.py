"""S12A3 core — encoder retention, RANDOM encoders only. Real production code path
(TUEncoder/GInfoMinMax), identity input, flatten ROI-aware readout. No training."""
import sys, os, json, hashlib, numpy as np, torch
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
sys.path.insert(0, "/users/3171356m/A-GCL")
from unsupervised.encoder.tu_encoder import TUEncoder
from unsupervised.learning.ginfominmax import GInfoMinMax

BASE = 20260818
S12A3 = "/users/3171356m/agcl_audit_s0/s12a3/"
ARMS = dict(
    A=dict(emb_dim=32,  normalize_nodes=True,  post_bn_relu=True),
    B=dict(emb_dim=32,  normalize_nodes=False, post_bn_relu=True),
    C=dict(emb_dim=32,  normalize_nodes=True,  post_bn_relu=False),
    D=dict(emb_dim=32,  normalize_nodes=False, post_bn_relu=False),
    E=dict(emb_dim=64,  normalize_nodes=True,  post_bn_relu=True),
    F=dict(emb_dim=128, normalize_nodes=True,  post_bn_relu=True))

def build_encoder_arm(n_feat, seed, cfg):
    """Same contract as S12A1 build_encoder (positional GInfoMinMax — S8 lesson),
    with the three audited knobs taken from cfg. Everything else = corrected-C."""
    torch.manual_seed(seed); np.random.seed(seed % (2**32))
    enc = TUEncoder(num_dataset_features=n_feat, emb_dim=cfg["emb_dim"],
                    num_gc_layers=2, drop_ratio=0.3, pooling_type="standard",
                    is_infograph=False, normalize_nodes=cfg["normalize_nodes"],
                    message_relu=True, post_bn_relu=cfg["post_bn_relu"])
    m = GInfoMinMax(enc, cfg["emb_dim"])
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m

def load_inputs():
    """Gated load: S11 manifest order; X_id = [M1_B | I90] exactly as S12A1."""
    df, X_fc, y, ids, gh = A1.load_gate()
    FC, Xold, stats = A1.load_tensors(df)
    assert stats["mism"] == 0 and stats["x_max"] == 0.0
    I90 = np.eye(90, dtype=np.float32)
    Xid = np.concatenate([Xold, np.repeat(I90[None], 954, 0)], axis=2)  # [954,90,93]
    assert Xid.shape == (954, 90, 93)
    return df, FC, Xid, y, ids

@torch.no_grad()
def extract_nodes(model, Xid, FC, bs=64):
    """Final node output (2nd return of real forward) for all 954, batched.
    Reuses the S12A1 proven batching/stage helper; 'final_postnorm' is bound to the
    encoder's actual node return, whatever the flags."""
    outs = []
    for lo in range(0, len(Xid), bs):
        st = A1.stages_for_subject_batch(model, Xid[lo:lo + bs],
                                         FC[lo:lo + bs], want_z=False)
        outs.append(st["final_postnorm"])
    N = np.concatenate(outs, 0)
    assert np.isfinite(N).all()
    return N.astype(np.float32)

def arr_sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def emb_path(arm, sidx): return f"{S12A3}emb/{arm}_s{sidx}.npz"
