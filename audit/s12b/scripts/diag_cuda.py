"""Diagnose Gate-2 CUDA run-to-run non-determinism: locate it and measure it."""
import sys, os, numpy as np, torch
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12b/scripts"); import s12b_core as B

d = B.load_all(); FC = d["FC"]; X = B.arm_features("B", d["M1B"], FC)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def two_runs(det, bn):
    if det:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)
    outs = []
    for _ in range(2):
        enc = B.S12BEncoder(93, 32, "bn" if bn else "none", True, B.BASE)
        H1, H2 = B.extract_stages(enc, X[:16], FC[:16], "cuda",
                                  bn_train_idx=np.arange(8) if bn else None)
        outs.append((H1.copy(), H2.copy()))
    d1 = float(np.abs(outs[0][0] - outs[1][0]).max())
    d2 = float(np.abs(outs[0][1] - outs[1][1]).max())
    sc = float(np.abs(outs[0][1]).max())
    return d1, d2, sc

for det in (False, True):
    for bn in (True, False):
        a, b, sc = two_runs(det, bn)
        print(f"deterministic={det} bn={bn}: maxdiff H1={a:.3e} H2={b:.3e} "
              f"(|H2|max={sc:.3e}, rel={b/max(sc,1e-12):.2e})", flush=True)

# isolate: is it the scatter-add in propagate?
torch.use_deterministic_algorithms(False)
x, ei, ew, bt = B.make_batch(X, FC, range(16), "cuda")
from unsupervised.convs.wgin_conv import WGINConv
import torch.nn as nn
conv = WGINConv(nn.Identity(), message_relu=True).cuda()
r = [conv.propagate(ei, x=(x, x), edge_weight=ew, size=None) for _ in range(3)]
print("raw propagate maxdiff:",
      max(float((r[0] - r[i]).abs().max()) for i in (1, 2)), flush=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
r = [conv.propagate(ei, x=(x, x), edge_weight=ew, size=None) for _ in range(3)]
print("raw propagate maxdiff (deterministic):",
      max(float((r[0] - r[i]).abs().max()) for i in (1, 2)), flush=True)
print("DIAG_DONE")
