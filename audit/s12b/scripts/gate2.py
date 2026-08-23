"""GATE 2 — forward correctness. Toy-graph hand verification, self-loop fork,
feature bitwise check, finiteness, determinism (CPU+GPU), save/reload, production
parity, A1 identity. Every check asserts; report written only if ALL pass."""
import sys, os, json, time, copy, numpy as np, torch
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12b/scripts"); import s12b_core as B
sys.path.insert(0, "/users/3171356m/A-GCL")
from unsupervised.convs.wgin_conv import WGINConv
from unsupervised.encoder.tu_encoder import TUEncoder
import torch.nn as nn

R = []
def check(name, ok, detail=""):
    R.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name, detail, flush=True)
    assert ok, f"GATE2 FAIL: {name} {detail}"

def toy_check(mrelu, dtype):
    """4-node toy graph, hand-computed WGIN aggregation. Includes a negative edge
    weight and a node whose message sum flips sign."""
    torch.manual_seed(0)
    x = torch.tensor([[1., 2.], [-3., 0.5], [0., -1.], [2., 2.]], dtype=dtype)
    # edges (src, dst, w): self-loops w=1 on all nodes + off-diagonal incl. negative
    E = [(0, 0, 1.), (1, 1, 1.), (2, 2, 1.), (3, 3, 1.),
         (0, 1, 0.5), (1, 0, 0.5), (2, 3, -0.8), (3, 2, -0.8), (0, 2, 2.0), (2, 0, 2.0)]
    ei = torch.tensor([[e[0] for e in E], [e[1] for e in E]], dtype=torch.long)
    ew = torch.tensor([e[2] for e in E], dtype=dtype)
    conv = WGINConv(nn.Identity(), message_relu=mrelu).to(dtype)
    out = conv(x, ei, ew)
    f = (lambda v: torch.clamp(v, min=0)) if mrelu else (lambda v: v)
    hand = torch.zeros_like(x)
    for j in range(4):
        s = torch.zeros(2, dtype=dtype)
        for (a, b, w) in E:
            if b == j: s = s + w * f(x[a])
        hand[j] = s + (1 + 0.0) * x[j]                    # eps=0, non-trainable
    err = float((out - hand).abs().max())
    tol = 1e-6 if dtype == torch.float64 else 1e-4
    check(f"toy_wgin mrelu={mrelu} {str(dtype)[-9:]}", err < tol, f"max_err={err:.2e}")
    # sign-flip evidence: node 2 aggregate sign differs from its own feature sign
    return err

def main():
    t0 = time.time()
    d = B.load_all(); FC, M1B, y = d["FC"], d["M1B"], d["y"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    # (1) hand verification, both message_relu branches, both precisions
    for mr in (True, False):
        toy_check(mr, torch.float64); toy_check(mr, torch.float32)

    # (2) self-loop fork: dataset edges include (i,i) with FC diag=1 AND (1+eps)x
    x0, ei0, ew0, _ = B.make_batch(arm_X("A", d), FC, [0], "cpu")
    n_self = int(((ei0[0] == ei0[1])).sum())
    diag = FC[0][np.arange(90), np.arange(90)]
    check("selfloop_fork", n_self == 90 and np.allclose(diag, 1.0, atol=1e-6),
          f"90 self-edges, FC diag=1 -> own features counted TWICE (kept, logged, all arms)")

    # (3) arm C features == FC rows bitwise, 8 random subjects x 8 ROIs
    XC = B.arm_features("C", M1B, FC)
    rng = np.random.default_rng(B.BASE)
    for s in rng.integers(0, 954, 8):
        for r in rng.integers(0, 90, 8):
            assert np.array_equal(XC[s, r, 3:], FC[s, r].astype(np.float32))
    check("armC_fcrow_bitwise", True, "8x8 spot checks exact")

    # (8) A1 identity: param-free X + FC^T@X == conv.propagate(mrelu=F) + x
    conv = WGINConv(nn.Identity(), message_relu=False)
    a1 = B.a1_stage(XC[:1].astype(np.float32), FC[:1])[0]
    x0c, _, _, _ = B.make_batch(XC, FC, [0], "cpu")
    q2 = conv.propagate(ei0, x=(x0c, x0c), edge_weight=ew0, size=None) + (1 + conv.eps) * x0c
    errA1 = float(np.abs(q2.detach().numpy() - a1).max())
    check("A1_identity", errA1 < 1e-3, f"max_err={errA1:.2e} (f32 90-term sums)")

    # (4) finiteness for every arm/stage, 8 subjects, emb32 bn mrelu=T
    for arm in B.ARMS:
        X = B.arm_features(arm, M1B, FC)
        enc = B.S12BEncoder(X.shape[2], 32, "bn", True, B.BASE)
        H1, H2 = B.extract_stages(enc, X[:8], FC[:8], "cpu")
        der = B.derived_stages(H2, True)
        for k, v in dict(S1=B.a1_stage(X[:8], FC[:8]), S2=H1, S3=H2, **der).items():
            assert np.isfinite(v).all(), f"{arm}/{k} non-finite"
    check("finiteness_all_arms", True, "6 arms x 6 stages x 8 subjects")

    # (5) determinism: same seed -> bitwise-identical stage tensors, CPU and GPU
    X = B.arm_features("B", M1B, FC)
    outs = {}
    for device in (["cpu", "cuda"] if dev == "cuda" else ["cpu"]):
        pair = []
        for rep in range(2):
            enc = B.S12BEncoder(93, 32, "bn", True, B.BASE)
            H1, H2 = B.extract_stages(enc, X[:16], FC[:16], device,
                                      bn_train_idx=np.arange(8))
            pair.append((H1.copy(), H2.copy()))
        same = all(np.array_equal(pair[0][i], pair[1][i]) for i in (0, 1))
        check(f"determinism_{device}", same, "2 runs bitwise equal (incl. BN calibration)")
        outs[device] = pair[0]

    # (6) save/reload bitwise
    enc = B.S12BEncoder(93, 64, "ln", False, B.BASE + 1)
    H1a, H2a = B.extract_stages(enc, X[:16], FC[:16], "cpu")
    pth = B.S12B + "cache/gate2_enc.pt"
    torch.save(enc.state_dict(), pth)
    enc2 = B.S12BEncoder(93, 64, "ln", False, B.BASE + 3)   # different init, then load
    enc2.load_state_dict(torch.load(pth, weights_only=True))
    H1b, H2b = B.extract_stages(enc2, X[:16], FC[:16], "cpu")
    check("save_reload_bitwise",
          np.array_equal(H1a, H1b) and np.array_equal(H2a, H2b), "")

    # (7) production parity: replica with copied weights reproduces TUEncoder
    tu = TUEncoder(num_dataset_features=93, emb_dim=32, num_gc_layers=2,
                   drop_ratio=0.3, pooling_type="standard", is_infograph=False,
                   normalize_nodes=True, message_relu=True, post_bn_relu=True)
    tu.eval()
    rep = B.S12BEncoder(93, 32, "bn", True, B.BASE)
    sd = tu.state_dict()
    rep.load_state_dict({k.replace("convs.0.", "conv0.").replace("convs.1.", "conv1.")
                          .replace("bns.0.", "n0.").replace("bns.1.", "n1."): v
                         for k, v in sd.items()})
    rep.eval()
    xb, eib, ewb, bb = B.make_batch(X, FC, range(16), "cpu")
    with torch.no_grad():
        xpool, node = tu(bb, xb, eib, None, ewb)
        h1r, h2r = rep.stages(xb, eib, ewb)
    der = B.derived_stages(h2r.numpy().reshape(16, 90, 32), True)
    e_node = float(np.abs(der["S4"].reshape(-1, 32) - node.numpy()).max())
    e_pool = float(np.abs(der["S5"] - xpool.numpy()).max())
    check("production_parity", e_node < 1e-5 and e_pool < 1e-5,
          f"node_err={e_node:.2e} pool_err={e_pool:.2e}")

    md = ["# S12B GATE 2 — FORWARD CORRECTNESS: **PASS** (all checks asserted)", ""]
    md += [f"- [{'PASS' if ok else 'FAIL'}] {n} {det}" for n, ok, det in R]
    md += ["", "Self-loop accounting (fork, kept as production): edge set includes"
           " (i,i) with FC[i,i]=1 and WGINConv adds (1+eps)*x_r with eps=0 ->"
           " own node features enter the aggregation with total coefficient 2."
           " A1 := X + FC^T@X (diag included) matches conv.propagate to f32 tolerance.",
           f"- device: {dev}; cudnn.deterministic=True",
           f"- wall {time.time()-t0:.1f}s"]
    open(B.S12B + "GATE2_SMOKE.md", "w").write("\n".join(md) + "\n")
    B.atomic_json(dict(checks=[(n, ok, det) for n, ok, det in R],
                       provenance=B.provenance()), B.S12B + "out/GATE2.json")
    print("GATE2 PASS all", len(R), "checks", flush=True)

def arm_X(arm, d):
    return B.arm_features(arm, d["M1B"], d["FC"])

if __name__ == "__main__":
    main()
