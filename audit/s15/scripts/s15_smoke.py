"""S15 SMOKE SUITE (array index 0). S1-S8. Exits non-zero on any BLOCKING failure
so the dependency cancels the wave cleanly instead of burning a night."""
import sys, os, json, time, math, socket, copy, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/users/3171356m/A-GCL/audit/s15/scripts")
import s15_data as DAT, s15_models as MO, s15_train as TR, s15_grid as G
import s15_worker as W
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
sys.path.insert(0, "/users/3171356m/A-GCL")
from unsupervised.convs.wgin_conv import WGINConv
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

S15 = DAT.S15
OUT, BLOCKING_FAIL = [], []
def rec(name, ok, detail, blocking=True):
    OUT.append((name, bool(ok), detail, blocking))
    print(("PASS " if ok else "FAIL ") + name + " | " + detail, flush=True)
    if blocking and not ok: BLOCKING_FAIL.append(name)

def main():
    t0 = time.time()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    # ---------------- S1 GATE-C ----------------
    d, man = DAT.load(where="smoke")
    FC, ALFF, y = d["FC"], d["ALFF"], d["y"].astype(np.int64)
    folds_lab = DAT.folds(d, "lab"); folds_site = DAT.folds(d, "site")
    folds_loso = DAT.folds(d, "loso"); allf = DAT.all_folds(d)
    rec("S1_gate_c", True,
        f"954 subjects, ASD {int((y==1).sum())}/NC {int((y==0).sum())}, "
        f"FC sym {man['fc_sym_max']:.1e}, diag dev {man['fc_diag_dev']:.1e}, "
        f"|FC|max {man['fc_absmax']:.6f}, folds lab {len(folds_lab)}/site "
        f"{len(folds_site)}/loso {len(folds_loso)} = {len(allf)}; all S11 hashes match")
    # ---------------- S2 INSTRUMENT ----------------
    Xfc, _y, _ids, _m = K.load_Xfc()
    ordf = [(tr, te) for _, tr, te in folds_lab]
    losf = [(tr, te) for _, tr, te in folds_loso]
    r1, _ = K.probe_pipe(Xfc, y, ordf, [])
    r1l, _ = K.probe_pipe(Xfc, y, losf, [])
    yp = np.random.default_rng(DAT.BASE).permutation(y)
    r3, _ = K.probe_pipe(Xfc, yp, ordf, [])
    r4, _ = K.probe_pipe(ALFF.reshape(954, -1).astype(np.float64), y, ordf, [])
    ok2 = abs(r1["auc"] - 0.7565) <= 0.015 and abs(r1l["auc"] - 0.7432) <= 0.015
    ok3 = abs(r3["auc"] - 0.50) <= 0.03
    rec("S2_instrument", ok2 and ok3,
        f"R1/R2 F-LAB {r1['auc']:.4f} (need 0.7565+-0.015), LOSO {r1l['auc']:.4f} "
        f"(need 0.7432+-0.015); R3 permuted {r3['auc']:.4f} (need 0.50+-0.03); "
        f"R4 ALFF floor {r4['auc']:.4f}")
    CEIL = r1["auc"]
    # ---------------- S3 CAPACITY SWEEP (recorded, not gating) ----------------
    tr0, te0 = folds_lab[0][1], folds_lab[0][2]
    Xfc64 = Xfc.astype(np.float64)
    def fc_recon_r2(R):
        sc = StandardScaler().fit(R[tr0])
        rr = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(sc.transform(R[tr0]), Xfc64[tr0])
        return float(r2_score(Xfc64[te0], rr.predict(sc.transform(R[te0])),
                              multioutput="variance_weighted"))
    cap = []
    Xb = W.build_X("fcrow+alff", FC, ALFF, tr0)[0]
    for Kc in (2, 4, 8, 16, 32, 64):
        for H in (64, 128, 256):
            if H < Xb.shape[2]: continue                    # H >= D asserted
            rs = []
            for s in G.SEEDS:
                m = MO.BNTR(Xb.shape[2], K_clusters=Kc, H=H, seed=s); m.eval()
                assert not m.E.requires_grad and "E" in m.state_dict()
                with torch.no_grad():
                    R, _ = TR.extract(m, Xb, FC, np.arange(954), False)
                rs.append((fc_recon_r2(R), m._last_entropy, MO.n_trainable(m), R.shape[1]))
            cap.append(dict(arch="BNT", K=Kc, H=H, repr_dim=rs[0][3],
                            n_trainable=rs[0][2],
                            fc_recon_r2=float(np.mean([x[0] for x in rs])),
                            entropy=float(np.mean([x[1] for x in rs])),
                            entropy_max=float(np.log(Kc))))
    for hid in (128, 256):
        for ro in ("sum", "roi"):
            rs = []
            for s in G.SEEDS:
                m = MO.WGINR(Xb.shape[2], hidden=hid, seed=s, readout=ro); m.eval()
                with torch.no_grad():
                    R, _ = TR.extract(m, Xb, FC, np.arange(954), True)
                rs.append((fc_recon_r2(R), MO.n_trainable(m), R.shape[1]))
            cap.append(dict(arch="WGIN", hidden=hid, readout=ro, repr_dim=rs[0][2],
                            n_trainable=rs[0][1],
                            fc_recon_r2=float(np.mean([x[0] for x in rs])),
                            entropy=float("nan"), entropy_max=float("nan")))
    # S13 post-mortem: K=2, H=128 exactly
    pm = [c for c in cap if c["arch"] == "BNT" and c.get("K") == 2 and c["H"] == 128]
    rec("S3_capacity_sweep", True,
        f"{len(cap)} configs; S13 POST-MORTEM K=2/H=128 repr_dim "
        f"{pm[0]['repr_dim']} FC-recon R2 {pm[0]['fc_recon_r2']:.4f}; "
        f"best BNT R2 {max(c['fc_recon_r2'] for c in cap if c['arch']=='BNT'):.4f}",
        blocking=False)
    # ---------------- S4 OVERFIT ONE BATCH (BLOCKING, both) ----------------
    ii = np.concatenate([np.where(y == 0)[0][:16], np.where(y == 1)[0][:16]])
    for arch, kh in (("BNT", 32), ("WGIN", 128)):
        Xa = W.build_X(G.ARMS["B2" if arch == "BNT" else "W3"][1], FC, ALFF, tr0)[0]
        m = MO.build_model(arch, Xa.shape[2], DAT.BASE, kh, p=0.0)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
        b = MO.make_batch(Xa, FC, ii, arch == "WGIN")
        t = torch.tensor(y[ii], dtype=torch.float32); lf = nn.BCEWithLogitsLoss()
        m.train()
        for _ in range(500):
            opt.zero_grad(); _, lg = m(b, None); l = lf(lg, t); l.backward(); opt.step()
        m.eval()
        with torch.no_grad(): _, lg = m(b, None)
        auc = float(roc_auc_score(y[ii], lg.numpy())); loss = float(lf(lg, t))
        rec(f"S4_overfit_{arch}", auc == 1.0 and loss < 0.01,
            f"train AUC {auc:.4f} (need 1.000), loss {loss:.6f} (need <0.01), "
            f"500 steps, dropout 0, wd 0, no clip")
    # ---------------- S5 BNT CORRECTNESS ----------------
    Xb2 = W.build_X("fcrow+alff", FC, ALFF, tr0)[0]; D = Xb2.shape[2]
    m = MO.BNTR(D, K_clusters=32, H=128, seed=DAT.BASE); m.eval()
    E = m.E; err = float((E @ E.t() - torch.eye(32)).abs().max())
    rec("S5a_orthonormal_buffer", err < 1e-5 and "E" in m.state_dict()
        and not E.requires_grad
        and not any(n == "E" for n, _ in m.named_parameters()),
        f"max|EE^T-I| {err:.2e}; E in state_dict, buffer, requires_grad False")
    b8 = MO.make_batch(Xb2, FC, range(8), False)
    with torch.no_grad():
        ZL = m.encode(b8.X, keep_attn=True); ZG, P = m.ocread(ZL); r = m.repr_of(b8)
    rec("S5b_soft_assignment",
        float((P.sum(-1) - 1).abs().max()) < 1e-6 and tuple(ZG.shape) == (8, 32, 128)
        and tuple(r.shape) == (8, 4096),
        f"P sums to 1 ({float((P.sum(-1)-1).abs().max()):.1e}); Z_G {tuple(ZG.shape)}; "
        f"repr {tuple(r.shape)}; entropy {m._last_entropy:.4f} (max {math.log(32):.4f})")
    A0 = m.blocks[0].attn.last_attn
    rec("S5c_attention", float((A0.sum(-1) - 1).abs().max()) < 1e-6
        and tuple(A0.shape) == (8, 4, 90, 90) and len(m.blocks) == 2,
        f"rows sum to 1 ({float((A0.sum(-1)-1).abs().max()):.1e}); shape {tuple(A0.shape)}; "
        f"2 layers x 4 heads; FC NEVER injected into scores")
    rng = np.random.default_rng(DAT.BASE); pi = rng.permutation(90)
    prof_ok = all(np.array_equal(Xb2[s, i, :90], FC[s, i])
                  for s in rng.integers(0, 954, 8) for i in rng.integers(0, 90, 8))
    symm = float(np.abs(FC - FC.transpose(0, 2, 1)).max())
    m2 = copy.deepcopy(m)
    with torch.no_grad():
        idx = np.concatenate([pi, np.arange(90, D)])
        m2.inp.weight.copy_(m.inp.weight.clone()[:, idx])
    Xp = np.concatenate([FC[:8][:, pi][:, :, pi], W.alff_z(ALFF, tr0)[:8][:, pi]], 2)
    with torch.no_grad():
        Z1 = m.encode(MO.make_batch(Xb2[:8], FC, range(8), False).X)
        Z2 = m2.encode(MO.make_batch(Xp, FC, range(8), False).X)
    eq = float((Z2 - Z1[:, pi]).abs().max())
    rec("S5d_roi_equivariance", prof_ok and symm == 0.0 and eq < 1e-4,
        f"profile==FC row bitwise ({prof_ok}); FC symmetry EXACTLY {symm:.1e} so a "
        f"transposed profile is provably a no-op; permuting data AND inp.weight cols "
        f"-> max|Z_L(perm)-perm(Z_L)| {eq:.2e} <1e-4 (guards a [B,D,90] axis swap)")
    m1 = MO.BNTR(D, K_clusters=1, H=128, seed=DAT.BASE); m1.eval()
    with torch.no_grad():
        ZL1 = m1.encode(b8.X); ZG1, P1 = m1.ocread(ZL1)
    dm = float((ZG1.squeeze(1) / 90.0 - ZL1.mean(1)).abs().max())
    rec("S5e_K1_equals_mean", dm < 1e-5 and float((P1 - 1).abs().max()) < 1e-6,
        f"max|Z_G/90 - mean| {dm:.2e} (declared factor 90; P==1)")
    m3 = MO.BNTR(D, K_clusters=32, H=128, seed=DAT.BASE); m3.train()
    _, lg = m3(MO.make_batch(Xb2, FC, range(32), False), None)
    nn.BCEWithLogitsLoss()(lg, torch.tensor(y[:32], dtype=torch.float32)).backward()
    dead = [n for n, p in m3.named_parameters() if p.grad is None or float(p.grad.norm()) == 0]
    rec("S5f_gradient_flow", not dead and m3.E.grad is None,
        f"{len(dead)} dead trainable params; E.grad is None ({m3.E.grad is None})")
    ra, la = TR.extract(MO.BNTR(D, 32, 128, seed=DAT.BASE), Xb2, FC, range(16), False)
    rb, lb = TR.extract(MO.BNTR(D, 32, 128, seed=DAT.BASE), Xb2, FC, range(16), False)
    pth = S15 + "cache/smoke_bnt.pt"
    mm = MO.BNTR(D, 32, 128, seed=DAT.BASE + 5); torch.save(mm.state_dict(), pth)
    r1_, l1_ = TR.extract(mm, Xb2, FC, range(16), False)
    m4 = MO.BNTR(D, 32, 128, seed=DAT.BASE + 9)
    m4.load_state_dict(torch.load(pth, weights_only=True))
    r2_, l2_ = TR.extract(m4, Xb2, FC, range(16), False)
    rec("S5g_determinism_reload",
        np.array_equal(ra, rb) and np.array_equal(la, lb)
        and np.array_equal(r1_, r2_) and np.array_equal(l1_, l2_),
        "same seed bitwise identical; checkpoint reload bitwise identical (CPU)")
    # ---------------- S6 WGIN CORRECTNESS ----------------
    for dt, tol in ((torch.float64, 1e-6), (torch.float32, 1e-4)):
        x = torch.tensor([[1., 2.], [-3., .5], [0., -1.], [2., 2.]], dtype=dt)
        Ed = [(0,0,1.),(1,1,1.),(2,2,1.),(3,3,1.),(0,1,.5),(1,0,.5),
              (2,3,-.8),(3,2,-.8),(0,2,2.),(2,0,2.)]
        ei = torch.tensor([[e[0] for e in Ed],[e[1] for e in Ed]], dtype=torch.long)
        ew = torch.tensor([e[2] for e in Ed], dtype=dt)
        conv = WGINConv(nn.Identity(), message_relu=True).to(dt)
        out = conv(x, ei, ew)
        hand = torch.zeros_like(x)
        for j in range(4):
            s = torch.zeros(2, dtype=dt)
            for (a, bb, w) in Ed:
                if bb == j: s = s + w * torch.clamp(x[a], min=0)
            hand[j] = s + x[j]
        e = float((out - hand).abs().max())
        rec(f"S6a_wgin_hand_{str(dt)[-9:]}", e < tol,
            f"max err {e:.2e} < {tol:.0e} (negative edge weight and sign-flip node included)")
    Xw = W.build_X("fcrow+alff", FC, ALFF, tr0)[0]
    rng2 = np.random.default_rng(DAT.BASE + 1)
    okb = all(np.array_equal(Xw[s, i, :90], FC[s, i])
              for s in rng2.integers(0, 954, 8) for i in rng2.integers(0, 90, 8))
    ei0 = MO.edge_struct(); nself = int((ei0[0] == ei0[1]).sum())
    rec("S6b_wgin_features_selfloop", okb and nself == 90,
        f"x[i,:90]==FC[i,:] bitwise (8x8); LOGGED FORK: {nself} self-loops with "
        f"FC[i,i]=1.0 AND (1+eps)x_r -> own features counted TWICE, kept as S12A5 ran it")
    # ---------------- S7 ADAPTIVE-CLIP CALIBRATION (recorded) ----------------
    surv = []
    for arch, kh, spec in (("BNT", 32, "fcrow+alff"), ("WGIN", 128, "fcrow+alff")):
        Xs = W.build_X(spec, FC, ALFF, tr0)[0]
        m = MO.build_model(arch, Xs.shape[2], DAT.BASE, kh)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-3)
        gen = torch.Generator().manual_seed(DAT.BASE); norms = []
        m.train()
        for ep in range(20):
            perm = torch.randperm(len(tr0), generator=gen).numpy()
            for lo in range(0, len(perm), TR.BATCH):
                ii2 = np.asarray(tr0)[perm[lo:lo + TR.BATCH]]
                if len(ii2) < 2: continue
                opt.zero_grad()
                _, lg = m(MO.make_batch(Xs, FC, ii2, arch == "WGIN"), None)
                TR.loss_bce(lg, torch.tensor(y, dtype=torch.float32)[ii2]).backward()
                norms.append(float(torch.nn.utils.clip_grad_norm_(
                    m.parameters(), float("inf"))))          # measure, NO clipping
                opt.step()
        q = {f"p{p}": float(np.percentile(norms, p)) for p in (50, 90, 95, 99)}
        q["max"] = float(np.max(norms)); q["arch"] = arch; q["n_steps"] = len(norms)
        q["frac_above_1.0"] = float(np.mean(np.array(norms) > 1.0))
        surv.append(q)
    rec("S7_clip_calibration", True,
        " | ".join(f"{q['arch']}: p50 {q['p50']:.3f} p90 {q['p90']:.3f} p95 {q['p95']:.3f} "
                   f"p99 {q['p99']:.3f} max {q['max']:.2f}, "
                   f"{100*q['frac_above_1.0']:.0f}% above S13's fixed 1.0" for q in surv),
        blocking=False)
    # ---------------- S8 determinism env ----------------
    rec("S8_determinism_env", True,
        f"use_deterministic_algorithms(True); OMP_NUM_THREADS="
        f"{os.environ.get('OMP_NUM_THREADS')}; torch threads {torch.get_num_threads()}")
    # ---------------- emit ----------------
    md = ["# S15 SMOKE SUITE", "",
          f"host {socket.gethostname()} | {time.strftime('%F %T')} | "
          f"wall {time.time()-t0:.0f}s | CEILING_PROBE {CEIL:.4f}", "",
          "## FROZEN REFERENCE LINES (never recomputed)",
          "| reference | F-LAB | LOSO |", "|---|---|---|",
          "| LinearSVC 4005 FC edges | 0.7565 | 0.7432 |",
          "| ridge logistic | 0.7561 | 0.7406 |",
          "| BNT S13 winner | 0.6583 | 0.6619 |",
          "| WGIN S12A5 arm A | 0.6307 | — |",
          "| RANDOM WGIN S12A3 (untrained watermark) | 0.6539 | — |", "",
          "## GATES"]
    md += [f"- [{'PASS' if ok else 'FAIL'}]{'' if blk else ' (recorded, not gating)'} "
           f"**{n}** — {det}" for n, ok, det, blk in OUT]
    md += ["", "## S3 CAPACITY SWEEP (forward-only, random init, 3 seeds)", "",
           "| arch | K/hidden | H/readout | repr_dim | trainable | FC-recon R2 | entropy |",
           "|---|---|---|---|---|---|---|"]
    for c in cap:
        kk = c.get("K", c.get("hidden")); hh = c.get("H", c.get("readout"))
        ent = "—" if not np.isfinite(c["entropy"]) else \
              f"{c['entropy']:.3f}/{c['entropy_max']:.3f}"
        md.append(f"| {c['arch']} | {kk} | {hh} | {c['repr_dim']} | "
                  f"{c['n_trainable']:,} | {c['fc_recon_r2']:.4f} | {ent} |")
    md += ["", "## S7 GRADIENT SURVEY (clipping OFF) — the S13 post-mortem", "",
           "| arch | p50 | p90 | p95 | p99 | max | % above S13's fixed 1.0 |",
           "|---|---|---|---|---|---|---|"]
    for q in surv:
        md.append(f"| {q['arch']} | {q['p50']:.3f} | {q['p90']:.3f} | {q['p95']:.3f} | "
                  f"{q['p99']:.3f} | {q['max']:.2f} | {100*q['frac_above_1.0']:.0f}% |")
    open(S15 + "SMOKE.md.tmp", "w").write("\n".join(md) + "\n")
    os.replace(S15 + "SMOKE.md.tmp", S15 + "SMOKE.md")
    W.atomic_json(dict(checks=[(n, ok, dd, blk) for n, ok, dd, blk in OUT],
                       capacity=cap, grad_survey=surv, ceiling_probe=CEIL,
                       blocking_failures=BLOCKING_FAIL), S15 + "out/SMOKE.json")
    if BLOCKING_FAIL:
        print("SMOKE BLOCKING FAILURES:", BLOCKING_FAIL, flush=True); sys.exit(1)
    print(f"SMOKE_ALL_PASS ({len(OUT)} checks, {time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
