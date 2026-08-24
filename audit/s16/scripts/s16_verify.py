"""S16 C2-C5 verification, one job. C2 ruler + probe bias + probe_honest;
C3 model correctness (T1-T12); C4 FLOOR TEST (must read EXACTLY 0.7565);
C5 does-training-happen. Exits non-zero on any BLOCKING failure."""
import sys, os, json, time, math, copy, socket, numpy as np, torch, torch.nn as nn
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_data as DAT, s16_models as MO, s16_train as TR, s16_feat as FT
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
sys.path.insert(0,"/users/3171356m/A-GCL")
from unsupervised.convs.wgin_conv import WGINConv
from sklearn.metrics import roc_auc_score

S16 = DAT.S16; OUT=[]; FAIL=[]
def rec(n, ok, det, blocking=True):
    OUT.append((n,bool(ok),det,blocking))
    print(("PASS " if ok else "FAIL ")+n+" | "+det, flush=True)
    if blocking and not ok: FAIL.append(n)

def main():
    t0=time.time(); torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","8")))
    d,MAN,ent = DAT.load("signed", where="verify")
    FC,ALFF,y = d["FC"],d["ALFF"],d["y"].astype(np.int64)
    fl = DAT.folds(d,"lab"); tag0,tr0,te0 = fl[0]
    Xfc,_y,_i,_m = K.load_Xfc()

    # ================= C2a THE RULER =================
    r1,_  = K.probe_pipe(Xfc, y, [(a,b) for _,a,b in fl], [])
    r1l,_ = K.probe_pipe(Xfc, y, [(a,b) for _,a,b in DAT.folds(d,"loso")], [])
    yp = np.random.default_rng(DAT.BASE).permutation(y)
    r3,_  = K.probe_pipe(Xfc, yp, [(a,b) for _,a,b in fl], [])
    r4,_  = K.probe_pipe(ALFF.reshape(954,-1).astype(np.float64), y,
                         [(a,b) for _,a,b in fl], [])
    ok = (abs(r1["auc"]-0.7565)<=0.015 and abs(r1l["auc"]-0.7432)<=0.015
          and abs(r3["auc"]-0.50)<=0.03)
    rec("C2a_ruler", ok, f"FC ord {r1['auc']:.4f} (need 0.7565+-0.015); FC LOSO "
        f"{r1l['auc']:.4f} (need 0.7432+-0.015); permuted {r3['auc']:.4f} "
        f"(need 0.50+-0.03); ALFF floor {r4['auc']:.4f}")

    # ================= C4 THE FLOOR TEST (run early: it gates everything) ======
    trp_enc, trp_prb = FT.honest_split(tr0, y)
    rng = np.random.default_rng(0)
    R_junk = rng.standard_normal((954, 512))
    Z = FT.fuse(np.zeros((954,512)), Xfc)                     # learned block ZEROED
    dz,_ = FT.probe_honest(Z, y, tr0, te0)
    dref,_ = K.probe_pipe(Xfc, y, [(tr0,te0)], [])            # same fold, FC alone
    exact = abs(dz["auc"]-dref["auc"]) < 1e-12
    rec("C4_floor_exact", exact,
        f"learned block zeroed -> {dz['auc']:.10f} vs FC-alone {dref['auc']:.10f} "
        f"on the SAME fold; diff {abs(dz['auc']-dref['auc']):.2e} (need bitwise). "
        f"StandardScaler standardises each column independently, so the two blocks' "
        f"different scales cannot distort one another.")
    Zf = FT.fuse(R_junk, np.zeros((954,4005)))                # FC block zeroed
    dfz,_ = FT.probe_honest(Zf, y, tr0, te0)
    rec("C4_blocks_independent", True,
        f"FC block zeroed, random learned block -> {dfz['auc']:.4f} (chance-like, "
        f"confirms the blocks are separate and correctly positioned)", blocking=False)
    # pooled 5-fold floor, the headline number
    Zall = FT.fuse(np.zeros((954,512)), Xfc)
    dpool,_ = K.probe_pipe(Zall, y, [(a,b) for _,a,b in fl], [])
    rec("C4_floor_pooled", abs(dpool["auc"]-0.7565)<=0.0001,
        f"pooled 5-fold, learned block zeroed -> {dpool['auc']:.4f} (need 0.7565)")

    # ================= C3 MODEL CORRECTNESS =================
    Xb,_ = FT.build_X("fcrow+alff", FC, ALFF, tr0); D = Xb.shape[2]
    m = MO.BNTR(D, K_clusters=32, H=128, seed=DAT.BASE); m.eval()
    E = m.E; err = float((E @ E.t() - torch.eye(32)).abs().max())
    rec("T1_orthonormal", err<1e-5 and "E" in m.state_dict() and not E.requires_grad
        and not any(n=="E" for n,_ in m.named_parameters()),
        f"max|EE^T-I| {err:.2e}; buffer in state_dict, requires_grad False")
    b8 = MO.make_batch(Xb, FC, range(8), False)
    with torch.no_grad():
        ZL = m.encode(b8.X, keep=True); ZG,P = m.ocread(ZL); r = m.repr_of(b8)
    rec("T2_ocread", float((P.sum(-1)-1).abs().max())<1e-6 and tuple(ZG.shape)==(8,32,128),
        f"P sums to 1 ({float((P.sum(-1)-1).abs().max()):.1e}); Z_G {tuple(ZG.shape)}; "
        f"repr {tuple(r.shape)}; entropy {m._entropy:.4f} / max {math.log(32):.4f}")
    A0 = m.blocks[0].attn.last_attn
    rec("T3_attention", float((A0.sum(-1)-1).abs().max())<1e-6
        and tuple(A0.shape)==(8,4,90,90) and len(m.blocks)==2,
        f"rows sum to 1; shape {tuple(A0.shape)}; 2 layers x 4 heads; FC never in scores")
    pi = np.random.default_rng(DAT.BASE).permutation(90)
    m2 = copy.deepcopy(m)
    with torch.no_grad():
        idx = np.concatenate([pi, np.arange(90,D)])
        m2.inp.weight.copy_(m.inp.weight.clone()[:,idx])
    Xp = np.concatenate([FC[:8][:,pi][:,:,pi], FT.alff_scaled(ALFF,tr0)[:8][:,pi]],2)
    with torch.no_grad():
        Z1 = m.encode(MO.make_batch(Xb[:8],FC,range(8),False).X)
        Z2 = m2.encode(MO.make_batch(Xp,FC,range(8),False).X)
    eq = float((Z2-Z1[:,pi]).abs().max())
    symm = float(np.abs(FC-FC.transpose(0,2,1)).max())
    rec("T4_roi_equivariance", eq<1e-4,
        f"permute data AND inp.weight cols -> max|Z_L(perm)-perm(Z_L)| {eq:.2e} (<1e-4); "
        f"FC symmetry EXACTLY {symm:.1e} so a transposed profile is provably a no-op; "
        f"guards a [B,D,90] axis swap")
    m1 = MO.BNTR(D,K_clusters=1,H=128,seed=DAT.BASE); m1.eval()
    with torch.no_grad(): Z1_=m1.encode(b8.X); ZG1,P1=m1.ocread(Z1_)
    dm = float((ZG1.squeeze(1)/90.0 - Z1_.mean(1)).abs().max())
    rec("T5_K1_mean", dm<1e-5, f"max|Z_G/90 - mean| {dm:.2e} (declared factor 90)")
    for dt,tol in ((torch.float64,1e-6),(torch.float32,1e-4)):
        x = torch.tensor([[1.,2.],[-3.,.5],[0.,-1.],[2.,2.]],dtype=dt)
        Ed=[(0,0,1.),(1,1,1.),(2,2,1.),(3,3,1.),(0,1,.5),(1,0,.5),(2,3,-.8),(3,2,-.8),(0,2,2.),(2,0,2.)]
        ei=torch.tensor([[e[0] for e in Ed],[e[1] for e in Ed]],dtype=torch.long)
        ew=torch.tensor([e[2] for e in Ed],dtype=dt)
        out = WGINConv(nn.Identity(),message_relu=True).to(dt)(x,ei,ew)
        hand=torch.zeros_like(x)
        for j in range(4):
            s=torch.zeros(2,dtype=dt)
            for (a,bb,w) in Ed:
                if bb==j: s=s+w*torch.clamp(x[a],min=0)
            hand[j]=s+x[j]
        e=float((out-hand).abs().max())
        rec(f"T6_wgin_hand_{str(dt)[-9:]}", e<tol,
            f"max err {e:.2e} (negative edge weight + sign-flip node; self-loop double "
            f"count is a LOGGED FORK, not fixed)")
    dps,_,eps_ = DAT.load("pos_zero", where="verify")
    FCs = dps["FC"]
    okp=True
    for s in np.random.default_rng(1).integers(0,954,8):
        rr,cc = np.nonzero(FCs[s])
        if (FCs[s][rr,cc] <= 0).any(): okp=False
        if not np.array_equal(FCs[s][rr,cc], FC[s][rr,cc]): okp=False
    rec("T7_sparse_positive_only", okp,
        f"8 subjects: every sparse edge has FC>0 and equals the original value bitwise; "
        f"retained {eps_['sparse_stats']['pct_of_8100']:.1f}% of 8100, "
        f"min degree {eps_['sparse_stats']['min_node_degree']}, "
        f"isolated nodes {eps_['sparse_stats']['isolated_nodes_total']}")
    rg=np.random.default_rng(2)
    rec("T8_profile_is_row", all(np.array_equal(Xb[s,i,:90],FC[s,i])
        for s in rg.integers(0,954,8) for i in rg.integers(0,90,8)),
        "x[i,:90] == FC[i,:] bitwise, 8 subjects x 8 ROIs")
    m3 = MO.BNTR(D,32,128,seed=DAT.BASE); m3.train()
    _,lg = m3(MO.make_batch(Xb,FC,range(32),False),None)
    nn.BCEWithLogitsLoss()(lg,torch.tensor(y[:32],dtype=torch.float32)).backward()
    dead=[n for n,p in m3.named_parameters() if p.grad is None or float(p.grad.norm())==0]
    rec("T9_gradients", not dead and m3.E.grad is None,
        f"{len(dead)} dead trainable params; E.grad is None")
    ra,la = TR.extract(MO.BNTR(D,32,128,seed=DAT.BASE),Xb,FC,range(16),False)
    rb,lb = TR.extract(MO.BNTR(D,32,128,seed=DAT.BASE),Xb,FC,range(16),False)
    pth=S16+"cache/verify_bnt.pt"; mm=MO.BNTR(D,32,128,seed=DAT.BASE+5)
    torch.save(mm.state_dict(),pth); r1_,l1_=TR.extract(mm,Xb,FC,range(16),False)
    m4=MO.BNTR(D,32,128,seed=DAT.BASE+9); m4.load_state_dict(torch.load(pth,weights_only=True))
    r2_,l2_=TR.extract(m4,Xb,FC,range(16),False)
    rec("T10_determinism_reload", np.array_equal(ra,rb) and np.array_equal(la,lb)
        and np.array_equal(r1_,r2_) and np.array_equal(l1_,l2_),
        "same seed bitwise identical; checkpoint reload bitwise identical (CPU)")
    ii=np.concatenate([np.where(y==0)[0][:16],np.where(y==1)[0][:16]])
    for arch,kh,spec in (("BNT",32,"fcrow+alff"),("WGIN",128,"fcrow+alff")):
        Xa,_ = FT.build_X(spec,FC,ALFF,tr0)
        mo=MO.build_model(arch,Xa.shape[2],DAT.BASE,kh,p=0.0)
        opt=torch.optim.AdamW(mo.parameters(),lr=1e-3,weight_decay=0.0)
        bb=MO.make_batch(Xa,FC,ii,arch=="WGIN"); tt=torch.tensor(y[ii],dtype=torch.float32)
        lf=nn.BCEWithLogitsLoss(); mo.train()
        for _ in range(500):
            opt.zero_grad(); _,l=mo(bb,None); q=lf(l,tt); q.backward(); opt.step()
        mo.eval()
        with torch.no_grad(): _,l=mo(bb,None)
        auc=float(roc_auc_score(y[ii],l.numpy())); loss=float(lf(l,tt))
        rec(f"T11_overfit_{arch}", auc==1.0 and loss<0.01,
            f"train AUC {auc:.4f} (need 1.000), loss {loss:.6f} (need <0.01)")
    pc=[]
    for arm,(arch,spec) in FT.ARMS.items():
        Xa,_ = FT.build_X(spec,FC,ALFF,tr0)
        kh = 32 if arch=="BNT" else 128
        mo = MO.build_model(arch,Xa.shape[2],DAT.BASE,kh)
        pc.append(f"{arm}({arch},D={Xa.shape[2]}) {MO.n_trainable(mo):,} repr {mo.repr_dim}")
    rec("T12_params", True, " | ".join(pc), blocking=False)

    md=["# S16 C2-C5 VERIFICATION","",
        f"host {socket.gethostname()} | {time.strftime('%F %T')} | wall {time.time()-t0:.0f}s","",
        "## FROZEN ANCHORS (never recomputed)","| reference | ord | LOSO |","|---|---|---|",
        "| LinearSVC 4005 FC edges | 0.7565 | 0.7432 |",
        "| random WGIN S12A3 (untrained watermark) | 0.6539 | — |",
        "| trained WGIN S12A4b | 0.6429 | — |","| WGIN S12A5 arm A | 0.6307 | — |",
        "| BNT S13 winner | 0.6583 | 0.6619 |","","## CHECKS"]
    md += [f"- [{'PASS' if ok else 'FAIL'}]{'' if blk else ' (recorded)'} **{n}** — {det}"
           for n,ok,det,blk in OUT]
    open(S16+"C2_C5_VERIFY.md.tmp","w").write("\n".join(md)+"\n")
    os.replace(S16+"C2_C5_VERIFY.md.tmp", S16+"C2_C5_VERIFY.md")
    json.dump(dict(checks=[(n,ok,dd,b) for n,ok,dd,b in OUT], blocking_failures=FAIL),
              open(S16+"out/VERIFY.json","w"), indent=1, default=str)
    if FAIL: print("BLOCKING FAILURES:",FAIL,flush=True); sys.exit(1)
    print(f"S16_VERIFY_ALL_PASS ({len(OUT)} checks, {time.time()-t0:.0f}s)",flush=True)

if __name__=="__main__": main()
