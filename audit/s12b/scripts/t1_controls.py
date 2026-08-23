"""S12B Track 1 winner controls (post-grid): P-lab (labels permuted) and P-roi
(per-subject ROI permutation of features AND FC) on the BEST config's S6 stage,
plus full stage-tensor persistence for top-5 configs + arm A baseline.
Forward-only; runs AFTER consolidate.py has written out/BEST_CONFIG.json."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12b/scripts"); import s12b_core as B

def forward_config(arm, emb, norm, mr, seed, d, perm_roi=None, fold_tr=None, dev="cpu"):
    FC = d["FC"]; M1B = d["M1B"]
    if perm_roi is not None:
        M1B = np.stack([M1B[i][perm_roi[i]] for i in range(954)])
        FC = np.stack([FC[i][perm_roi[i]][:, perm_roi[i]] for i in range(954)])
    X = B.arm_features(arm, M1B, FC)
    enc = B.S12BEncoder(X.shape[2], emb, norm, mr, seed)
    H1, H2 = B.extract_stages(enc, X, FC, dev,
                              bn_train_idx=fold_tr if norm == "bn" else None)
    return X, FC, H1, H2

def probe_stage6(arm, emb, norm, mr, nn, y, d, folds, perm_roi=None, dev="cpu"):
    """Probe S6 (= flatten of S4 if nn else H2) across ordinary folds, I1 only.
    For bn: per-fold calibration exactly as the grid."""
    ev = B.StageEval(y, XFC=None, conf=None, keep_proj=False)
    seed = B.BASE                                        # seed 0 of the grid
    if norm == "bn":
        for t, tr, te in folds:
            _, _, H1, H2 = forward_config(arm, emb, norm, mr, seed, d, perm_roi, tr, dev)
            M = B.derived_stages(H2, nn)
            X6 = M["S4"].reshape(954, -1) if nn else H2.reshape(954, -1)
            ev.fold_fit(X6, tr, te, t, njobs=int(os.environ.get("S11_NJOBS", "4")))
    else:
        _, _, H1, H2 = forward_config(arm, emb, norm, mr, seed, d, perm_roi, None, dev)
        M = B.derived_stages(H2, nn)
        X6 = M["S4"].reshape(954, -1) if nn else H2.reshape(954, -1)
        for t, tr, te in folds:
            ev.fold_fit(X6, tr, te, t, njobs=int(os.environ.get("S11_NJOBS", "4")))
    return ev

def main():
    jp = B.S12B + "out/CONTROLS.json"
    if os.path.exists(jp): print("skip controls"); return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = B.load_all(); y = d["y"]
    folds = [(t, tr, te) for t, tr, te in B.folds_all(y) if t.startswith("o")]
    bc = json.load(open(B.S12B + "out/BEST_CONFIG.json"))["best"]
    arm, emb = bc["arm"], int(bc["emb_dim"])
    norm, mr, nn = bc["norm_type"], bool(bc["message_relu"]), bool(bc["normalize_nodes"])
    t0 = time.time()
    # P-lab: same representation, labels permuted (BN calibration is label-free)
    yp = np.random.default_rng(B.BASE).permutation(y)
    assert yp.sum() == y.sum() and (yp == y).mean() < 1.0
    ev = probe_stage6(arm, emb, norm, mr, nn, yp, d, folds, None, dev)
    plab = ev.finalize()["pooled_ordinary"]["auc"]
    # P-roi: consistent per-subject ROI permutation of features AND FC
    pr = np.stack([np.random.default_rng(B.BASE + 7000 + i).permutation(90)
                   for i in range(954)])
    ev = probe_stage6(arm, emb, norm, mr, nn, y, d, folds, pr, dev)
    proi = ev.finalize()["pooled_ordinary"]["auc"]
    res = dict(best_config=bc, P_lab_auc=float(plab), P_roi_auc=float(proi),
               PASS_P_lab=bool(0.45 <= plab <= 0.55),
               PASS_P_roi=bool(0.45 <= proi <= 0.55),
               wall_s=round(time.time() - t0, 1), provenance=B.provenance())
    B.atomic_json(res, jp)
    print("CONTROLS", json.dumps(res)[:250], flush=True)
    # persist full stage tensors: top-5 by S6 + arm A default baseline
    top = json.load(open(B.S12B + "out/BEST_CONFIG.json"))["top10"][:5]
    keep = [(t["arm"], int(t["emb_dim"]), t["norm_type"], bool(t["message_relu"]))
            for t in top] + [("A", 32, "bn", True)]
    for arm2, emb2, norm2, mr2 in dict.fromkeys(keep):
        fp = f"{B.S12B}feat/tensors_{arm2}_e{emb2}_{norm2}_mr{int(mr2)}_s0.npz"
        if os.path.exists(fp): continue
        tr0 = folds[0][1]
        X, FCu, H1, H2 = forward_config(arm2, emb2, norm2, mr2, B.BASE, d,
                                        None, tr0 if norm2 == "bn" else None, dev)
        tmp = fp + ".tmp.npz"
        np.savez_compressed(tmp[:-4], H0=X.astype(np.float32),
                            A1=B.a1_stage(X, FCu), H1=H1, H2=H2,
                            note=np.array(["bn calibrated on fold o0 train" if norm2 == "bn"
                                           else "fold-independent"]))
        zz = np.load(tmp); assert np.isfinite(zz["H2"]).all(); os.replace(tmp, fp)
        print("tensors saved", fp, flush=True)

if __name__ == "__main__":
    main()
