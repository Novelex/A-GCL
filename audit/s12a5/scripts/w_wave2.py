"""S12A5 Wave-2 job: TRANSDUCTIVE A-GCL (corrected-C), one unit = seed_idx.
SSL on all 954 (no labels), fixed 200 epochs. Downstream probes fold-safe (S11 harness,
frozen folds) on h/z/flatten at epoch 0 and epoch 200. Never mixed with inductive."""
import sys, os, json, time, socket, hashlib, numpy as np, torch
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a5/scripts"); import s12a5_core as M
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a4/scripts"); import s12a4_core as M4
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
import s11_core as K, s7_core as C7

SKIP = ("running_mean", "running_var", "num_batches_tracked")

def run(sidx, device):
    seed = M.BASE + sidx; tag = f"trans_s{sidx}"
    fp = f"{M.S12A5}feat/{tag}.npz"
    if os.path.exists(fp): print("skip", tag); return
    t0 = time.time()
    df, X_fc, y, ids, gh = A1.load_gate()
    FC, Xold, stats = A1.load_tensors(df)
    I90 = np.eye(90, dtype=np.float32)
    Xid = np.concatenate([Xold, np.repeat(I90[None], 954, 0)], axis=2)
    torch.manual_seed(seed)
    m0 = M4.ROIModel(seed, with_head=False).to(device)
    H0, Z0, ND0 = M4.extract_all(m0, Xid, FC, device)
    init_sd = {k: v.cpu() for k, v in m0.state_dict().items()}
    model, view, curves = M.train_transductive(seed, Xid, FC, device, epochs=200,
                                               log=tag)
    H, Z, ND = M4.extract_all(model, Xid, FC, device)
    fin_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    mvi = [ (fin_sd[k].float()-init_sd[k].float()).flatten() for k in fin_sd
            if k.startswith("gim.encoder.") and not any(s in k for s in SKIP)
            and torch.is_floating_point(fin_sd[k]) ]
    ini = [ init_sd[k].float().flatten() for k in fin_sd
            if k.startswith("gim.encoder.") and not any(s in k for s in SKIP)
            and torch.is_floating_point(fin_sd[k]) ]
    enc_mv = float(torch.cat(mvi).norm()/torch.cat(ini).norm())
    folds = K.folds_ordinary(); loso = K.folds_loso(y)
    res = {}
    for name, X in (("h", H), ("z", Z), ("h0", H0), ("z0", Z0)):
        d, _ = K.probe_pipe(X.astype(np.float64), y, folds, [])
        res[name] = d["auc"]
    dl, _ = K.probe_pipe(H.astype(np.float64), y, loso, [])
    res["h_loso"] = dl["auc"]
    dfl, _ = K.probe_pipe(ND.astype(np.float64), y, folds, [])
    res["flat"] = dfl["auc"]
    dfl0, _ = K.probe_pipe(ND0.astype(np.float64), y, folds, [])
    res["flat0"] = dfl0["auc"]
    ck = f"{M.S12A5}ckpt/{tag}.pt"
    torch.save(model.state_dict(), ck + ".tmp"); os.replace(ck + ".tmp", ck)
    rec = dict(unit=tag, mode="TRANSDUCTIVE", seed=seed, res=res,
               encoder_movement_trainable=enc_mv,
               keep_mu_first=curves[0]["keep_mu"], keep_mu_last=curves[-1]["keep_mu"],
               infonce_last=curves[-1]["infonce"], cr_last=curves[-1]["cr"],
               view_loss_last=curves[-1]["view_loss"],
               ckpt_sha=hashlib.sha256(open(ck, "rb").read()).hexdigest()[:16],
               hostname=socket.gethostname(),
               gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               wall_s=round(time.time() - t0, 1), git="c0dd9f2",
               cfg=dict(contract="corrected-C (S8) on S12A4 ROI stack", epochs=200,
                        lr=M4.LR, batch=M4.BATCH, wd=0.0))
    jp = f"{M.S12A5}jobs/{tag}.json"
    json.dump(dict(rec=rec, curve=curves,
                   provenance=C7.provenance({"unit": tag})),
              open(jp + ".tmp", "w"), indent=1, default=str)
    assert json.load(open(jp + ".tmp"))["rec"]; os.replace(jp + ".tmp", jp)
    tmp = fp + ".tmp.npz"
    np.savez_compressed(tmp[:-4], h=H, z=Z, h0=H0, z0=Z0,
                        flat=ND.astype(np.float32),
                        flat0=ND0.astype(np.float32), y=y)
    zz = np.load(tmp); assert np.isfinite(zz["h"]).all(); os.replace(tmp, fp)
    print("DONE", tag, json.dumps(res), f"{rec['wall_s']}s", flush=True)

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run(int(sys.argv[1]), dev)
