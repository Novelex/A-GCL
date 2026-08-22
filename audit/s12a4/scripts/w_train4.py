"""S12A4 training job: one unit = (arm, seed_idx). Loops 7 ordinary + all LOSO folds.
Per fold: train (ES on val AUC only) -> best ckpt -> extract h/z/flatten for all 954 ->
in-job S11 probe on h and z for THIS fold -> atomic save. Fully resumable per fold."""
import sys, os, json, time, socket, numpy as np, torch
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a4/scripts"); import s12a4_core as M
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
import s11_core as K, s7_core as C7
from sklearn.metrics import roc_auc_score

UNITS = [(a, s) for a in (1, 2, 3, 4) for s in (0, 1, 2)]

def fold_list(y):
    F = [("o%d" % i, tr, te) for i, (tr, te) in enumerate(K.folds_ordinary())]
    F += [("l%d" % i, tr, te) for i, (tr, te) in enumerate(K.folds_loso(y))]
    return F

def run(arm, sidx, device):
    seed = M.BASE + sidx
    tagj = f"a{arm}_s{sidx}"
    df, X_fc, y, ids, gh = A1.load_gate()
    FC, Xold, stats = A1.load_tensors(df)
    I90 = np.eye(90, dtype=np.float32)
    Xid = np.concatenate([Xold, np.repeat(I90[None], 954, 0)], axis=2)
    for tag, tr, te in fold_list(y):
        fp = f"{M.S12A4}feat/{tagj}_{tag}.npz"
        if os.path.exists(fp):
            print("skip", tagj, tag, flush=True); continue
        t0 = time.time()
        model, curve, info = M.train_fold(arm, seed, np.asarray(tr), Xid, FC, y,
                                          device, log=f"{tagj}/{tag}")
        H, Z, ND = M.extract_all(model, Xid, FC, device)
        head_scores = None
        if arm in (1, 2):
            with torch.no_grad():
                head_scores = model.head(torch.from_numpy(H).to(device)) \
                                   .squeeze(-1).cpu().numpy()
        # in-job S11 probes on this fold (32-d h and z; flatten deferred to winner)
        dh, oh = K.probe_pipe(H.astype(np.float64), y, [(np.asarray(tr), np.asarray(te))], [])
        dz, oz = K.probe_pipe(Z.astype(np.float64), y, [(np.asarray(tr), np.asarray(te))], [])
        # arm4 validity: hard keep rate within 0.75-0.85 every epoch
        valid = True
        if arm == 4:
            kr = [k for k in info["keep_rates"] if k is not None]
            valid = bool(kr) and all(0.75 <= k <= 0.85 for k in kr)
        ck = f"{M.S12A4}ckpt/{tagj}_{tag}.pt"
        torch.save(model.state_dict(), ck + ".tmp"); os.replace(ck + ".tmp", ck)
        rec = dict(unit=f"{tagj}_{tag}", arm=arm, seed=seed, fold=tag,
                   n_tr=len(tr), n_te=len(te),
                   fold_auc_h=dh["fold_auc"][0] if dh["fold_auc"] else None,
                   fold_auc_z=dz["fold_auc"][0] if dz["fold_auc"] else None,
                   fold_auc_head=(float(roc_auc_score(y[te], head_scores[te]))
                                  if head_scores is not None
                                  and len(np.unique(y[te])) > 1 else None),
                   best_val_auc=info["best_val_auc"], best_epoch=info["best_epoch"],
                   epochs_run=info["epochs_run"], arm4_valid=valid,
                   keep_rate_range=([min(k for k in info["keep_rates"] if k is not None),
                                     max(k for k in info["keep_rates"] if k is not None)]
                                    if arm == 4 and info["keep_rates"] else None),
                   grad_finite_all=bool(all(c.get("grad_finite", True) for c in curve)),
                   hostname=socket.gethostname(),
                   gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                   wall_s=round(time.time() - t0, 1),
                   dataset_sha=K.DATASET_SHA, splits_sha=K.SPLITS_SHA,
                   git_sha="7886277", cfg=dict(arm=arm, lr=M.LR, batch=M.BATCH,
                       emb=M.EMB, norm_nodes=False, max_epochs=M.MAX_EPOCHS))
        # review blocker fix: JSON written atomically BEFORE the npz resume marker,
        # so a crash between them re-runs the fold instead of losing the record.
        jp = f"{M.S12A4}out/fold_{tagj}_{tag}.json"
        json.dump(dict(rec=rec, curve=curve,
                       provenance=C7.provenance({"unit": rec["unit"]})),
                  open(jp + ".tmp", "w"), indent=1, default=str)
        assert json.load(open(jp + ".tmp"))["rec"]; os.replace(jp + ".tmp", jp)
        tmp = fp + ".tmp.npz"
        np.savez_compressed(tmp[:-4], h=H.astype(np.float32), z=Z.astype(np.float32),
                            nodes_flat=ND.astype(np.float32), y=y,
                            tr=np.asarray(tr), te=np.asarray(te),
                            oof_h=oh, oof_z=oz,
                            head=head_scores if head_scores is not None else np.zeros(1))
        zz = np.load(tmp); assert np.isfinite(zz["h"]).all(); os.replace(tmp, fp)
        print("DONE", tagj, tag, "h", rec["fold_auc_h"], "head", rec["fold_auc_head"],
              f"{rec['wall_s']}s ep{rec['best_epoch']}/{rec['epochs_run']}", flush=True)
    open(f"{M.S12A4}out/JOB_{tagj}.done", "w").write("done")

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu": torch.set_num_threads(int(os.environ.get("S12A4_THREADS", "4")))
    a, s = UNITS[int(sys.argv[1])]
    run(a, s, dev)
