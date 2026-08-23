"""S13 worker — one unit = (arm, K_clusters, wd, seed) over 24 frozen folds.
Mirrors w_wave1.py: atomic writes, fold JSON BEFORE the npz resume marker
(S12A4b lesson), SKIP-keyed on the npz, Gate-0 hashes re-verified at start."""
import sys, os, json, time, socket, hashlib, numpy as np, torch
sys.path.insert(0, "/users/3171356m/A-GCL/audit/s13/scripts"); import bnt_core as B
import s11_core as K
from sklearn.metrics import roc_auc_score

# ---- unit table: index is STABLE, never reorder ----
UNITS  = [("T2", k, wd, s) for k in B.K_GRID for wd in B.WD_GRID for s in range(3)]  # 18
UNITS += [(a, 4, 1e-4, s) for a in ("T1", "T4", "T5", "T6") for s in range(3)]       # 12
# T3 (116 ROIs) is NOT in this table — see PROTOCOL.md Stage 2 for the reason.

def unit_tag(arm, k, wd, sidx): return f"{arm}_K{k}_wd{wd:g}_s{sidx}"

def run(idx):
    arm, Kc, wd, sidx = UNITS[idx]
    seed = B.SEEDS[sidx]; tag = unit_tag(arm, Kc, wd, sidx)
    jd = f"{B.S13}jobs/{tag}"; os.makedirs(jd, exist_ok=True)
    t_start = time.time()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    d = B.load_all()                                    # Gate-2 test 9
    FC, ALFF = d["FC"], d["ALFF"]
    y_true = d["y"]; y_use = B.arm_y(arm, y_true)       # T6 permutes labels
    folds = B.folds_all(y_true)
    print(f"[{tag}] arm={arm} K={Kc} wd={wd:g} seed={seed} folds={len(folds)} "
          f"host={socket.gethostname()} cores={os.environ.get('SLURM_CPUS_PER_TASK')}",
          flush=True)
    for tagf, tr, te in folds:
        fp = f"{B.S13}feat/{tag}__{tagf}.npz"
        if os.path.exists(fp): print(f"skip {tag} {tagf}", flush=True); continue
        t0 = time.time()
        X = B.arm_X(arm, FC, ALFF, tr)                  # ALFF z-scored on TRAIN ONLY
        model, curve, info = B.train_fold(arm, Kc, wd, seed, tr, X, y_use,
                                          log=f"{tag}/{tagf}")
        R, S = B.extract(model, X, np.arange(954))      # repr = Z_G, head logits
        # PROBE: K.probe_pipe on Z_G immediately after the encoder
        pd_, poof = K.probe_pipe(R.astype(np.float64), y_use, [(tr, te)], [])
        ck = f"{B.S13}ckpt/{tag}__{tagf}.pt"
        torch.save(model.state_dict(), ck + ".tmp"); os.replace(ck + ".tmp", ck)
        head_m  = B.metric_block(y_use[te], S[te])
        probe_m = B.metric_block(y_use[te], poof[te])
        rec = dict(unit=tag, arm=arm, K=Kc, wd=wd, seed=seed, seed_idx=sidx, fold=tagf,
                   n_params=info["n_params"], head=head_m, probe=probe_m,
                   best_val_auc=info["best_val_auc"], best_epoch=info["best_epoch"],
                   epochs_run=info["epochs_run"], train_val_gap=info["train_val_gap"],
                   verdict=info["verdict"], integrity=info["integrity"],
                   movement=info["movement"],
                   train_auc_at_best=curve[info["best_epoch"]-1]["train_auc"],
                   val_auc_at_best=curve[info["best_epoch"]-1]["val_auc"],
                   clip_events_total=int(sum(c["clip_events"] for c in curve)),
                   n_train=info["n_train"], n_val=info["n_val"],
                   n_test=int(len(te)), label_convention="ASD=1 NC=0 (A-GCL uses ASD=0"
                   " HC=1: AUC identical, SENS/SPEC SWAPPED vs the paper's table)",
                   ckpt_sha=hashlib.sha256(open(ck, "rb").read()).hexdigest()[:16],
                   wall_s=round(time.time()-t0, 1), peak_rss_mb=B.peak_rss_mb(),
                   provenance=B.provenance({"unit": tag, "fold": tagf}))
        # fold JSON FIRST, npz resume marker LAST (S12A4b ordering lesson)
        B.atomic_json(dict(rec=rec, curve=curve), f"{jd}/fold_{tagf}.json")
        tmp = fp + ".tmp.npz"
        np.savez_compressed(tmp[:-4], repr=R.astype(np.float32),
                            head=S.astype(np.float32), probe_oof=poof,
                            y_true=y_true, y_used=y_use,
                            tr=np.asarray(tr), te=np.asarray(te))
        zz = np.load(tmp); assert np.isfinite(zz["repr"]).all(); os.replace(tmp, fp)
        print(f"DONE {tag} {tagf} head {head_m['auc']:.4f} probe {probe_m['auc']:.4f} "
              f"val {info['best_val_auc']:.4f} ep{info['best_epoch']} "
              f"{info['verdict']} {rec['wall_s']}s", flush=True)
    B.atomic_text("done", f"{jd}/UNIT.done")
    print(f"UNIT_COMPLETE {tag} total {time.time()-t_start:.0f}s", flush=True)

if __name__ == "__main__":
    run(int(sys.argv[1]))
