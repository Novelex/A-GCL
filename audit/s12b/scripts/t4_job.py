"""S12B Track 4 — protocol inflation on the EXISTING production backbone
(s8_core CFG 'C': corrected-C, TUEncoder(3,emb32), global_add_pool, verbatim
train_step). Representation h ONLY (z never probed). One task = one seed.
T-ind: SSL per ordinary fold on TRAIN subjects; T-trans: SSL on all 954.
Eval every 5 epochs, 100 epochs -> 20 evals. E-best = test-selected best eval
(the inflationary practice, measured deliberately); E-final = epoch 100."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12b/scripts"); import s12b_core as B
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s8"); import s8_core as S8
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
from torch_geometric.loader import DataLoader

EPOCHS, EVERY = 100, 5

def probe_fold(H, y, tr, te):
    ev = B.StageEval(y, XFC=None, conf=None, keep_proj=False)
    ev.fold_fit(H, tr, te, "o0", njobs=1)
    p = ev.p_ord[te]
    return p, float(ev.fold_metrics["o0"]["auc"])

def extract_h_manifest(model, dl, cache2man, dev):
    H, Z, Y = S8.extract(model, dl, dev)                 # cache order; Z ignored (banned)
    Hm = np.empty_like(H); Hm[cache2man] = H
    return Hm

def ssl_train(seed, sub_dl, dev, evals_cb):
    model, view, mopt, vopt, bank, c = S8.build("C", seed, dev)
    g = torch.Generator(); g.manual_seed(S8.BASE + seed)
    loader = DataLoader(sub_dl, batch_size=S8.BATCH, shuffle=True, generator=g,
                        drop_last=True)
    curves = []
    for ep in range(1, EPOCHS + 1):
        st = []
        for batch in loader:
            batch = batch.to(dev)
            S8.train_step("C", model, view, mopt, vopt, bank, batch, dev, stats=st)
        agg = {k: float(np.mean([s[k] for s in st])) for k in st[0]}
        agg["epoch"] = ep; curves.append(agg)
        if ep % EVERY == 0:
            evals_cb(ep, model)
    return model, curves

def main(sidx):
    seed = sidx                                           # S8.build uses BASE+seed inside
    tag = f"t4_s{sidx}"
    jp = f"{B.S12B}jobs/{tag}.json"
    if os.path.exists(jp): print("skip", tag); return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = B.load_all(); y = d["y"]
    df, _, y2, ids, _ = A1.load_gate(); assert np.array_equal(y, y2)
    cache2man = np.empty(954, dtype=int)                  # cache idx -> manifest idx
    man2cache = np.empty(954, dtype=int)
    for r in df.itertuples():
        cache2man[int(r.s5_cache_index)] = int(r.row_index)
        man2cache[int(r.row_index)] = int(r.s5_cache_index)
    dl = S8.load_dataset()
    yc = np.array([int(g_.y) for g_ in dl]); assert np.array_equal(yc[man2cache], y)
    folds = [(t, tr, te) for t, tr, te in B.folds_all(y) if t.startswith("o")]
    t0 = time.time()

    # ---------------- T-trans: SSL on all 954 ----------------
    trans_evals = []
    def cb_trans(ep, model):
        H = extract_h_manifest(model, dl, cache2man, dev)
        oof = np.full(954, np.nan)
        for t, tr, te in folds:
            p, _ = probe_fold(H, y, tr, te); oof[te] = p
        a = B.metric_block(y, oof)["auc"]             # boot=2000 as registered (review R5)
        trans_evals.append(dict(epoch=ep, pooled_auc=float(a)))
        print(f"[{tag}] trans ep{ep} pooled {a:.4f}", flush=True)
    _, trans_curves = ssl_train(seed, dl, dev, cb_trans)

    # ---------------- T-ind: SSL per fold on TRAIN only ----------------
    ind = {}
    for t, tr, te in folds:
        sub = [dl[man2cache[i]] for i in tr]
        fold_evals = []
        def cb_ind(ep, model, _t=t, _tr=tr, _te=te):
            H = extract_h_manifest(model, dl, cache2man, dev)
            p, a = probe_fold(H, y, _tr, _te)
            fold_evals.append(dict(epoch=ep, auc=float(a), p=p.tolist()))
        _, cur = ssl_train(seed, sub, dev, cb_ind)
        ind[t] = dict(evals=[{k: v for k, v in e.items() if k != "p"} for e in fold_evals],
                      p_final=fold_evals[-1]["p"],
                      p_best=max(fold_evals, key=lambda e: e["auc"])["p"],
                      best_epoch=max(fold_evals, key=lambda e: e["auc"])["epoch"],
                      keep_last=cur[-1]["keep_sampled"], model_loss_last=cur[-1]["model_loss"])
        print(f"[{tag}] ind {t} final {fold_evals[-1]['auc']:.4f}"
              f" best {max(e['auc'] for e in fold_evals):.4f}", flush=True)

    oof_fin = np.full(954, np.nan); oof_best = np.full(954, np.nan)
    for t, tr, te in folds:
        oof_fin[te] = np.array(ind[t]["p_final"]); oof_best[te] = np.array(ind[t]["p_best"])
    ind_final = float(B.metric_block(y, oof_fin)["auc"])
    ind_best = float(B.metric_block(y, oof_best)["auc"])
    trans_final = trans_evals[-1]["pooled_auc"]
    trans_best = max(e["pooled_auc"] for e in trans_evals)
    res = dict(unit=tag, seed_arg=sidx, backbone="production corrected-C (s8 CFG C), h only",
               trans=dict(evals=trans_evals, final=trans_final, best=trans_best,
                          model_loss_last=trans_curves[-1]["model_loss"],
                          keep_last=trans_curves[-1]["keep_sampled"]),
               ind=dict(per_fold={t: {k: v for k, v in ind[t].items()
                                      if k not in ("p_final", "p_best")} for t in ind},
                        final=ind_final, best=ind_best),
               deltas=dict(trans_minus_ind_final=trans_final - ind_final,
                           trans_minus_ind_best=trans_best - ind_best,
                           best_minus_final_trans=trans_best - trans_final,
                           best_minus_final_ind=ind_best - ind_final),
               wall_s=round(time.time() - t0, 1), provenance=B.provenance())
    B.atomic_json(res, jp)
    print("DONE", tag, json.dumps(res["deltas"]), flush=True)

if __name__ == "__main__":
    main(int(sys.argv[1]))
