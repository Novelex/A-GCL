"""S16 C2 (AMENDED) — decompose the probe delta. RETRAINS NOTHING.

The original single delta was CONFOUNDED: probe_honest changed three things at once
(removed the bias, cut the probe's training set 763->~153, cut the encoder's 763->610).
Four probes are now computed on the SAME saved representations:

  1 probe_old_full      fit on tr (~763, MEMORISED by the encoder), score te
  2 probe_old_subset    fit on tr_probe (~153, MEMORISED), score te      [same encoder]
  3 probe_honest_teSpl  fit on one half of te (UNSEEN), score the other half
  4 probe_biased_match  fit on a MEMORISED subset of tr of EXACTLY the same size as
                        the half of te used in (3), score the SAME half of te

DECOMPOSITION
  (1) - (2)  = SAMPLE-SIZE EFFECT      probe training set 763 -> 153, no bias change
  (4) - (3)  = PURE BIAS               identical training-set SIZE, identical SCORING
                                       set; differs ONLY in whether the probe's
                                       training subjects were memorised by the encoder
  (2) - (3)  = the commissioned "actual bias" (retains a residual size confound,
                                       153 vs ~95 — reported, but (4)-(3) is cleaner)

WHY probe_honest ITSELF IS NOT COMPUTABLE HERE: for an already-trained checkpoint the
encoder saw ALL of tr, so no subset of tr is out-of-sample and S16's tr_enc/tr_probe
split cannot be reconstructed after the fact. Only te is unseen. S16's own C6 runs DO
produce a true probe_honest because their encoders train on tr_enc only.

CALIBRATION POINT: the RANDOM (epoch-0) encoder never trained, so it has NO
distribution shift and its (4)-(3) must be ~0, while its (1)-(2) shows the pure
sample-size effect. It isolates the two effects cleanly."""
import sys, os, json, glob, time, numpy as np
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
S16="/users/3171356m/A-GCL/audit/s16/"; BASE=20260818

def _p(R,y,fit,score):
    d,o = K.probe_pipe(np.asarray(R,dtype=np.float64), y,
                       [(np.asarray(fit), np.asarray(score))], [])
    return o[np.asarray(score)]

def four_probes(R, y, tr, te):
    tr=np.asarray(tr); te=np.asarray(te); yt=y[te]
    if len(np.unique(yt))<2 or len(te)<12: return None
    # (1) full
    s1=_p(R,y,tr,te); a1=roc_auc_score(yt,s1)
    # (2) subset: the SAME tr_probe split S16 uses (20% of tr, stratified, seed BASE)
    _, ip = train_test_split(np.arange(len(tr)), test_size=0.20,
                             stratify=y[tr], random_state=BASE)
    tr_probe = tr[ip]
    s2=_p(R,y,tr_probe,te); a2=roc_auc_score(yt,s2)
    # (3)/(4): two-fold split of te; both directions; size-matched biased twin
    skf=StratifiedKFold(2,shuffle=True,random_state=BASE)
    o3=np.full(len(te),np.nan); o4=np.full(len(te),np.nan)
    rng=np.random.default_rng(BASE)
    for a,b in skf.split(np.zeros(len(te)), yt):
        o3[b]=_p(R,y,te[a],te[b])                      # UNSEEN training subjects
        n=len(a)
        # size-matched MEMORISED training subjects, class balance matched to te[a]
        pos=tr[y[tr]==1]; neg=tr[y[tr]==0]
        npos=int((y[te[a]]==1).sum()); nneg=n-npos
        sel=np.concatenate([rng.choice(pos,min(npos,len(pos)),replace=False),
                            rng.choice(neg,min(nneg,len(neg)),replace=False)])
        o4[b]=_p(R,y,sel,te[b])                        # MEMORISED, same size
    m3=np.isfinite(o3); m4=np.isfinite(o4)
    a3=roc_auc_score(yt[m3],o3[m3]) if len(np.unique(yt[m3]))>1 else np.nan
    a4=roc_auc_score(yt[m4],o4[m4]) if len(np.unique(yt[m4]))>1 else np.nan
    return dict(old_full=float(a1), old_subset=float(a2), honest_teSplit=float(a3),
                biased_matched=float(a4), n_tr=int(len(tr)), n_tr_probe=int(len(tr_probe)),
                n_te=int(len(te)), n_honest_score=int(m3.sum()),
                _oof=dict(old_full=(te, s1), old_subset=(te, s2),
                          honest_teSplit=(te[m3], o3[m3]),
                          biased_matched=(te[m4], o4[m4])))

SRC=[
 ("RANDOM WGIN (S12A5 A repr0, epoch-0)  [CALIBRATION]",
  "/users/3171356m/agcl_audit_s0/s12a5/feat/A_s0_o*.npz","y","repr0"),
 ("trained WGIN (S12A4 arm1 h)",
  "/users/3171356m/agcl_audit_s0/s12a4/feat/a1_s0_o*.npz","y","h"),
 ("S12A5 arm A (WGIN)",
  "/users/3171356m/agcl_audit_s0/s12a5/feat/A_s0_o*.npz","y","repr"),
 ("S12A5 arm B (WGIN+edge skip)",
  "/users/3171356m/agcl_audit_s0/s12a5/feat/B_s0_o*.npz","y","repr"),
 ("S12A5 arm C (edge MLP)",
  "/users/3171356m/agcl_audit_s0/s12a5/feat/C_s0_o*.npz","y","repr"),
 ("S13 BNT winner (K=2 wd1e-4)",
  "/users/3171356m/A-GCL/audit/s13/feat/T2_K2_wd0.0001_s0__o*.npz","y_true","repr"),
 ("S15 B1 BNT K=32 (terminated)",
  "/users/3171356m/A-GCL/audit/s15/feat/main_B1_BNT_kh32_L-BCE_lr0.0003_wd0.001_s0__lab*.npz",
  "y_true","repr"),
]

def main():
    t0=time.time(); rows=[]
    for name,pat,ykey,rkey in SRC:
        fs=sorted(glob.glob(pat))
        if not fs: rows.append(dict(source=name,n=0,note="NO SAVED FOLDS")); continue
        acc=[]
        for f in fs:
            z=np.load(f)
            if rkey not in z.files: continue
            r=four_probes(z[rkey], z[ykey].astype(np.int64), z["tr"], z["te"])
            if r is None: continue
            acc.append(r)
            print(f"  {name[:38]:38s} {os.path.basename(f)[:26]:26s} "
                  f"full {r['old_full']:.4f} sub {r['old_subset']:.4f} "
                  f"hon {r['honest_teSplit']:.4f} match {r['biased_matched']:.4f}", flush=True)
        if not acc: rows.append(dict(source=name,n=0,note="no evaluable folds")); continue
        # POOLED out-of-fold AUC (headline): one AUC over all covered subjects,
        # not a mean of small per-fold AUCs.
        yfull = np.load(fs[0])[ykey].astype(np.int64)
        pooled={}
        for pk in ("old_full","old_subset","honest_teSplit","biased_matched"):
            o=np.full(len(yfull), np.nan)
            for a in acc:
                idx,val = a["_oof"][pk]; o[idx]=val
            m=np.isfinite(o)
            pooled[pk]=float(roc_auc_score(yfull[m], o[m])) if len(np.unique(yfull[m]))>1 else np.nan
            pooled[pk+"_n"]=int(m.sum())
        g=lambda k: float(np.mean([a[k] for a in acc]))
        sd=lambda k: float(np.std([a[k] for a in acc]))
        row=dict(source=name,n=len(acc),repr_dim=int(z[rkey].shape[1]),
                 old_full=g("old_full"),old_full_sd=sd("old_full"),
                 old_subset=g("old_subset"),old_subset_sd=sd("old_subset"),
                 honest=g("honest_teSplit"),honest_sd=sd("honest_teSplit"),
                 matched=g("biased_matched"),matched_sd=sd("biased_matched"),
                 n_tr=acc[0]["n_tr"],n_tr_probe=acc[0]["n_tr_probe"],n_te=acc[0]["n_te"])
        row.update({"pooled_"+k:v for k,v in pooled.items()})
        # HEADLINE decomposition uses POOLED values
        row["size_effect"]=pooled["old_full"]-pooled["old_subset"]
        row["pure_bias"]=pooled["biased_matched"]-pooled["honest_teSplit"]
        row["commissioned_bias"]=pooled["old_subset"]-pooled["honest_teSplit"]
        row["perfold_mean_note"]="per-fold means retained in JSON; NOISE-DOMINATED"
        row["n_honest_score_perfold"]=acc[0]["n_honest_score"]
        rows.append(row)
        print(f"== {name}: POOLED full {pooled['old_full']:.4f} sub {pooled['old_subset']:.4f} "
              f"hon {pooled['honest_teSplit']:.4f} match {pooled['biased_matched']:.4f} "
              f"|| size {row['size_effect']:+.4f} | PURE BIAS {row['pure_bias']:+.4f} | "
              f"commissioned {row['commissioned_bias']:+.4f}",flush=True)
    md=["# S16 C2 (AMENDED) — DECOMPOSING THE PROBE DELTA","",
      "The earlier single delta was CONFOUNDED: `probe_honest` changed three things at",
      "once — it removed the bias, cut the probe's training set from ~763 to ~153, and",
      "cut the encoder's from ~763 to ~610. What follows separates them.","",
      "## The four probes (all on the SAME saved representations, nothing retrained)",
      "| probe | fit on | memorised by encoder? | scores |",
      "|---|---|---|---|",
      "| 1 `old_full` | tr (~763) | YES | te |",
      "| 2 `old_subset` | tr_probe (~153) | YES | te |",
      "| 3 `honest_teSplit` | one half of te (~95) | **NO** | other half of te |",
      "| 4 `biased_matched` | tr subset, SAME SIZE as (3), class-balance matched | YES | same half of te |",
      "","## Decomposition",
      "- **(1) − (2) = SAMPLE-SIZE EFFECT** — probe training set shrinks, bias unchanged",
      "- **(4) − (3) = PURE BIAS** — identical training-set size, identical scoring set;",
      "  differs ONLY in whether the probe's training subjects were memorised",
      "- (2) − (3) = the commissioned \"actual bias\" (retains a residual 153-vs-95 size",
      "  confound; reported for continuity, but (4) − (3) is the clean number)","",
      "**Why true `probe_honest` is not computable here:** for an already-trained",
      "checkpoint the encoder saw ALL of tr, so no subset of tr is out-of-sample and",
      "S16's tr_enc/tr_probe split cannot be reconstructed after the fact. Only te is",
      "unseen. S16's own C6 runs DO yield a true `probe_honest`, because their encoders",
      "train on tr_enc only.","",
      "## PRECISION WARNING — READ BEFORE ANY NUMBER",
      "`honest_teSplit` scores only ~95 subjects per fold, so a PER-FOLD AUC carries a",
      "standard error of roughly **±0.05**. Per-fold values are therefore",
      "**NOISE-DOMINATED and must never be quoted individually.** Every headline number",
      "below is a **POOLED out-of-fold AUC** — one AUC computed over all covered",
      "subjects at once, not a mean of small per-fold AUCs. Per-fold values are retained",
      "in `out/C2_RESCORE.json` and are explicitly marked noise-dominated there.","",
      "## Results — POOLED out-of-fold AUC (headline)","",
      "| source | folds | dim | 1 old_full | 2 old_subset | 3 honest_teSplit | 4 biased_matched | size effect (1−2) | **PURE BIAS (4−3)** | commissioned (2−3) |",
      "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("n",0)==0: md.append(f"| {r['source']} | 0 | — | — | — | — | — | — | — | {r.get('note','')} |"); continue
        md.append(f"| {r['source']} | {r['n']} | {r['repr_dim']} | "
          f"{r['pooled_old_full']:.4f} | {r['pooled_old_subset']:.4f} | "
          f"{r['pooled_honest_teSplit']:.4f} | {r['pooled_biased_matched']:.4f} | "
          f"{r['size_effect']:+.4f} | **{r['pure_bias']:+.4f}** | {r['commissioned_bias']:+.4f} |")
    md+=["","**FIXED ANCHOR, carries no bias:** LinearSVC on raw FC = **0.7565** ord /",
      "**0.7432** LOSO. No encoder is fitted to produce raw FC, so no distribution shift",
      "exists and neither correction applies to it.","",
      "**CALIBRATION:** the RANDOM (epoch-0) encoder never trained, so its PURE BIAS",
      "must be ≈ 0 while its SAMPLE-SIZE EFFECT is real. It is the control that separates",
      "the two columns.","",
      "**FLAGGED FOR C7, NOT NOW:** the statistically clean fix is CROSS-FITTING — an",
      "inner K-fold that builds a fully out-of-sample R[tr] using all 763 subjects, so",
      "there is no bias AND no sample starvation. It costs 5× encoder training, so it is",
      "for the winner only, at C7.",f"","wall {time.time()-t0:.0f}s"]
    open(S16+"C2_PROBE.md.tmp","w").write("\n".join(md)+"\n")
    os.replace(S16+"C2_PROBE.md.tmp",S16+"C2_PROBE.md")
    json.dump(rows,open(S16+"out/C2_RESCORE.json","w"),indent=1,default=str)
    print("C2_PROBE.md written",flush=True)

if __name__=="__main__": main()
