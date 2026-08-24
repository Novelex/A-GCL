"""S16 C2 PRECISION REFIT. Repeats BOTH random draws (the te half-split and the
matched tr subset) over 20 seeds and reports mean +/- SE plus sign-flip frequency.
RETRAINS NOTHING — refits the frozen probe on saved representations only.

WHY: every pure-bias value previously rested on ONE draw of each split. With ~95
scoring subjects the paired difference carries SE ~ +-0.02-0.03, the same order as the
+0.033..+0.054 effects reported. No single-draw number is quotable at that precision."""
import sys, os, json, glob, time, numpy as np
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
S16="/users/3171356m/A-GCL/audit/s16/"; BASE=20260818; NREP=20

def _p(R,y,fit,score):
    d,o=K.probe_pipe(np.asarray(R,dtype=np.float64),y,
                     [(np.asarray(fit),np.asarray(score))],[])
    return o[np.asarray(score)]

def one_rep(R,y,tr,te,seed):
    """Returns (honest_oof_idx,val), (matched_oof_idx,val) for ONE draw of both splits."""
    tr=np.asarray(tr); te=np.asarray(te); yt=y[te]
    skf=StratifiedKFold(2,shuffle=True,random_state=seed)
    o3=np.full(len(te),np.nan); o4=np.full(len(te),np.nan)
    rng=np.random.default_rng(seed)
    pos=tr[y[tr]==1]; neg=tr[y[tr]==0]
    for a,b in skf.split(np.zeros(len(te)),yt):
        o3[b]=_p(R,y,te[a],te[b])
        npos=int((y[te[a]]==1).sum()); nneg=len(a)-npos
        sel=np.concatenate([rng.choice(pos,min(npos,len(pos)),replace=False),
                            rng.choice(neg,min(nneg,len(neg)),replace=False)])
        o4[b]=_p(R,y,sel,te[b])
    m3=np.isfinite(o3); m4=np.isfinite(o4)
    return (te[m3],o3[m3]),(te[m4],o4[m4])

SRC=[
 ("RANDOM WGIN (S12A5 A repr0, epoch-0)  [CALIBRATION]",
  "/users/3171356m/agcl_audit_s0/s12a5/feat/A_s0_o*.npz","y","repr0"),
 ("trained WGIN (S12A4 arm1 h)",
  "/users/3171356m/agcl_audit_s0/s12a4/feat/a1_s0_o*.npz","y","h"),
 ("S12A5 arm A (WGIN)","/users/3171356m/agcl_audit_s0/s12a5/feat/A_s0_o*.npz","y","repr"),
 ("S12A5 arm B (WGIN+edge skip)","/users/3171356m/agcl_audit_s0/s12a5/feat/B_s0_o*.npz","y","repr"),
 ("S12A5 arm C (edge MLP)","/users/3171356m/agcl_audit_s0/s12a5/feat/C_s0_o*.npz","y","repr"),
 ("S13 BNT winner (K=2 wd1e-4)",
  "/users/3171356m/A-GCL/audit/s13/feat/T2_K2_wd0.0001_s0__o*.npz","y_true","repr"),
 ("S15 B1 BNT K=32 (terminated)",
  "/users/3171356m/A-GCL/audit/s15/feat/main_B1_BNT_kh32_L-BCE_lr0.0003_wd0.001_s0__lab*.npz",
  "y_true","repr"),
]

def main():
    t0=time.time(); out=[]
    for name,pat,ykey,rkey in SRC:
        fs=sorted(glob.glob(pat))
        if not fs: out.append(dict(source=name,n_folds=0,note="NO SAVED FOLDS")); continue
        y=np.load(fs[0])[ykey].astype(np.int64)
        diffs=[]; hons=[]; mats=[]
        for rep in range(NREP):
            seed=BASE+rep
            oh=np.full(len(y),np.nan); om=np.full(len(y),np.nan)
            for f in fs:
                z=np.load(f)
                if rkey not in z.files: continue
                (i3,v3),(i4,v4)=one_rep(z[rkey],y,z["tr"],z["te"],seed)
                oh[i3]=v3; om[i4]=v4
            mh=np.isfinite(oh); mm=np.isfinite(om)
            ah=roc_auc_score(y[mh],oh[mh]); am=roc_auc_score(y[mm],om[mm])
            hons.append(ah); mats.append(am); diffs.append(am-ah)
            print(f"  {name[:38]:38s} rep{rep:02d} honest {ah:.4f} matched {am:.4f} "
                  f"bias {am-ah:+.4f}",flush=True)
        d=np.array(diffs)
        r=dict(source=name,n_folds=len(fs),n_reps=NREP,repr_dim=int(np.load(fs[0])[rkey].shape[1]),
               honest_mean=float(np.mean(hons)),honest_se=float(np.std(hons,ddof=1)/np.sqrt(NREP)),
               matched_mean=float(np.mean(mats)),matched_se=float(np.std(mats,ddof=1)/np.sqrt(NREP)),
               pure_bias_mean=float(d.mean()),
               pure_bias_se=float(d.std(ddof=1)/np.sqrt(NREP)),
               pure_bias_sd=float(d.std(ddof=1)),
               sign_flips=int((d<0).sum()),
               sign_flip_pct=float(100.0*(d<0).sum()/NREP),
               ci_lo=float(np.percentile(d,2.5)),ci_hi=float(np.percentile(d,97.5)))
        out.append(r)
        print(f"== {name}: PURE BIAS {r['pure_bias_mean']:+.4f} +- {r['pure_bias_se']:.4f} (SE) "
              f"| sd {r['pure_bias_sd']:.4f} | sign flips {r['sign_flips']}/{NREP} "
              f"({r['sign_flip_pct']:.0f}%)",flush=True)
    md=["# S16 C2 — PRECISION REFIT (20 repeats of BOTH random draws)","",
      f"Every pure-bias value in the first C2 pass rested on ONE draw of the te",
      f"half-split and ONE draw of the matched tr subset. At ~95 scoring subjects the",
      f"paired difference carries SE of roughly +-0.02-0.03 — the SAME ORDER as the",
      f"+0.033..+0.054 effects being reported. Both draws are now repeated over",
      f"**{NREP} seeds** ({BASE}..{BASE+NREP-1}), the frozen probe refitted on the same",
      f"saved representations each time. Nothing retrained.","",
      "| source | dim | honest (mean±SE) | matched (mean±SE) | **PURE BIAS (mean±SE)** | sd | sign flips |",
      "|---|---|---|---|---|---|---|"]
    for r in out:
        if r.get("n_folds",0)==0: md.append(f"| {r['source']} | — | — | — | — | — | {r.get('note','')} |"); continue
        md.append(f"| {r['source']} | {r['repr_dim']} | {r['honest_mean']:.4f}±{r['honest_se']:.4f} | "
                  f"{r['matched_mean']:.4f}±{r['matched_se']:.4f} | "
                  f"**{r['pure_bias_mean']:+.4f}±{r['pure_bias_se']:.4f}** | {r['pure_bias_sd']:.4f} | "
                  f"{r['sign_flips']}/{r['n_reps']} ({r['sign_flip_pct']:.0f}%) |")
    md+=["","**How to read the sign-flip column.** It is the fraction of the 20 draws in which",
      "the measured bias came out NEGATIVE. For a genuine positive effect it should be",
      "near 0; for a true zero it should sit near 50%. It is the non-parametric check",
      "that does not depend on the SE being well estimated.","",
      "**Quotability rule.** Report mean ± SE. A single draw is NOT quotable: the",
      "per-draw sd column shows how far one draw can land from the mean.",
      f"","wall {time.time()-t0:.0f}s"]
    open(S16+"C2_PRECISION.md.tmp","w").write("\n".join(md)+"\n")
    os.replace(S16+"C2_PRECISION.md.tmp",S16+"C2_PRECISION.md")
    json.dump(out,open(S16+"out/C2_PRECISION.json","w"),indent=1,default=str)
    print("C2_PRECISION.md written",flush=True)

if __name__=="__main__": main()
