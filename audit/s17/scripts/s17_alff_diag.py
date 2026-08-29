"""S17 ALFF diagnostic: is ALFF redundant with FC, or complementary? CPU only, no
encoder training, no new architecture, no wave. Six pre-specified steps.

Scores (all matched OOF on the frozen S3C 5 folds, LinearSVC, S5.5 inner-fold C):
  s_FC   FC 4005 flat                         gate: 0.7565
  s_TAN  FC tangent+shrinkage (tangent2 grid) gate: 0.7783  (same code path as tangent2)
  s_ALFF M1D per-subject z within band, 270   gate: 0.6486  (S5.5 M1D_only linsvm; the
                                              S3C M1 tensor is BITWISE the raw npz)
M1 is read from ALFF_func_proc/method1/alff_roi_first.npz directly; no S16 loader.
Writes ONLY under audit/s17/runs/alffdiag/.
"""
import sys, os, json, time, shutil
import numpy as np, pandas as pd
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s55")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s55_core as C
from s17_tanlib import TangentShrink
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
from joblib import Memory

OUT = "/users/3171356m/A-GCL/audit/s17/runs/alffdiag/"; os.makedirs(OUT, exist_ok=True)
assert "/audit/s17/" in OUT and "/audit/s16/" not in OUT
NJ = int(os.environ.get("DIAG_NJOBS", "6")); BOOT_N = 2000
MEMDIR = "/tmp/claude-102000043/-users-3171356m-A-GCL/a9e2c01c-7bf7-4c3e-b518-73a2642d296c/scratchpad/diag_cache"
shutil.rmtree(MEMDIR, ignore_errors=True); mem = Memory(MEMDIR, verbose=0)
ALPHAS2 = [0.01, 0.05, 0.1, 0.2, 0.4]; CGRID = list(C.GRID["clf__C"])   # tangent2's grids
SKF = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
log_lines = []
def log(s): print(s, flush=True); log_lines.append(s)
def aj(o, p):
    json.dump(o, open(p + ".tmp", "w"), indent=1, default=str); json.load(open(p + ".tmp")); os.replace(p + ".tmp", p)

# ------------------------------------------------ data
Z = np.load("/users/3171356m/A-GCL/ALFF_func_proc/method1/alff_roi_first.npz", allow_pickle=True)
M1 = Z["alff"].astype(np.float64); raw_ids = [str(x) for x in Z["file_ids"]]
smin, smax = M1.min((1, 2)), M1.max((1, 2))
assert not ((smin == 0.0).all() and (smax == 1.0).all()), "min-maxed tensor: WRONG FILE"
F = C.build(); y = np.asarray(F["y"]).astype(int); ids = [str(x) for x in F["ids"]]
assert raw_ids == ids, "order drift"
meta = pd.read_csv("/users/3171356m/agcl_audit_s0/s3c/meta.csv")
assert meta.subject_id.astype(str).tolist() == ids, "meta order drift"
site = meta.site.values
aal = pd.read_csv("/users/3171356m/agcl_audit_s0/s1_roi_alignment_90.csv")
assert len(aal) == 90 and (aal.position.values == np.arange(90)).all()
ROI = aal.aal_name.tolist()
iu = np.triu_indices(90, k=1)
FCu = np.asarray(F["FC"], dtype=np.float64)
FCsq = np.zeros((954, 90, 90)); FCsq[:, iu[0], iu[1]] = FCu
FCsq += FCsq.transpose(0, 2, 1); FCsq[:, np.arange(90), np.arange(90)] = 1.0
# M1D: per-subject z-score WITHIN each band across the 90 ROIs (S5.5 nb('D'))
M1D = ((M1 - M1.mean(1, keepdims=True)) / M1.std(1, keepdims=True)).reshape(954, -1)
assert M1D.shape == (954, 270) and np.isfinite(M1D).all()
assert np.array_equal(M1D, np.asarray(F["M1D"], dtype=np.float64)), "M1D != S5.5's M1D"
log("DATA ok: raw M1 verified; M1D bitwise == S5.5's M1D; meta/AAL aligned")

# ------------------------------------------------ STEP 1: matched OOF scores
def nested_capture(X, folds, grid=None, pipe_fn=None):
    """Same construction as C.nested (make_pipe/GRID/SKF/seed) but also returns the
    per-fold fitted estimator so Step 6 can read coefficients fitted on TRAIN ONLY."""
    oof = np.full(954, np.nan); models = []
    for tr, te in folds:
        gs = GridSearchCV(pipe_fn() if pipe_fn else C.make_pipe("linsvm"),
                          grid or C.GRID, cv=SKF, scoring="roc_auc", n_jobs=NJ, refit=True)
        gs.fit(X[tr], y[tr]); oof[te] = gs.decision_function(X[te]); models.append(gs)
    return float(roc_auc_score(y, oof)), oof, models

t0 = time.time()
auc_fc_ref, oof_fc_ref = (lambda r: (r[1]["auc"], r[2]))(C.nested(FCu, y, "linsvm", folds=C.OUTER, n_jobs=NJ))
auc_fc, s_FC, mdl_fc = nested_capture(FCu, C.OUTER)
assert np.array_equal(oof_fc_ref, s_FC), "nested_capture diverges from C.nested"
log(f"STEP1 s_FC   AUC {auc_fc:.4f} (C.nested {auc_fc_ref:.4f}, bitwise same OOF) ({time.time()-t0:.0f}s)")
if abs(auc_fc - 0.7565) > 5e-4: log("*** GATE s_FC != 0.7565 — STOP"); sys.exit(3)

t0 = time.time(); auc_alff, s_ALFF, mdl_alff = nested_capture(M1D, C.OUTER)
log(f"STEP1 s_ALFF AUC {auc_alff:.4f} (S5.5 M1D_only linsvm 0.6486) ({time.time()-t0:.0f}s)")
if abs(auc_alff - 0.6486) > 5e-4: log("*** GATE s_ALFF != 0.6486 — STOP"); sys.exit(3)

def tan_pipe():
    return Pipeline([("tan", TangentShrink()), ("sc", StandardScaler()),
                     ("clf", LinearSVC(dual="auto", max_iter=20000, random_state=C.SEED))], memory=mem)
t0 = time.time()
auc_tan, s_TAN, mdl_tan = nested_capture(FCsq.reshape(954, 8100), C.OUTER,
                                          grid={"tan__alpha": ALPHAS2, "clf__C": CGRID}, pipe_fn=tan_pipe)
hp_tan = [(m.best_params_["tan__alpha"], m.best_params_["clf__C"]) for m in mdl_tan]
log(f"STEP1 s_TAN  AUC {auc_tan:.4f} (tangent2 0.7783) hp {hp_tan} ({time.time()-t0:.0f}s)")
if abs(auc_tan - 0.7783) > 5e-4: log("*** GATE s_TAN != 0.7783 — STOP"); sys.exit(3)
log("STEP1 GATES: all three scores reproduce")

# ------------------------------------------------ STEP 2: redundancy
def resid_y(s): 
    Xd = np.c_[np.ones(954), y]; b = np.linalg.lstsq(Xd, s, rcond=None)[0]; return s - Xd @ b
def corrs(a, b): return dict(pearson=float(pearsonr(a, b)[0]), spearman=float(spearmanr(a, b)[0]))
step2 = {}
for nm, s in (("s_FC", s_FC), ("s_TAN", s_TAN)):
    step2[f"ALFF~{nm}"] = corrs(s_ALFF, s)
    step2[f"ALFF~{nm}|y_removed"] = corrs(resid_y(s_ALFF), resid_y(s))
for k, v in step2.items(): log(f"STEP2 {k:22s} pearson {v['pearson']:+.4f}  spearman {v['spearman']:+.4f}")

# ------------------------------------------------ STEP 3: error overlap
def youden(s):
    fpr, tpr, thr = roc_curve(y, s); j = int(np.argmax(tpr - fpr)); return float(thr[j])
thr = {k: youden(v) for k, v in (("s_FC", s_FC), ("s_TAN", s_TAN), ("s_ALFF", s_ALFF))}
pred = {k: (v > thr[k]).astype(int) for k, v in (("s_FC", s_FC), ("s_TAN", s_TAN), ("s_ALFF", s_ALFF))}
right = {k: (pred[k] == y) for k in pred}
t = dict(both_right=int((right["s_TAN"] & right["s_ALFF"]).sum()),
         tan_right_alff_wrong=int((right["s_TAN"] & ~right["s_ALFF"]).sum()),
         tan_wrong_alff_right=int((~right["s_TAN"] & right["s_ALFF"]).sum()),
         both_wrong=int((~right["s_TAN"] & ~right["s_ALFF"]).sum()))
log(f"STEP3 Youden thresholds {thr}")
log(f"STEP3 2x2 vs s_TAN: {t}  (sum {sum(t.values())})")
sub = ~right["s_TAN"]
sub_auc = float(roc_auc_score(y[sub], s_ALFF[sub])) if len(np.unique(y[sub])) > 1 else None
sub_acc = float(accuracy_score(y[sub], pred["s_ALFF"][sub]))
log(f"STEP3 subjects s_TAN gets WRONG: n={int(sub.sum())} (ASD {int(y[sub].sum())}) -> "
    f"ALFF AUC {sub_auc}  ALFF acc@Youden {sub_acc:.4f}")

# ------------------------------------------------ STEP 4: shared confound
conf = dict(mean_fd=meta.func_mean_fd.values.astype(float), T=meta["T"].values.astype(float),
            TR=meta["TR"].values.astype(float))
step4 = {}
for nm, s in (("s_FC", s_FC), ("s_TAN", s_TAN), ("s_ALFF", s_ALFF)):
    for cn, cv in conf.items():
        m = np.isfinite(cv); step4[f"{nm}~{cn}"] = corrs(s[m], cv[m])
    lr = LogisticRegression(max_iter=5000, random_state=C.SEED)
    sp = cross_val_predict(lr, s.reshape(-1, 1), site, cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED))
    step4[f"{nm}|site_bacc"] = float(balanced_accuracy_score(site, sp))
for nm in ("s_FC", "s_TAN", "s_ALFF"):
    log(f"STEP4 {nm}: " + "  ".join(f"{cn} r={step4[f'{nm}~{cn}']['pearson']:+.3f}" for cn in conf)
        + f"  | site bacc {step4[f'{nm}|site_bacc']:.4f} (chance {1/len(np.unique(site)):.4f}, {len(np.unique(site))} sites)")

# ------------------------------------------------ STEP 5: score-level fusion
W2 = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.1)]
fused = np.full(954, np.nan); w2_per_fold = []
for k, (tr, te) in enumerate(C.OUTER):
    tr = np.asarray(tr); te = np.asarray(te)
    a_k, c_k = hp_tan[k]; c_a = mdl_alff[k].best_params_["clf__C"]
    # inner cross-fit WITHIN tr with the fold's already-selected hyperparameters, so
    # the fusion weights see only out-of-sample inner scores of TRAINING subjects
    in_tan = np.full(len(tr), np.nan); in_alff = np.full(len(tr), np.nan)
    for itr, ite in SKF.split(tr, y[tr]):
        p = Pipeline([("tan", TangentShrink(alpha=a_k)), ("sc", StandardScaler()),
                      ("clf", LinearSVC(dual="auto", max_iter=20000, random_state=C.SEED, C=c_k))])
        p.fit(FCsq.reshape(954, 8100)[tr[itr]], y[tr[itr]]); in_tan[ite] = p.decision_function(FCsq.reshape(954, 8100)[tr[ite]])
        q = Pipeline([("sc", StandardScaler()), ("clf", LinearSVC(dual="auto", max_iter=20000, random_state=C.SEED, C=c_a))])
        q.fit(M1D[tr[itr]], y[tr[itr]]); in_alff[ite] = q.decision_function(M1D[tr[ite]])
    mu_t, sd_t = in_tan.mean(), in_tan.std(); mu_a, sd_a = in_alff.mean(), in_alff.std()
    zt, za = (in_tan - mu_t) / sd_t, (in_alff - mu_a) / sd_a
    best = max(W2, key=lambda w: (roc_auc_score(y[tr], (1 - w) * zt + w * za), -w))  # ties -> smaller w2
    w2_per_fold.append(best)
    fused[te] = (1 - best) * (s_TAN[te] - mu_t) / sd_t + best * (s_ALFF[te] - mu_a) / sd_a
assert np.isfinite(fused).all()
auc_fused = float(roc_auc_score(y, fused))
def boot(oa, ob):
    rng = np.random.default_rng(C.SEED); d = []
    while len(d) < BOOT_N:
        i = rng.integers(0, 954, 954)
        if len(np.unique(y[i])) < 2: continue
        d.append(roc_auc_score(y[i], oa[i]) - roc_auc_score(y[i], ob[i]))
    d = np.array(d)
    return dict(delta=float(roc_auc_score(y, oa) - roc_auc_score(y, ob)), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), p_boot=float(2 * min((d <= 0).mean(), (d >= 0).mean())))
b5 = boot(fused, s_TAN)
log(f"STEP5 fused AUC {auc_fused:.4f}  w2 per fold {w2_per_fold}  nonzero-w2 folds {sum(w > 0 for w in w2_per_fold)}/5")
log(f"STEP5 fused - s_TAN: {b5['delta']:+.4f} [{b5['lo']:+.4f},{b5['hi']:+.4f}] p={b5['p_boot']:.3f}")

# ------------------------------------------------ STEP 6: interpretability
def top15(per_fold_roi_w):
    W = np.stack(per_fold_roi_w); mean_w = W.mean(0); order = np.argsort(-mean_w)[:15]
    per_fold_top = [set(np.argsort(-w)[:15]) for w in W]
    return [dict(roi=int(i), name=ROI[i], weight=float(mean_w[i]),
                 in_top15_folds=int(sum(i in s for s in per_fold_top))) for i in order]
alff_roi = []
for gs in mdl_alff:
    coef = gs.best_estimator_.named_steps["clf"].coef_.ravel().reshape(90, 3)
    alff_roi.append(np.linalg.norm(coef, axis=1))          # per-ROI norm over 3 bands
tan_roi = []
for gs in mdl_tan:
    coef = np.abs(gs.best_estimator_.named_steps["clf"].coef_.ravel())
    strength = np.zeros(90); np.add.at(strength, iu[0], coef); np.add.at(strength, iu[1], coef)
    tan_roi.append(strength)                               # |w| summed over incident edges
top_alff, top_tan = top15(alff_roi), top15(tan_roi)
ov = sorted({d["roi"] for d in top_alff} & {d["roi"] for d in top_tan})
log("STEP6 ALFF top-15 (fold-averaged |coef| norm over bands; stability = folds in top-15 of 5):")
for d in top_alff: log(f"    {d['roi']:2d} {d['name']:22s} {d['weight']:.4f}  {d['in_top15_folds']}/5")
log("STEP6 TAN top-15 (fold-averaged |coef| summed over incident tangent edges):")
for d in top_tan: log(f"    {d['roi']:2d} {d['name']:22s} {d['weight']:.4f}  {d['in_top15_folds']}/5")
log(f"STEP6 overlap between the two top-15 lists: {len(ov)} ROIs -> {[ROI[i] for i in ov]}")

aj(dict(step1=dict(s_FC=auc_fc, s_TAN=auc_tan, s_ALFF=auc_alff, tan_hp=hp_tan,
                   alff_C=[m.best_params_["clf__C"] for m in mdl_alff]),
        step2=step2, step3=dict(thresholds=thr, table=t, tan_wrong_subgroup=dict(n=int(sub.sum()), alff_auc=sub_auc, alff_acc=sub_acc)),
        step4=step4, step5=dict(fused_auc=auc_fused, w2_per_fold=w2_per_fold, boot_vs_tan=b5),
        step6=dict(alff_top15=top_alff, tan_top15=top_tan, overlap=[ROI[i] for i in ov]),
        log=log_lines), OUT + "ALFF_DIAG.json")
np.savez_compressed(OUT + "ALFF_DIAG_oof.npz", y=y, s_FC=s_FC, s_TAN=s_TAN, s_ALFF=s_ALFF, fused=fused)
shutil.rmtree(MEMDIR, ignore_errors=True); log("COMPLETE")
