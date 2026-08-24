"""S16 shared feature/eval helpers: arm definitions, probe_honest (C2c), fusion (C4)."""
import sys, numpy as np
sys.path.insert(0, "/users/3171356m/A-GCL/audit/s16/scripts"); import s16_data as DAT
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
from sklearn.model_selection import train_test_split

# arm -> (arch, node spec). E acts on FC EVERYWHERE it appears (decision Q1).
ARMS = {"A1": ("WGIN","alff"), "A3": ("WGIN","fcrow"), "A4": ("WGIN","fcrow+alff"),
        "A5": ("BNT","fcrow"),  "A6": ("BNT","fcrow+alff"),
        # A7 = the S12A5 arm C edge MLP. Its input is the FC UPPER TRIANGLE, taken
        # from the E-TRANSFORMED matrix so that E is meaningful for this arm too.
        # At E=signed the triangle is bitwise the frozen X_fc, giving parity with
        # S12A5 arm C.
        "A7": ("EDGEMLP","edgetri")}

def edge_triangle(FCt):
    """[954,4005] upper triangle (k=1) of the E-transformed FC, in K.IU order."""
    return FCt[:, K.IU[0], K.IU[1]].astype(np.float32)

def alff_scaled(ALFF, tr, mode="z"):
    A = ALFF.astype(np.float64)
    if mode=="z":
        mu,sd = A[tr].mean(0,keepdims=True), A[tr].std(0,keepdims=True)
        return ((A-mu)/np.maximum(sd,1e-6)).astype(np.float32)
    if mode=="perband":
        mn,mx = A.min(1,keepdims=True), A.max(1,keepdims=True); sp = mx-mn
        return np.where(sp>0,(A-mn)/np.where(sp>0,sp,1.0),A).astype(np.float32)
    if mode=="joint":
        mn,mx = A.min((1,2),keepdims=True), A.max((1,2),keepdims=True)
        return ((A-mn)/np.maximum(mx-mn,1e-12)).astype(np.float32)
    if mode=="raw": return A.astype(np.float32)
    raise ValueError(mode)

def build_X(spec, FCt, ALFF, tr, control=None, alff_mode="z"):
    """FCt is ALREADY E-transformed. Controls act on the feature pathway."""
    R = FCt.astype(np.float32)
    if control=="C-SHUF":
        Rs = np.empty_like(R)
        for s in range(len(R)):
            Rs[s] = R[s][:, np.random.default_rng(DAT.BASE+s).permutation(90)]
        R = Rs
    A = alff_scaled(ALFF, tr, alff_mode)
    I90 = np.repeat(np.eye(90,dtype=np.float32)[None], len(R), 0)
    if spec=="edgetri":                      # A7: 2-D [954,4005], no node axis
        return edge_triangle(R), FCt
    X = {"alff":A, "fcrow":R, "fcrow+alff":np.concatenate([R,A],2),
         "alff+onehot":np.concatenate([A,I90],2)}[spec]
    FCu = FCt
    if control=="C-ROI":
        Xs = np.empty_like(X); Fs = np.empty_like(FCt)
        for s in range(len(X)):
            p = np.random.default_rng(DAT.BASE+7000+s).permutation(90)
            Xs[s] = X[s][p]; Fs[s] = FCt[s][p][:,p]
        X, FCu = Xs, Fs
    return X, FCu

# ---------------------------------------------------------------- C2c probe_honest
def honest_split(tr, y, seed=DAT.BASE):
    """tr -> (tr_enc 80%, tr_probe 20%), stratified by label. Encoder sees tr_enc
    ONLY; probe fits on tr_probe. Both sides of the probe are then out-of-sample."""
    a, b = train_test_split(np.arange(len(tr)), test_size=0.20,
                            stratify=y[tr], random_state=seed)
    return np.asarray(tr)[a], np.asarray(tr)[b]

def probe_honest(R, y, tr_probe, te):
    """The frozen ruler (K.probe_pipe, unchanged) fitted on tr_probe, scored on te."""
    d, oof = K.probe_pipe(np.asarray(R, dtype=np.float64), y,
                          [(np.asarray(tr_probe), np.asarray(te))], [])
    return d, oof

# ------------------------------------------------- SCORE-LEVEL FUSION (item 3)
# NOTE: this REPLACES the earlier feature-concatenation fusion. s_FC and s_learned
# live on different scales, so an unstandardised alpha would measure scale, not
# information. Both are z-scored using mean/sd from the INNER VALIDATION SPLIT ONLY.
ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 4)

def zfit(v, idx):
    """mean/sd from idx (the inner validation split) only."""
    mu = float(np.mean(v[idx])); sd = float(np.std(v[idx]))
    return mu, (sd if sd > 1e-12 else 1.0)

def zapply(v, mu, sd): return (np.asarray(v, dtype=np.float64) - mu) / sd

def fuse_scores(s_fc, s_learned, alpha, inner_idx):
    """alpha*z(s_FC) + (1-alpha)*z(s_learned). At alpha=1.0 the result is z(s_FC),
    a strictly increasing transform of s_FC, so the AUC equals the FC-only AUC
    EXACTLY and the ranking is bitwise identical."""
    mf, sf = zfit(s_fc, inner_idx); ml, sl = zfit(s_learned, inner_idx)
    return alpha * zapply(s_fc, mf, sf) + (1.0 - alpha) * zapply(s_learned, ml, sl)

def scores_for_fusion(R, Xfc, y, tr_enc, tr_prb, te):
    """Produce s_FC and s_learned DEFINED ON BOTH tr_prb AND te, both OUT-OF-SAMPLE.

    Why this is not trivial: K.probe_pipe returns an OOF vector that is NaN outside
    the fold it scored, and the learned probe is FITTED on tr_prb — so its scores on
    tr_prb would be IN-SAMPLE. In-sample scores have inflated spread, which would
    shrink the learned block's z-scored magnitude and bias alpha toward FC. Both
    sides are therefore made out-of-sample on tr_prb:
      s_FC       : SVM fitted on tr_enc -> tr_prb and te are both unseen. Clean.
      s_learned  : 2-fold cross-fit WITHIN tr_prb for the tr_prb values; fitted on
                   the whole of tr_prb for the te values.
    Returns (s_fc, s_learned), each a length-954 array finite on tr_prb and te."""
    from sklearn.model_selection import StratifiedKFold
    n = len(y)
    s_fc = np.full(n, np.nan); s_le = np.full(n, np.nan)
    both = np.concatenate([np.asarray(tr_prb), np.asarray(te)])
    _, o = K.probe_pipe(np.asarray(Xfc, dtype=np.float64), y, [(tr_enc, both)], [])
    s_fc[both] = o[both]
    _, ol = K.probe_pipe(np.asarray(R, dtype=np.float64), y,
                         [(np.asarray(tr_prb), np.asarray(te))], [])
    s_le[np.asarray(te)] = ol[np.asarray(te)]
    tp = np.asarray(tr_prb)
    for a, b in StratifiedKFold(2, shuffle=True, random_state=20260818).split(
            np.zeros(len(tp)), y[tp]):
        _, oi = K.probe_pipe(np.asarray(R, dtype=np.float64), y, [(tp[a], tp[b])], [])
        s_le[tp[b]] = oi[tp[b]]
    return s_fc, s_le

def stack_scores(s_fc, s_learned, y, inner_idx, score_idx):
    """Stacking variant: logistic regression on [s_FC, s_learned], fitted on the
    INNER SPLIT ONLY, scored on score_idx."""
    from sklearn.linear_model import LogisticRegression
    mf, sf = zfit(s_fc, inner_idx); ml, sl = zfit(s_learned, inner_idx)
    Z = np.column_stack([zapply(s_fc, mf, sf), zapply(s_learned, ml, sl)])
    lr = LogisticRegression(max_iter=5000).fit(Z[inner_idx], y[inner_idx])
    return lr.decision_function(Z[score_idx]), lr.coef_[0].tolist()
