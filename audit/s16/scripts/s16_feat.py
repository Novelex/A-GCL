"""S16 shared feature/eval helpers: arm definitions, probe_honest (C2c), fusion (C4)."""
import sys, numpy as np
sys.path.insert(0, "/users/3171356m/A-GCL/audit/s16/scripts"); import s16_data as DAT
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
from sklearn.model_selection import train_test_split

# arm -> (arch, node spec). E acts on FC EVERYWHERE it appears (decision Q1).
ARMS = {"A1": ("WGIN","alff"), "A3": ("WGIN","fcrow"), "A4": ("WGIN","fcrow+alff"),
        "A5": ("BNT","fcrow"),  "A6": ("BNT","fcrow+alff")}

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

# ---------------------------------------------------------------- C4 fusion
def fuse(R_learned, X_fc_raw):
    """repr = concat( raw FC edges (4005) , learned(d) ). The floor is 0.7565 BY
    CONSTRUCTION: zeroing the learned block leaves exactly the SVM's input.
    StandardScaler standardises EACH COLUMN independently, so the two blocks'
    different scales cannot distort one another and the FC block is reproduced
    column-for-column as the SVM would see it."""
    return np.concatenate([np.asarray(X_fc_raw, dtype=np.float64),
                           np.asarray(R_learned, dtype=np.float64)], axis=1)
