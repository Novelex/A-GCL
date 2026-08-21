"""S11 core: authoritative manifest, data gate, X_fc builder, leakage-safe pipeline probe."""
import sys, os, json, hashlib, numpy as np, pandas as pd
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C7
S11="/users/3171356m/agcl_audit_s0/s11/"; REPO="/users/3171356m/A-GCL/"
BASE=20260818
DATASET_SHA="312266b23ecf1348ce083cb25d9c5e5a51d5595dab9ce5639875a51c12f1f844"
SPLITS_SHA ="28fed44dc4666066cc0621f329392e58050b39d5ef1371ec5327830518d98916"
ROI_SHA    ="a7632cd95f3e7c245d3af73dd1b358fed88d52186891cd197f3407d205e4e989"
EXCLUDED={"CMU_b_0050669","Leuven_1_0050706"}
IU=np.triu_indices(90,k=1)                       # ONE deterministic pair ordering (k=1)

def sha(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()

def build_manifest():
    """One explicit row per subject; exact FILE_ID lookup, NO substring/glob matching."""
    m5=pd.read_csv("/users/3171356m/agcl_audit_s0/s5/subject_id_map.csv")   # frozen S5/S10 order
    s1=pd.read_csv("/users/3171356m/agcl_audit_s0/s1_audit_table.csv").set_index("subject_id")
    rows=[]
    for r in m5.itertuples():
        cls="ASD" if r.dx_storage=="ASD" else "NC"
        rows.append(dict(row_index=int(r.subject_id), FILE_ID=r.FILE_ID, y=int(r.y),
            dx=r.dx_storage, dx_upstream=int(r.DX_GROUP_upstream),
            fc_path=f"{REPO}data/raw/{cls}_ADJ/{r.FILE_ID}_adj.mat",
            m1_source="X_sources.npz[M1] via exact FILE_ID index",
            graph_cache_entry=int(r.subject_id), roi_count=90))
    df=pd.DataFrame(rows)
    # hard assertions
    assert len(df)==954 and df.FILE_ID.nunique()==954 and df.fc_path.nunique()==954
    assert (df.y==1).sum()==455 and (df.y==0).sum()==499
    assert not (set(df.FILE_ID)&EXCLUDED), "excluded subject present"
    assert all(os.path.exists(p) for p in df.fc_path), "missing FC file"
    for r in df.itertuples():                     # labels == frozen S1 mapping
        assert s1.dx_storage[ list(s1.index).index(r.FILE_ID) if False else r.FILE_ID ]==r.dx if False else True
    s1m=dict(zip(s1.index,s1.dx_storage))
    assert all(s1m[r.FILE_ID]==r.dx for r in df.itertuples()), "label != frozen S1"
    return df

def verify_frozen_hashes():
    assert sha("/users/3171356m/agcl_audit_s0/s5/M1_B/processed/M1_B_v1.pt")==DATASET_SHA, "FAIL dataset sha"
    assert sha("/users/3171356m/agcl_audit_s0/s3c/splits.json")==SPLITS_SHA, "FAIL splits sha"
    assert sha("/users/3171356m/agcl_audit_s0/s1_roi_alignment_90.csv")==ROI_SHA, "FAIL ROI manifest sha"

def load_Xfc():
    z=np.load(S11+"manifest/X_fc.npz",allow_pickle=True)
    meta=json.load(open(S11+"manifest/GATE_HASHES.json"))
    a=hashlib.sha256(np.ascontiguousarray(z["X_fc"]).tobytes()).hexdigest()
    assert a==meta["X_fc_sha256"], "X_fc drift"
    return z["X_fc"], z["y"].astype(int), [str(s) for s in z["ids"]], meta

# ---------------- leakage-safe pipeline probe (mirrors s75/S5.5 exactly) ----------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
class PLSScores(BaseEstimator, TransformerMixin):
    """PLS-DA transformer returning X-scores ONLY (PLSRegression.fit_transform inside a
    Pipeline returns the (x_scores, y_scores) TUPLE, which breaks downstream steps)."""
    def __init__(self, n_components=2): self.n_components=n_components
    def fit(self, X, y):
        self.pls_=PLSRegression(n_components=self.n_components, scale=False).fit(X, y.astype(float))
        return self
    def transform(self, X): return self.pls_.transform(X)
    def fit_transform(self, X, y=None): return self.fit(X,y).transform(X)
GRID={"clf__C":[1e-4,1e-3,1e-2,1e-1,1,10,100]}
def probe_pipe(X, y, folds, mid_steps, boot=2000, seed=BASE, njobs=None):
    """Pipeline = [scaler] + mid_steps + [LinearSVC], ALL fitted inside inner CV folds.
    mid_steps: list of (name, estimator) — PCA/RP/PLS live HERE, never pre-fitted."""
    nj=int(os.environ.get("S11_NJOBS","1")) if njobs is None else njobs
    oof=np.full(len(y),np.nan); fa=[]
    for tr,te in folds:
        pipe=Pipeline([("sc",StandardScaler())]+[ (n,__import__("copy").deepcopy(e)) for n,e in mid_steps ]
                      +[("clf",LinearSVC(dual="auto",max_iter=20000,random_state=seed))])
        nmin=int(np.bincount(y[tr]).min())
        gs=GridSearchCV(pipe,GRID,cv=StratifiedKFold(min(5,max(2,nmin)),shuffle=True,random_state=seed),
                        scoring="roc_auc",n_jobs=nj)
        gs.fit(X[tr],y[tr]); s=gs.decision_function(X[te]); oof[te]=s
        if len(np.unique(y[te]))>1: fa.append(roc_auc_score(y[te],s))
    cov=np.isfinite(oof)                              # smoke may cover a subset of folds
    yc,oc=y[cov],oof[cov]
    yh=(oc>0).astype(int); tn,fp,fn,tp=confusion_matrix(yc,yh,labels=[0,1]).ravel()
    rng=np.random.default_rng(seed); bs=[]
    for _ in range(boot):
        j=rng.integers(0,len(yc),len(yc))
        if len(np.unique(yc[j]))>1: bs.append(roc_auc_score(yc[j],oc[j]))
    return dict(auc=float(roc_auc_score(yc,oc)),bacc=float(balanced_accuracy_score(yc,yh)),
        acc=float(accuracy_score(yc,yh)),sens=float(tp/(tp+fn)),spec=float(tn/(tn+fp)),
        fold_auc=[float(a) for a in fa],fold_sd=float(np.std(fa,ddof=1)) if len(fa)>1 else 0.0,
        ci_lo=float(np.percentile(bs,2.5)),ci_hi=float(np.percentile(bs,97.5))), oof

def folds_ordinary():
    F,_=C7.splits(); return F
def folds_loso(y):
    meta=pd.read_csv("/users/3171356m/agcl_audit_s0/s3c/meta.csv"); site=meta.site.values
    return [(np.where(site!=s)[0],np.where(site==s)[0]) for s in sorted(set(site))
            if len(np.unique(y[np.where(site==s)[0]]))>1]
