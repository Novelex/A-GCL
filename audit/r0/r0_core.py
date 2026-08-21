"""R0 audit config: released_code_compat_08339b7. Reuses the parity-verified S8 'O'
training contract; adds the released downstream evaluator (with Phase-1 repairs only)."""
import sys, os, json, hashlib, time, numpy as np
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s8"); import s8_core as S
import s7_core as C7
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
R0="/users/3171356m/agcl_audit_s0/r0/"
CONFIG_NAME="released_code_compat_08339b7"
released_bug_unweighted_eval=True          # released get_embeddings drops edge_weight into edge_attr

# ---- verbatim upstream helpers (08339b7 embedding_evaluation.py:53-79) ----
def sensitivity(y_pred, y_true):
    CM=confusion_matrix(y_true,y_pred)
    tp=CM[1,1]; fn=CM[1,0]
    return tp/(tp+fn+1e-6)
def specificity(y_pred, y_true):
    CM=confusion_matrix(y_true,y_pred)
    tn=CM[0,0]; fp=CM[0,1]
    return tn/(tn+fp+1e-6)

def rng_state_hash():
    s=np.random.get_state()
    return hashlib.sha256(s[1].tobytes()+bytes([s[2]&0xff])).hexdigest()[:16]

def upstream_kf_eval(emb, y, folds=10, flag=False, collect_auc=True):
    """kf_embedding_evaluation @08339b7 with ONLY the Phase-1 repairs.
    random_state=None everywhere -> consumes the ORIGINAL GLOBAL NumPy RNG stream."""
    kf=KFold(n_splits=folds, shuffle=True, random_state=None)          # upstream line 241
    test_id=[]                                                         # REPAIR R2
    running_times=[]
    K={k:[] for k in ("tr","va","te","trf","vaf","tef","trs","vas","tes","trp","vap","tep")}
    fold_hashes=[]; auc_diag=[]
    for k_id,(train_val_index,test_index) in enumerate(kf.split(emb)):
        test_id.append(test_index)
        train_index,val_index=train_test_split(train_val_index,test_size=0.2,random_state=None)
        fold_hashes.append(hashlib.sha256(test_index.tobytes()+train_index.tobytes()).hexdigest()[:12])
        t0=time.time()
        clf=make_pipeline(StandardScaler(),
             GridSearchCV(LinearSVC(dual=False,fit_intercept=True,max_iter=10000),
                          {"C":[0.001,0.01,0.1,1,10,100,1000]},cv=5,scoring="accuracy",
                          n_jobs=int(os.environ.get("R0_NJOBS","16")),verbose=0))
        Xtr,ytr=emb[train_index],y[train_index]
        Xva,yva=emb[val_index],y[val_index]
        Xte,yte=emb[test_index],y[test_index]
        clf.fit(Xtr,np.squeeze(ytr))
        pr={k:clf.predict(v) for k,v in (("tr",Xtr),("va",Xva),("te",Xte))}
        running_time=(time.time()-t0) if flag else 0.0                  # REPAIR R4 (timing-only)
        running_times.append(running_time)
        for tag,yy,pp in (("tr",ytr,pr["tr"]),("va",yva,pr["va"]),("te",yte,pr["te"])):
            K[tag].append(accuracy_score(yy,pp))                       # scorer == accuracy (TUEvaluator)
            K[tag+"f"].append(f1_score(yy,pp))
            K[tag+"s"].append(sensitivity(pp,yy))                      # upstream reversed arg order
            K[tag+"p"].append(specificity(pp,yy))
        if collect_auc:                                                # diagnostic_not_upstream_parity
            sc=clf.decision_function(Xte)
            auc_diag.append(roc_auc_score(yte,sc) if len(np.unique(yte))>1 else np.nan)
        fpr,tpr=np.array([]),np.array([])                              # REPAIR R3: dead slots
    ms=lambda a:(float(np.mean(a)),float(np.std(a)))
    def pack(p): return [*ms(K[p]),*ms(K[p+"f"]),*ms(K[p+"s"]),*ms(K[p+"p"])]
    return dict(train=pack("tr"),val=pack("va"),test=pack("te"),
        fold_hashes=fold_hashes, rng_hash_after=rng_state_hash(),
        auc_diagnostic_not_upstream_parity=float(np.nanmean(auc_diag)) if auc_diag else None,
        auc_diag_folds=[float(a) for a in auc_diag])
