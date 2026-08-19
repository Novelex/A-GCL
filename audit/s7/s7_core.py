"""S7 shared core. Read-only w.r.t. production code/data. NO TRAINING ANYWHERE."""
import os, sys, json, hashlib, tempfile, socket, subprocess, time, datetime
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/users/3171356m/A-GCL")
from unsupervised.encoder.tu_encoder import TUEncoder
from unsupervised.learning.ginfominmax import GInfoMinMax

REPO="/users/3171356m/A-GCL"; S7="/users/3171356m/agcl_audit_s0/s7/"
S3C="/users/3171356m/agcl_audit_s0/s3c/"
BASE_SEED=20260818
COHORT_SHA="aca3d945f7d89ccb1cc6fb46ca01f6036ccf036d81d120d2b09fe4bd0d1d68c9"
XSRC_SHA="dc10bf36c4124aa7f214ab6bbc5a89288adc03da747e3950485bb15c0da333a9"

# ---- P/O/C architecture ledger (proven in the S7 ledger step) ----
ARCH={
 "P":dict(normalize_nodes=False, message_relu=False, post_bn_relu=False, drop_ratio=0.0,
          note="paper_exact profile / paper-literal Sec 2.2.1"),
 "O":dict(normalize_nodes=True,  message_relu=True,  post_bn_relu=True,  drop_ratio=0.0,
          note="original qbmizsj hardcoded behaviour; dropout inactive in eval"),
 "C":dict(normalize_nodes=True,  message_relu=True,  post_bn_relu=True,  drop_ratio=0.3,
          note="current fork 'corrected' argparse defaults"),
}
BRANCHES=["B","C","D"]

def git_head():
    return subprocess.run(["git","-C",REPO,"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
def git_dirty():
    return subprocess.run(["git","-C",REPO,"status","--short"],capture_output=True,text=True).stdout.strip()

def provenance(extra=None):
    d=dict(host=socket.gethostname(), time=datetime.datetime.now().isoformat(timespec="seconds"),
           slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array=os.environ.get("SLURM_ARRAY_TASK_ID"),
           git_head=git_head(), git_dirty=git_dirty(), python=sys.version.split()[0],
           torch=torch.__version__, cuda_available=torch.cuda.is_available(),
           device_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
           base_seed=BASE_SEED)
    if extra: d.update(extra)
    return d

# ---------------- data ----------------
_CACHE={}
def load_all():
    if _CACHE: return _CACHE
    coh=pd.read_csv("/users/3171356m/agcl_audit_s0/s1_audit_table.csv")
    ids=coh.subject_id.tolist()
    assert hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()==COHORT_SHA, "COHORT DRIFT"
    assert hashlib.sha256(open(S3C+"X_sources.npz","rb").read()).hexdigest()==XSRC_SHA, "X_sources DRIFT"
    Z=np.load(S3C+"X_sources.npz",allow_pickle=True)
    assert [str(s) for s in Z["ids"]]==ids
    RAW=Z["M1"].astype(np.float64); y=Z["y"].astype(int)
    import scipy.io as sio
    dx=dict(zip(coh.subject_id,coh.dx_storage))
    FC=np.stack([sio.loadmat(f"{REPO}/data/raw/{'ASD' if dx[s]=='ASD' else 'NC'}_ADJ/{s}_adj.mat")
                 ["cropped_matrix"] for s in ids]).astype(np.float64)
    def nb(X,k):
        if k=="B": mn=X.min((1,2),keepdims=True); mx=X.max((1,2),keepdims=True); return (X-mn)/(mx-mn)
        if k=="C": mn=X.min(1,keepdims=True);    mx=X.max(1,keepdims=True);    return (X-mn)/(mx-mn)
        if k=="D": return (X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)
    _CACHE.update(ids=ids,y=y,RAW=RAW,FC=FC,NODE={k:nb(RAW,k) for k in BRANCHES},
                  meta=pd.read_csv(S3C+"meta.csv"))
    return _CACHE

def splits():
    S=json.load(open(S3C+"splits.json"))
    return [(np.array(f["train"]),np.array(f["test"])) for f in S["outer_folds"]], S["spec"]

N_NODES=90
def edge_index(device="cpu"):
    n=torch.arange(N_NODES,device=device)
    return torch.stack([n.repeat_interleave(N_NODES), n.repeat(N_NODES)],0)

def batch_graphs(idx, branch, device="cpu", dtype=torch.float32):
    """Build a PyG-style batched forward input for the given subject indices."""
    D=load_all(); X=D["NODE"][branch]; FC=D["FC"]
    xs=[]; eis=[]; ews=[]; bt=[]
    ei0=edge_index("cpu")
    for bi,i in enumerate(idx):
        xs.append(torch.tensor(X[i],dtype=dtype))
        eis.append(ei0+bi*N_NODES)
        ews.append(torch.tensor(FC[i].reshape(-1),dtype=dtype))
        bt.append(torch.full((N_NODES,),bi,dtype=torch.long))
    return (torch.cat(xs).to(device), torch.cat(eis,1).to(device),
            torch.cat(ews).to(device), torch.cat(bt).to(device))

# ---------------- model construction (uses PRODUCTION classes, unmodified) ----------------
def build_model(path, seed, emb_dim=32, num_gc_layers=2, proj_hidden_dim=None, device="cpu"):
    """Instantiate EXACTLY as the real A-GCL callers do:
         agcl_ABIDE.py:98-102 and agcl_ABIDE_queue.py:229-233  ->  GInfoMinMax(TUEncoder(...), args.emb_dim)
         original bed5441:A-GCL/adgcl_edge.py:50-53            ->  GInfoMinMax(TUEncoder(...), args.emb_dim)
       with --emb_dim default 32. The class default proj_hidden_dim=300 is a STANDALONE
       default that no A-GCL training script ever uses; relying on it was an audit-side
       constructor error (corrected 2026-08-19)."""
    if proj_hidden_dim is None: proj_hidden_dim = emb_dim      # == args.emb_dim at the real call site
    a=ARCH[path]
    torch.manual_seed(BASE_SEED+seed); np.random.seed(BASE_SEED+seed)
    enc=TUEncoder(num_dataset_features=3, emb_dim=emb_dim, num_gc_layers=num_gc_layers,
                  drop_ratio=a["drop_ratio"], pooling_type="standard", is_infograph=False,
                  normalize_nodes=a["normalize_nodes"], message_relu=a["message_relu"],
                  post_bn_relu=a["post_bn_relu"])
    m=GInfoMinMax(enc, proj_hidden_dim).to(device)          # POSITIONAL, mirroring the real call site
    ph=[l for l in m.proj_head if hasattr(l,"in_features")]
    assert ph[0].in_features==emb_dim and ph[0].out_features==proj_hidden_dim, "proj_head[0] contract"
    assert ph[-1].in_features==proj_hidden_dim and ph[-1].out_features==proj_hidden_dim, "proj_head[-1] contract"
    m.eval()                      # NEVER train. eval() also disables dropout + BN updates.
    for p in m.parameters(): p.requires_grad_(False)
    return m

# ---------------- atomic IO with validation + DONE sentinel ----------------
def _validate(obj):
    def chk(a):
        arr=np.asarray(a)
        if arr.dtype.kind in "fc" and arr.size and not np.isfinite(arr).all(): return False
        return True
    if isinstance(obj,dict):
        return all(chk(v) for v in obj.values() if isinstance(v,(np.ndarray,list,float,int)))
    return chk(obj)

def write_unit(outdir, name, payload_npz=None, payload_json=None):
    """TEMP -> validate -> atomic rename -> reopen -> validate -> DONE last."""
    os.makedirs(outdir,exist_ok=True)
    finals=[]
    if payload_npz is not None:
        if not _validate(payload_npz): raise ValueError(f"{name}: non-finite in payload")
        # NOTE: np.savez_compressed appends ".npz" unless the path already ends in it,
        # so the temp path MUST end in .npz or the atomic rename moves an empty file.
        tmp=os.path.join(outdir,f".{name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp,**payload_npz)
        with np.load(tmp,allow_pickle=True) as _z:      # validate TEMP before promoting
            for _k in _z.files: _ = _z[_k]
        os.replace(tmp,os.path.join(outdir,name+".npz"))
        finals.append(os.path.join(outdir,name+".npz"))
    if payload_json is not None:
        tmp=os.path.join(outdir,f".{name}.{os.getpid()}.tmp.json")
        with open(tmp,"w") as f: json.dump(payload_json,f,indent=1,default=str)
        json.load(open(tmp))                            # validate TEMP before promoting
        os.replace(tmp,os.path.join(outdir,name+".json"))
        finals.append(os.path.join(outdir,name+".json"))
    for p in finals:                                    # reopen + validate FINAL
        if p.endswith(".npz"):
            with np.load(p,allow_pickle=True) as z:
                for k in z.files:
                    a=z[k]
                    if a.dtype.kind in "fc" and a.size and not np.isfinite(a).all():
                        raise ValueError(f"{p}: non-finite after reopen")
        else:
            json.load(open(p))
    open(os.path.join(outdir,name+".DONE"),"w").write(git_head()+"\n")   # DONE LAST
    return finals

def is_done(outdir,name):
    d=os.path.join(outdir,name+".DONE")
    if not os.path.exists(d): return False
    for ext in (".npz",".json"):
        p=os.path.join(outdir,name+ext)
        if os.path.exists(p):
            try:
                if ext==".npz":
                    with np.load(p,allow_pickle=True) as z: [z[k] for k in z.files]
                else: json.load(open(p))
            except Exception: return False
    return True

# ---------------- geometry metrics ----------------
def geom(M):
    M=np.asarray(M,dtype=np.float64)
    if M.ndim!=2 or M.shape[0]<2: return {}
    C=M-M.mean(0,keepdims=True)
    s=np.linalg.svd(C,compute_uv=False); s=s[s>1e-12]
    er=float(np.exp(-((s/s.sum())*np.log(s/s.sum())).sum())) if len(s) else 0.0
    # pairwise metrics are O(n^2); subsample deterministically above MAXP rows so this
    # stays tractable for large stacks (e.g. 954*90 = 85,860 rows in J1A).
    MAXP=1500
    if len(M)>MAXP:
        sel=np.linspace(0,len(M)-1,MAXP).astype(int); P=M[sel]; sub=True
    else:
        P=M; sub=False
    Nn=P/np.clip(np.linalg.norm(P,axis=1,keepdims=True),1e-12,None)
    K=Nn@Nn.T; iu=np.triu_indices(len(P),1)
    d=np.linalg.norm(P[:,None,:]-P[None,:,:],axis=2)
    return dict(var=float(M.var(0).mean()), mean_cos=float(K[iu].mean()),
                eff_rank=er, frac_var_sv1=float((s[0]**2)/(s**2).sum()) if len(s) else 0.0,
                mean_pair_dist=float(d[iu].mean()), mean_norm=float(np.linalg.norm(M,axis=1).mean()),
                pairwise_subsampled=sub, n_pairwise=len(P),
                sv_top5=[float(x) for x in s[:5]])
