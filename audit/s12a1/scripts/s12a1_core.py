"""S12A1 core. ONE intervention: X_id = [M1_B | I90]. Corrected-C MAIN encoder only.
History applied: S11 manifest = sole order authority; S6 conv.propagate Q1 method;
eval()+no_grad extraction; frozen S3C/S11 splits via s11_core.probe_pipe."""
import sys, os, json, hashlib, numpy as np, pandas as pd, torch
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7");  import s7_core as C7
sys.path.insert(0,"/users/3171356m/A-GCL")
from unsupervised.encoder.tu_encoder import TUEncoder
from unsupervised.learning.ginfominmax import GInfoMinMax
A1="/users/3171356m/agcl_audit_s0/s12a1/"; BASE=20260818
CFG_C=dict(emb_dim=32,num_gc_layers=2,normalize_nodes=True,message_relu=True,
           post_bn_relu=True,drop_ratio=0.3,pooling_type="standard")   # corrected-C (S9/S10 path)

def load_gate():
    """Authoritative S11 load: manifest order, exact hashes, all-954 checks."""
    K.verify_frozen_hashes()
    gh=json.load(open(K.S11+"manifest/GATE_HASHES.json"))
    man=K.S11+"manifest/subject_manifest.csv"
    assert K.sha(man)==gh["manifest_sha256"], "S11 manifest drift"
    df=pd.read_csv(man)
    X_fc,y,ids,meta=K.load_Xfc()                            # asserts X_fc sha
    assert list(df.FILE_ID)==ids and np.array_equal(df.y.values,y)
    return df, X_fc, y, ids, gh

def load_tensors(df):
    """FC [954,90,90] from canonical source paths; x from frozen S5 cache at s5_cache_index;
    M1_B canonical from X_sources. Exact-ID lookup only."""
    import scipy.io as sio
    from torch_geometric.data import InMemoryDataset
    class Cache(InMemoryDataset):
        def __init__(s,root,fn): s._fn=fn; super().__init__(root); s.data,s.slices=torch.load(s.processed_paths[0],weights_only=False)
        @property
        def processed_file_names(s): return s._fn
        @property
        def raw_file_names(s): return []
        def download(s): pass
        def process(s): raise RuntimeError("no rebuild")
    ds=Cache("/users/3171356m/agcl_audit_s0/s5/M1_B","M1_B_v1.pt"); assert len(ds)==954
    Z=np.load("/users/3171356m/agcl_audit_s0/s3c/X_sources.npz",allow_pickle=True)
    xid=[str(s) for s in Z["ids"]]; M1=Z["M1"].astype(np.float64)
    mn=M1.min((1,2),keepdims=True); mx=M1.max((1,2),keepdims=True); M1B=((M1-mn)/(mx-mn)).astype(np.float32)
    FC=np.zeros((954,90,90)); Xold=np.zeros((954,90,3),dtype=np.float32); parents=set()
    stats=dict(fc_graph_max=0.0, x_max=0.0, sym=0.0, diag=0.0, mism=0)
    for r in df.itertuples():
        i=r.row_index
        M=sio.loadmat(r.fc_path)["cropped_matrix"]; parents.add(os.path.dirname(r.fc_path))
        assert M.shape==(90,90) and np.isfinite(M).all()
        stats["sym"]=max(stats["sym"],float(np.abs(M-M.T).max()))
        stats["diag"]=max(stats["diag"],float(np.abs(np.diag(M)-1).max()))
        FC[i]=M
        g=ds[int(r.s5_cache_index)]
        assert int(g.subject_id)==int(r.s5_cache_index) and int(g.y)==r.y
        assert g.x.shape==(90,3) and torch.isfinite(g.x).all()
        d=float(np.abs(g.edge_weight.numpy().reshape(90,90)-M).max())
        stats["fc_graph_max"]=max(stats["fc_graph_max"],d)
        if d>1e-6: stats["mism"]+=1
        kx=xid.index(r.FILE_ID)
        dx=float(np.abs(g.x.numpy()-M1B[kx]).max()); stats["x_max"]=max(stats["x_max"],dx)
        assert dx==0.0, f"x != canonical M1_B for {r.FILE_ID}"
        Xold[i]=g.x.numpy()
    stats["parents"]=sorted(parents)
    return FC, Xold, stats

def build_encoder(n_feat, seed):
    torch.manual_seed(seed); np.random.seed(seed % (2**32))
    enc=TUEncoder(num_dataset_features=n_feat, emb_dim=CFG_C["emb_dim"],
        num_gc_layers=CFG_C["num_gc_layers"], drop_ratio=CFG_C["drop_ratio"],
        pooling_type=CFG_C["pooling_type"], is_infograph=False,
        normalize_nodes=CFG_C["normalize_nodes"], message_relu=CFG_C["message_relu"],
        post_bn_relu=CFG_C["post_bn_relu"])
    m=GInfoMinMax(enc, CFG_C["emb_dim"])          # positional: real call-site contract (S8 lesson)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m

def sd_hash(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items(): h.update(k.encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()[:16]

def edge_struct(device="cpu"):
    n=torch.arange(90,device=device)
    return torch.stack([n.repeat_interleave(90), n.repeat(90)],0)

@torch.no_grad()
def stages_for_subject_batch(model, Xn, FCb, want_z=True):
    """Xn [B,90,F] float32, FCb [B,90,90]. Returns per-stage numpy arrays.
    eval() + no_grad throughout (dropout inert, BN uses init running stats)."""
    B,F=Xn.shape[0],Xn.shape[2]
    ei0=edge_struct(); xs=[];eis=[];ews=[];bt=[]
    for b in range(B):
        xs.append(torch.from_numpy(Xn[b])); eis.append(ei0+b*90)
        ews.append(torch.from_numpy(FCb[b].reshape(-1).astype(np.float32)))
        bt.append(torch.full((90,),b,dtype=torch.long))
    x=torch.cat(xs); ei=torch.cat(eis,1); ew=torch.cat(ews); batch=torch.cat(bt)
    assert ew is not None and torch.isfinite(ew).all()
    enc=model.encoder; acts={}
    hooks=[enc.convs[0].register_forward_hook(lambda m,i,o: acts.setdefault("H1_preBN",o.detach())),
           enc.bns[0].register_forward_hook(lambda m,i,o: acts.setdefault("H1_BN",o.detach())),
           enc.convs[1].register_forward_hook(lambda m,i,o: acts.setdefault("H2_preBN",o.detach())),
           enc.bns[1].register_forward_hook(lambda m,i,o: acts.setdefault("H2_BN",o.detach()))]
    h,z,node=model.encode(batch,x,ei,None,ew)
    for hk in hooks: hk.remove()
    # Q1 via the ACTUAL conv's MessagePassing op (S6 method) -- not an assumed formula
    conv=enc.convs[0]
    q=conv.propagate(ei, x=(x,x), edge_weight=ew, size=None)
    q=q+(1+conv.eps)*x
    H1_to_l2=torch.relu(acts["H1_BN"]) if enc.post_bn_relu else acts["H1_BN"]  # dropout inert in eval
    out=dict(Q1=q, H1_preBN=acts["H1_preBN"], H1_BN=acts["H1_BN"], H1_to_layer2=H1_to_l2,
             H2_preBN=acts["H2_preBN"], H2_BN=acts["H2_BN"],
             final_prenorm=acts["H2_BN"], final_postnorm=node, h=h)
    if want_z: out["z"]=z
    return {k:(v.numpy().reshape(B,90,-1) if k not in ("h","z") else v.numpy()) for k,v in out.items()}
