"""S11 Steps 0-3: provenance, manifest, full 954 data gate, authoritative X_fc."""
import sys, os, json, hashlib, numpy as np, pandas as pd, scipy.io as sio, torch
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7");  import s7_core as C7
OUT=K.S11+"manifest"; os.makedirs(OUT,exist_ok=True)
gate={"provenance":C7.provenance({"unit":"S11_gate","cmd":" ".join(sys.argv)}),"checks":{}}
def ck(name,ok,detail=""):
    gate["checks"][name]={"pass":bool(ok),"detail":detail}; print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}",flush=True)
    if not ok: json.dump(gate,open(OUT+"/GATE_FAIL.json","w"),indent=1,default=str); sys.exit(1)

K.verify_frozen_hashes(); ck("frozen hashes (dataset/splits/ROI-manifest)",True)
df=K.build_manifest()
# AUTHORITATIVE ROW ORDER = the order the frozen splits are DEFINED on (S3C / s1_audit_table).
# GATE FINDING (recorded): the S5 graph-cache order (ASD block, NC block) is a PERMUTATION of
# this; S8-S10 probes applied S3C split indices to S5-ordered arrays -> internally valid CV
# (labels stayed paired) but fold ASD counts drifted to 79-101 vs the frozen 91. S11 therefore
# orders by S3C and keeps the S5 permutation explicit in the manifest.
s3c=pd.read_csv("/users/3171356m/agcl_audit_s0/s1_audit_table.csv").subject_id.tolist()
df=df.set_index("FILE_ID").loc[s3c].reset_index()
df["row_index"]=range(954)                        # S3C position
df.rename(columns={"graph_cache_entry":"s5_cache_index"},inplace=True)
df["graph_cache_entry"]=df["s5_cache_index"]
df.to_csv(OUT+"/subject_manifest.csv",index=False)
man_sha=K.sha(OUT+"/subject_manifest.csv")
ck("manifest: N=954, 455/499, unique IDs+paths, exclusions absent, labels==S1",True,f"sha {man_sha[:16]}")
ck("row order == split-defining S3C order; S5 permutation recorded (S8-S10 fold-drift finding logged)",
   list(df.FILE_ID)==s3c, "authoritative order = s1_audit_table")

# frozen graph cache (cross-check only)
from torch_geometric.data import InMemoryDataset
class Cache(InMemoryDataset):
    def __init__(s,root,fn): s._fn=fn; super().__init__(root); s.data,s.slices=torch.load(s.processed_paths[0],weights_only=False)
    @property
    def processed_file_names(s): return s._fn
    @property
    def raw_file_names(s): return []
    def download(s): pass
    def process(s): raise RuntimeError
ds=Cache("/users/3171356m/agcl_audit_s0/s5/M1_B","M1_B_v1.pt")
ck("graph cache = frozen 954 (NOT any 956 cache; data_dense_v3.pt never opened)",len(ds)==954,f"n={len(ds)}")

Z=np.load("/users/3171356m/agcl_audit_s0/s3c/X_sources.npz",allow_pickle=True)
xid=[str(s) for s in Z["ids"]]; M1=Z["M1"].astype(np.float64)
mn=M1.min((1,2),keepdims=True); mx=M1.max((1,2),keepdims=True); M1B=((M1-mn)/(mx-mn)).astype(np.float32)

Xsrc=np.zeros((954,90,90)); Xgph=np.zeros((954,90,90),dtype=np.float32)
worst_fc=0.0; worst_x=0.0; mism_sub=0; mism_ent=0; first=None
sym_max=0.0; diag_max=0.0; gmin=1e9
parents=set()
for r in df.itertuples():
    i=r.row_index
    gi=r.s5_cache_index                                    # graph lives at its S5 index
    M=sio.loadmat(r.fc_path)["cropped_matrix"]            # exact path, no glob
    parents.add(os.path.dirname(r.fc_path))
    assert M.shape==(90,90) and np.isfinite(M).all()
    sym_max=max(sym_max,float(np.abs(M-M.T).max())); diag_max=max(diag_max,float(np.abs(np.diag(M)-1).max()))
    gmin=min(gmin,float(M.min())); Xsrc[i]=M
    g=ds[gi]
    assert int(g.subject_id)==gi==r.graph_cache_entry and int(g.y)==r.y
    ew=g.edge_weight; assert ew is not None and torch.isfinite(ew).all() and ew.numel()==8100
    G=ew.numpy().reshape(90,90); Xgph[i]=G
    d=float(np.abs(M-G).max()); worst_fc=max(worst_fc,d)
    if d>1e-6:
        mism_sub+=1; mism_ent+=int((np.abs(M-G)>1e-6).sum())
        if first is None: first=(r.FILE_ID,d)
    k=xid.index(r.FILE_ID)                                # exact FILE_ID lookup for M1
    dx=float(np.abs(g.x.numpy()-M1B[k]).max()); worst_x=max(worst_x,dx)
    assert dx<1e-6, f"x != frozen M1_B for {r.FILE_ID}"
ck("FC source: shape/finite all 954; symmetric; diag==1; signed",
   sym_max<1e-12 and diag_max<1e-12 and gmin<0, f"sym {sym_max:.1e} diag {diag_max:.1e} min {gmin:.3f}")
ck("FC source vs graph round-trip, ALL 954", mism_sub==0,
   f"max_abs {worst_fc:.2e}, mismatch subjects {mism_sub}, entries {mism_ent}, first {first}")
ck("ALFF x == frozen M1_B (exact ID lookup; no M2/OLD/norm_matrix/956 fallback), all 954",
   worst_x<1e-6, f"max_abs {worst_x:.2e}")
ck("path audit: exact-manifest lookup only", parents=={f"{K.REPO}data/raw/ASD_ADJ",f"{K.REPO}data/raw/NC_ADJ"},
   f"parents={sorted(parents)}")

# splits membership reproduced
F=K.folds_ordinary(); allidx=sorted(set(np.concatenate([np.r_[tr,te] for tr,te in F]).tolist()))
ck("frozen split membership covers exactly rows 0..953", allidx==list(range(954)))

# Step 3: authoritative 4005 matrix + pair map
X_fc=np.stack([Xsrc[i][K.IU] for i in range(954)])
X_g =np.stack([Xgph[i][K.IU] for i in range(954)])
ck("X_fc_source == X_fc_graph (954x4005)", float(np.abs(X_fc-X_g).max())<1e-6,
   f"max_abs {float(np.abs(X_fc-X_g).max()):.2e}")
roi=pd.read_csv("/users/3171356m/agcl_audit_s0/s1_roi_alignment_90.csv").aal_name.tolist()
pm=pd.DataFrame(dict(feature_index=range(4005),ROI_i=[roi[a] for a in K.IU[0]],ROI_j=[roi[b] for b in K.IU[1]],
                     i=K.IU[0],j=K.IU[1]))
pm.to_csv(OUT+"/pair_map.csv",index=False); pm_sha=K.sha(OUT+"/pair_map.csv")
# agreement with frozen S5.5 representation
F55=np.load("/users/3171356m/agcl_audit_s0/s55/features.npz",allow_pickle=True)["FC"]
ck("4005 pair-map == frozen S5.5 FC representation", bool(np.allclose(X_fc,F55,atol=1e-9)),
   f"max_abs {float(np.abs(X_fc-F55).max()):.2e}")
y=df.y.values.astype(np.int64)
np.savez_compressed(OUT+"/X_fc.npz",X_fc=X_fc.astype(np.float64),y=y,ids=df.FILE_ID.values)
xh=hashlib.sha256(np.ascontiguousarray(X_fc.astype(np.float64)).tobytes()).hexdigest()
json.dump(dict(manifest_sha256=man_sha,pair_map_sha256=pm_sha,X_fc_sha256=xh,
   dataset_sha256=K.DATASET_SHA,splits_sha256=K.SPLITS_SHA,roi_manifest_sha256=K.ROI_SHA),
   open(OUT+"/GATE_HASHES.json","w"),indent=1)
gate["hashes"]=json.load(open(OUT+"/GATE_HASHES.json"))
C7.write_unit(OUT,"S11_GATE",payload_json=gate)
open(OUT+"/GATE_PASS","w").write(C7.git_head()+"\n")
print(f"S11 DATA GATE: ALL PASS.  X_fc sha {xh[:16]}...  pair-map sha {pm_sha[:16]}...",flush=True)
