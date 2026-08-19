"""S5: build and verify M1_B / M1_C / M1_D PyG graph caches.
Reads ONLY frozen raw M1 ALFF + frozen FC .mat. Never touches the repo cache."""
import os, sys, json, hashlib, numpy as np, pandas as pd, scipy.io as sio, torch
from torch_geometric.data import Data, InMemoryDataset
A="/users/3171356m/A-GCL/"; S5="/users/3171356m/agcl_audit_s0/s5/"
coh=pd.read_csv("/users/3171356m/agcl_audit_s0/s1_audit_table.csv")

# ---------- frozen raw M1 (hash-verified against S3C) ----------
xh=hashlib.sha256(open("/users/3171356m/agcl_audit_s0/s3c/X_sources.npz","rb").read()).hexdigest()
assert xh=="dc10bf36c4124aa7f214ab6bbc5a89288adc03da747e3950485bb15c0da333a9", "X_sources drift"
Z=np.load("/users/3171356m/agcl_audit_s0/s3c/X_sources.npz",allow_pickle=True)
M1_by_id={str(s):Z["M1"][i].astype(np.float64) for i,s in enumerate(Z["ids"])}
dx_by_id=dict(zip(coh.subject_id,coh.dx_storage))
print(f"frozen M1 verified: {xh[:16]}...  n={len(M1_by_id)}")

# ---------- loader-identical subject ordering: ASD block then NC block ----------
def block(cls):
    d=f"{A}data/raw/{cls}_ADJ"
    files=sorted(os.listdir(d))
    return [f[:-len('_adj.mat')] for f in files if f.endswith('_adj.mat')]
asd=[s for s in block("ASD") if s in M1_by_id]
nc =[s for s in block("NC")  if s in M1_by_id]
order=asd+nc
assert len(order)==954 and len(set(order))==954
assert all(dx_by_id[s]=="ASD" for s in asd) and all(dx_by_id[s]=="NC" for s in nc)
print(f"ordering: {len(asd)} ASD then {len(nc)} NC = {len(order)}")
order_sha=hashlib.sha256("\n".join(order).encode()).hexdigest()
mapping=pd.DataFrame({"subject_id":range(954),"FILE_ID":order,
                      "y":[1]*len(asd)+[0]*len(nc),
                      "dx_storage":[dx_by_id[s] for s in order],
                      "DX_GROUP_upstream":[1 if dx_by_id[s]=="ASD" else 2 for s in order]})
mapping.to_csv(S5+"subject_id_map.csv",index=False)
map_sha=hashlib.sha256(open(S5+"subject_id_map.csv","rb").read()).hexdigest()

# ---------- normalizations, each starting from RAW M1 ----------
def norm(X,k):                       # X: (90,3) raw
    if k=="B": mn,mx=X.min(),X.max();               return (X-mn)/(mx-mn)
    if k=="C": mn,mx=X.min(0,keepdims=True),X.max(0,keepdims=True); return (X-mn)/(mx-mn)
    if k=="D": return (X-X.mean(0,keepdims=True))/X.std(0,keepdims=True)

FC={}
for s in order:
    cls="ASD" if dx_by_id[s]=="ASD" else "NC"
    FC[s]=sio.loadmat(f"{A}data/raw/{cls}_ADJ/{s}_adj.mat")["cropped_matrix"]

nodes=torch.arange(90,dtype=torch.long)
EI=torch.stack([nodes.repeat_interleave(90), nodes.repeat(90)],0)

report={}
for k in "BCD":
    dl=[]
    for i,s in enumerate(order):
        x=torch.tensor(norm(M1_by_id[s],k),dtype=torch.float)
        fc=np.nan_to_num(FC[s]).astype(np.float32)
        ma=np.abs(fc).max()
        if ma>0: fc=fc/ma                      # loader-identical edge rule (proven no-op in S4)
        d=Data(x=x, edge_index=EI.clone(),
               edge_weight=torch.from_numpy(fc.reshape(-1)).float(),
               y=torch.tensor([1 if dx_by_id[s]=="ASD" else 0]), num_nodes=90)
        d.subject_id=torch.tensor([i],dtype=torch.long)
        dl.append(d)
    root=S5+f"M1_{k}/processed"; os.makedirs(root,exist_ok=True)
    path=f"{root}/M1_{k}_v1.pt"
    torch.save(InMemoryDataset.collate(dl), path)
    h=hashlib.sha256(open(path,"rb").read()).hexdigest()
    report[f"M1_{k}"]={"path":path,"sha256":h,"n":len(dl),"bytes":os.path.getsize(path)}
    print(f"built M1_{k}: {len(dl)} graphs -> {path}  sha256 {h[:16]}...")

json.dump({"subject_order_sha256":order_sha,"subject_id_map_sha256":map_sha,
           "X_sources_sha256":xh,"caches":report,
           "ordering":"ASD block (sorted) then NC block (sorted) — identical to "
                      "datasets/abideDataset.py process()"},
          open(S5+"s5_hashes.json","w"),indent=1)
print(f"\nsubject_order_sha256 = {order_sha}")
print(f"subject_id_map_sha256 = {map_sha}")
