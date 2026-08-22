"""Winner (arm1 CE) secondary: flatten-2880 probe per ord fold, 3 seeds."""
import sys, json, numpy as np
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s12a4/scripts"); import s12a4_core as M
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
import s11_core as K
from sklearn.metrics import roc_auc_score
df,X_fc,y,ids,gh=A1.load_gate()
out={}
for s in (0,1,2):
    oof=np.full(954,np.nan)
    for i in range(5):
        z=np.load(f"{M.S12A4}feat/a1_s{s}_o{i}.npz")
        d,o=K.probe_pipe(z["nodes_flat"].astype(np.float64),y,[(z["tr"],z["te"])],[])
        oof[z["te"]]=o[z["te"]]
    out[f"s{s}"]=float(roc_auc_score(y,oof))
    print("flat seed",s,out[f"s{s}"],flush=True)
out["mean"]=float(np.mean(list(out.values())))
json.dump(out,open(f"{M.S12A4}out/WINNER_FLAT.json","w"),indent=1)
print("FLAT:",out)
