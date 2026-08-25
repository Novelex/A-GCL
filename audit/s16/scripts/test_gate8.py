"""S16 Gate 8: local miniature coverage. No cluster jobs, no scientific results."""
import sys, os, json, glob, numpy as np, torch
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_prov as P, s16_grid as G, s16_feat as FT, s16_models as MO, s16_data as DAT
import s16_ledger as L
F=[]
def ck(n,ok,d=""):
    print(("PASS " if ok else "FAIL ")+n+(" | "+d if d else ""),flush=True)
    if not ok: F.append(n)

print("--- A. E2E assertion semantics ---")
src=open("_e2e_check.py").read()
ck("A1_exact_half_is_valid", "0.0<=a<=1.0" in src and "abs(a-0.5)>1e-9" not in src,
   "AUC exactly 0.5 accepted; only finite and within [0,1] required (D9)")
import _e2e_run as R
T=R.targets(); labels=[l for _,_,l in T]
cov = {
 "every architecture": all(any(x in " ".join(labels) for x in a)
        for a in (("A1","A3","A4"),("A5","A6"),("A7",))),
 "plain and fused": any("-plain" in l for l in labels) and any("-fused" in l for l in labels),
 "A7 all four E": all(any(f"A7-{e}" in l for l in labels) for e in ("abs","pos_zero","shift"))
        and any(l=="A7-plain" for l in labels),
 "sparse WGIN": sum("pos_zero-SPARSE-WGIN" in l for l in labels)>=2,
 "all controls": all(f"{c}-{a}" in labels for c in ("C-RAND","C-PERM","C-SHUF","C-ROI")
                     for a in ("BNT","WGIN")),
 "all ALFF branches": all(any(f"ALFF-{m}" in l for l in labels)
        for m in ("raw","perband","joint")),
}
for k,v in cov.items(): ck(f"A2_cov::{k}", v)
ck("A3_target_count", len(T)==29, f"{len(T)} targets")

print("\n--- serialization round-trip / save-reload ---")
d,MAN,ent = DAT.load("signed", where="gate8"); y=d["y"].astype(np.int64)
tag,tr,te = DAT.folds(d,"lab")[0]; tr_enc,_ = FT.honest_split(tr,y)
for arch,kh,spec in (("BNT",32,"fcrow+alff"),("WGIN",128,"fcrow+alff"),("EDGEMLP",256,"edgetri")):
    X,FCu = FT.build_X(spec,d["FC"],d["ALFF"],tr_enc)
    m = MO.build_model(arch, X.shape[-1], DAT.BASE, kh)
    import s16_train as TR
    r1,s1 = TR.extract(m,X,FCu,range(16),arch=="WGIN")
    pth=f"/tmp/_g8_{arch}.pt"; torch.save(m.state_dict(),pth)
    m2 = MO.build_model(arch, X.shape[-1], DAT.BASE+7, kh)
    m2.load_state_dict(torch.load(pth,weights_only=True))
    r2,s2 = TR.extract(m2,X,FCu,range(16),arch=="WGIN")
    ck(f"A4_roundtrip_{arch}", np.array_equal(r1,r2) and np.array_equal(s1,s2), "bitwise")

print("\n--- alpha=1 identity + conservative tie-breaking (re-checked here) ---")
import s11_core as K
Xfc,_,_,_ = K.load_Xfc(); _,tp = FT.honest_split(tr,y)
Rj = np.random.default_rng(0).standard_normal((954,32))
s_fc,s_le = FT.scores_for_fusion(Rj,Xfc,y,tr_enc,tp,te)
from sklearn.metrics import roc_auc_score
f1 = FT.fuse_scores(s_fc,s_le,1.0,tp)
ck("A5_alpha1_identity", abs(roc_auc_score(y[te],f1[te])-roc_auc_score(y[te],s_fc[te]))<1e-12)
TB=lambda inner: max(inner,key=lambda r:(r["auc"],r["alpha"]))["alpha"]
ck("A6_tiebreak_conservative",
   abs(TB([dict(alpha=a,auc=0.6) for a in np.round(np.arange(0,1.001,.05),4)])-1.0)<1e-9)

print("\n--- namespace isolation / resume / collector rejection (re-checked) ---")
ck("A7_namespaces_disjoint", P.root("prod")!=P.root("e2e"))
ck("A8_resume_fields", len(P.MATCH_KEYS)>=26, f"{len(P.MATCH_KEYS)} contracted fields")
ck("A9_ledger_shape", L.assert_grid_shape()[0]==[], f"159x9=1431, hash {L.ledger_hash()}")

print("\n--- C. provenance completeness ---")
w=open("s16_worker.py").read(); pv=open("s16_prov.py").read()
for field,where in (("git_sha",pv),("worktree_clean",pv),("builder_sha",pv),
                    ("h_fc",pv),("h_folds_lab",pv),("cache_file",pv),
                    ("config_hash",pv),("worker_version",pv),
                    ("collector_version",pv),("environment",pv)):
    ck(f"C1_prov::{field}", field in where, "recorded" if field in where else "MISSING")

print(f"\n=== GATE 8 SUMMARY: {len(F)} failures ===")
for f in F: print("  FAIL "+f)
sys.exit(1 if F else 0)
