import numpy as np, pandas as pd, os, hashlib, collections, json
OUT="/users/3171356m/agcl_audit_s0"
R="data/raw"

# ---------- sources ----------
z1=np.load("ALFF_func_proc/method1/alff_roi_first.npz")
z2=np.load("ALFF_func_proc/method2/alff_voxel_first.npz")
zo=np.load("data/ALFF_need/alff_new.npz")
zc=np.load("data/ALFF_need/alff_new_combat.npz", allow_pickle=True)
A=[str(s) for s in z1["file_ids"]]; B=[str(s) for s in z2["file_ids"]]; O=[str(s) for s in zo["file_ids"]]
OC=[str(s) for s in zc["file_ids"]]

def raw_ids(d,suf):
    return sorted(f[:-len(suf)] for f in os.listdir(f"{R}/{d}") if f.endswith(suf))
ASD=set(raw_ids("ASD_ADJ","_adj.mat")); NC=set(raw_ids("NC_ADJ","_adj.mat"))
FC_ALL=sorted(ASD|NC)

EXCLUDED=["CMU_b_0050669","Leuven_1_0050706"]

# ---------- cohort ----------
COH=sorted(set(A)&set(B))
print("="*76); print("1-5 COHORT CHECKS"); print("="*76)
print(f"  |A|={len(A)} |B|={len(B)} |A n B|={len(COH)}")
print(f"  exactly 954 unique ids        : {len(COH)==954 and len(set(COH))==954}")
print(f"  duplicate ids in A/B/OLD/FC   : {len(A)-len(set(A))}/{len(B)-len(set(B))}/{len(O)-len(set(O))}/{len(FC_ALL)-len(set(FC_ALL))}")
print(f"  ASD n NC overlap (storage)    : {len(ASD&NC)}")
nASD=len([s for s in COH if s in ASD]); nNC=len([s for s in COH if s in NC])
print(f"  ASD={nASD}  NC={nNC}  (target 455/499)  -> {nASD==455 and nNC==499}")
print(f"  cohort subset of FC           : {set(COH)<=set(FC_ALL)}")
print(f"  OLD superset of cohort        : {set(COH)<=set(O)}  (OLD extra = {sorted(set(O)-set(COH))})")
print(f"  A ids == B ids                : {set(A)==set(B)}")
print(f"  OLD-subset == cohort          : {sorted(set(O)&set(COH))==COH}")
h=hashlib.sha256("\n".join(COH).encode()).hexdigest()
print(f"  SHA-256 of sorted 954 ids (newline-joined): {h}")
h2=hashlib.sha256("\n".join(A).encode()).hexdigest()
print(f"  SHA-256 of A-order 954 ids               : {h2}")

# ---------- 6 row order ----------
print(); print("="*76); print("6. ROW ORDER"); print("="*76)
print(f"  A order == B order            : {A==B}")
print(f"  A order == sorted(cohort)     : {A==COH}")
Osub=[s for s in O if s in set(COH)]
print(f"  OLD natural order restricted to cohort == A order : {Osub==A}")
print(f"  OLD full order is sorted      : {O==sorted(O)}")
print(f"  combat file order == OLD order: {OC==O}")
print(f"  FC dir listing sorted == cohort-sorted (on cohort): {[s for s in FC_ALL if s in set(COH)]==COH}")
oidx={s:i for i,s in enumerate(O)}
shift=[(s,oidx[s]) for s in A[:3]]
print(f"  NOTE: OLD row index != A row index for subjects after the first exclusion.")
print(f"        e.g. first 3 cohort ids -> OLD indices {shift}")
first_shift=next(i for i,s in enumerate(A) if oidx[s]!=i)
print(f"        first index where OLD_idx != A_idx : A position {first_shift} (id {A[first_shift]}, OLD idx {oidx[A[first_shift]]})")

# ---------- 7 diagnosis coding ----------
print(); print("="*76); print("7. DIAGNOSIS CODING SCHEMES (derived, not assumed)"); print("="*76)
tr=pd.read_csv("data/subject_tr.csv")
dm=pd.read_csv("data/ALFF_need/download_manifest.csv")
ph=pd.read_csv("/mnt/scratch/users/3171356m/muhammad-GraSTIACL/data/raw/qc_passed_func_proc/phenotypic_filtered_v2.csv")
print(f"  sources: subject_tr.csv({len(tr)}), download_manifest.csv({len(dm)}), phenotypic_filtered_v2.csv({len(ph)}), alff_new.npz dx_group({len(zo['dx_group'])})")

storage={s:("ASD" if s in ASD else "NC") for s in FC_ALL}
def crosstab(name, mapping):
    ct=collections.Counter((storage[s], mapping[s]) for s in FC_ALL if s in mapping)
    print(f"  [{name}] crosstab storage x value: {dict(ct)}")
    # derive mapping
    vals=sorted({v for _,v in ct})
    derived={}
    for v in vals:
        tally=collections.Counter(st for (st,vv) in ct.elements() if vv==v)
        derived[v]=tally.most_common(1)[0][0]
        pure = len(tally)==1
        print(f"      value {v!r:12s} -> storage {dict(tally)}  PURE={pure}")
    return derived

m_tr={r.FILE_ID:int(r.DX_GROUP) for r in tr.itertuples()}
m_dm={r.FILE_ID:str(r.label) for r in dm.itertuples()}
m_ph={r.FILE_ID:int(r.DX_GROUP) for r in ph.itertuples()}
m_np={str(s):int(v) for s,v in zip(zo["file_ids"], zo["dx_group"])}
d_tr=crosstab("subject_tr.csv DX_GROUP", m_tr)
d_dm=crosstab("download_manifest.csv label", m_dm)
d_ph=crosstab("phenotypic_filtered_v2 DX_GROUP", m_ph)
d_np=crosstab("alff_new.npz dx_group", m_np)
if "DSM_IV_TR" in ph.columns:
    m_ds={r.FILE_ID:int(r.DSM_IV_TR) for r in ph.itertuples() if pd.notna(r.DSM_IV_TR)}
    d_ds=crosstab("phenotypic DSM_IV_TR", m_ds)

# ---------- 8 mismatch report ----------
print(); print("="*76); print("8. SUBJECT-LEVEL DIAGNOSIS MISMATCH (cohort of 954)"); print("="*76)
rows=[]; mism=collections.Counter()
site_tr={r.FILE_ID:str(r.SITE_ID) for r in tr.itertuples()}
site_cb={str(s):str(v) for s,v in zip(zc["file_ids"], zc["site"])}
site_ph={r.FILE_ID:str(r.SITE_ID) for r in ph.itertuples()}
a_i={s:i for i,s in enumerate(A)}; b_i={s:i for i,s in enumerate(B)}
for s in COH:
    st=storage[s]
    conv={"tr":d_tr[m_tr[s]], "dm":d_dm[m_dm[s]], "ph":d_ph[m_ph[s]], "np":d_np[m_np[s]]}
    bad=[k for k,v in conv.items() if v!=st]
    for k in bad: mism[k]+=1
    rows.append(dict(subject_id=s, m1_row=a_i[s], m2_row=b_i[s], old_row=oidx[s],
        old_subset_row=COH.index(s), fc_adj_file=f"{'ASD' if st=='ASD' else 'NC'}_ADJ/{s}_adj.mat",
        fc_index=FC_ALL.index(s), dx_storage=st,
        dx_subject_tr=m_tr[s], dx_manifest=m_dm[s], dx_phenotypic=m_ph[s], dx_alff_npz=m_np[s],
        dsm_iv_tr=m_ds.get(s,""), site_tr=site_tr[s], site_combat=site_cb[s], site_pheno=site_ph[s],
        site_prefix=s.rsplit("_",1)[0], n_roi_m1=z1["alff"].shape[1], n_roi_m2=z2["alff"].shape[1],
        n_roi_old=zo["alff"].shape[1], n_roi_fc=90, mismatch_sources=";".join(bad)))
df=pd.DataFrame(rows)
df.to_csv(f"{OUT}/s1_audit_table.csv", index=False)
print(f"  audit table rows: {len(df)}  -> {OUT}/s1_audit_table.csv")
print(f"  per-source diagnosis mismatches vs storage: {dict(mism) if mism else 'ZERO across all 4 sources'}")
print(f"  subjects with ANY mismatch: {(df.mismatch_sources!='').sum()}")
print(f"  site agreement tr vs pheno : {(df.site_tr==df.site_pheno).sum()}/954")
print(f"  site_combat distinct values: {sorted(df.site_combat.unique())[:8]} ... n={df.site_combat.nunique()}")
print(f"  site_tr distinct: n={df.site_tr.nunique()}  site_prefix distinct: n={df.site_prefix.nunique()}")

# ---------- 10 excluded ----------
print(); print("="*76); print("10. EXCLUDED SUBJECTS"); print("="*76)
for s in EXCLUDED:
    print(f"  {s}: in_cohort={s in set(COH)} in_A={s in set(A)} in_B={s in set(B)} "
          f"in_OLD={s in set(O)} in_FC={s in set(FC_ALL)} storage={storage.get(s)}")
print(f"  cohort contains neither excluded id: {not (set(EXCLUDED) & set(COH))}")
json.dump({"cohort_sha256":h,"a_order_sha256":h2,"n":len(COH)}, open(f"{OUT}/s1_cohort_hash.json","w"), indent=1)
