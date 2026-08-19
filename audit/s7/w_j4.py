"""J4 - consolidation. NO NEW SCIENCE. Validates, counts, computes paired stats, always writes completion."""
import sys, os, json, glob, numpy as np, pandas as pd, datetime
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C
S=C.S7
def expected_units():
    W=[]
    for s in range(30):
        for p in ["P","O","C"]:
            for b in C.BRANCHES: W.append((s,p,b))
    for s in range(30,50):
        for p in ["P","O","C"]: W.append((s,p,"B"))
    return W
EXP=expected_units()
comp={}; missing=[]; failed=[]
for (s,p,b) in EXP:
    n=f"emb_s{s:03d}_{p}_{b}"
    a=C.is_done(S+"J3a",n); bdone=C.is_done(S+"J3b","probe_"+n)
    comp[n]=dict(j3a=a,j3b=bdone)
    if not a: missing.append("J3a:"+n)
    if not bdone: missing.append("J3b:probe_"+n)
j1=os.path.exists(S+"J1/J1_DONE"); j2=os.path.exists(S+"J2/J2_DONE")
rows=[]
for f in sorted(glob.glob(S+"J3b/probe_*.json")):
    try: rows.append(json.load(open(f)))
    except Exception as e: failed.append(f+f" ({e})")
df=pd.DataFrame(rows)
paired=[]
if len(df):
    df=df[np.isfinite(df.h_auc)&np.isfinite(df.z_auc)]
    df.to_csv(S+"S7_RESULTS_SUMMARY.csv",index=False)
    rng=np.random.default_rng(C.BASE_SEED)
    def pstat(a,b,lab):
        m=pd.merge(a,b,on="seed",suffixes=("_1","_2"))
        if not len(m): return None
        d=(m.iloc[:,1]-m.iloc[:,2]).values
        bs=[np.mean(d[rng.integers(0,len(d),len(d))]) for _ in range(2000)]
        return dict(comparison=lab,n_pairs=len(d),mean=float(d.mean()),median=float(np.median(d)),
            sd=float(d.std(ddof=1)) if len(d)>1 else 0.0,ci_lo=float(np.percentile(bs,2.5)),
            ci_hi=float(np.percentile(bs,97.5)),min=float(d.min()),max=float(d.max()))
    for rep in ["h","z"]:
        sub={p:df[(df.path==p)&(df.branch=="B")][["seed",f"{rep}_auc"]] for p in ["P","O","C"]}
        for x,z in (("O","P"),("C","P"),("O","C")):
            r=pstat(sub[x],sub[z],f"{x}_B_{rep} - {z}_B_{rep}")
            if r: paired.append(r)
    for p in ["P","O","C"]:
        a=df[(df.path==p)&(df.branch=="B")][["seed","h_auc"]]; b=df[(df.path==p)&(df.branch=="B")][["seed","z_auc"]]
        r=pstat(a,b,f"{p}_B_h - {p}_B_z")
        if r: paired.append(r)
    pd.DataFrame(paired).to_csv(S+"S7_PAIRED_STATS.csv",index=False)
n_exp=len(EXP)*2+2
n_done=sum(v["j3a"]+v["j3b"] for v in comp.values())+int(j1)+int(j2)
status="COMPLETE" if (not missing and j1 and j2 and not failed) else ("FAILED" if not j1 else "PARTIAL")
out=dict(status=status,expected_units=n_exp,completed_units=int(n_done),
    missing_units=missing[:200],n_missing=len(missing),failed_units=failed,
    J1_status="DONE" if j1 else "MISSING", J2_status="DONE" if j2 else "MISSING",
    J3a_status=f"{sum(v['j3a'] for v in comp.values())}/{len(EXP)}",
    J3b_status=f"{sum(v['j3b'] for v in comp.values())}/{len(EXP)}",
    git_head=C.git_head(), git_dirty=C.git_dirty(),
    timestamp=datetime.datetime.now().isoformat(timespec="seconds"), paired_stats=paired)
json.dump(out,open(S+"S7_COMPLETION.json","w"),indent=1,default=str)
open(S+"S7_MISSING_UNITS.txt","w").write("\n".join(missing) or "(none)\n")
man=[]
for f in sorted(glob.glob(S+"J3a/emb_*.json")):
    try: man.append(json.load(open(f)))
    except Exception: pass
if man: pd.DataFrame(man).to_csv(S+"S7_JOB_MANIFEST.csv",index=False)
with open(S+"S7_EVIDENCE_DRAFT.md","w") as f:
    f.write(f"# S7 EVIDENCE DRAFT (auto-generated, NOT reviewed)\nstatus: {status}\n")
    f.write(f"git HEAD {out['git_head']}\ntimestamp {out['timestamp']}\n\n")
    f.write(f"J1 {out['J1_status']} | J2 {out['J2_status']} | J3a {out['J3a_status']} | J3b {out['J3b_status']}\n\n")
    if paired:
        f.write("## paired random-encoder differences\n")
        for r in paired: f.write(f"  {r['comparison']}: n={r['n_pairs']} mean={r['mean']:+.4f} "
                                 f"95%CI [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]\n")
    f.write("\nNO A-GCL TRAINING WAS PERFORMED.\n")
if status=="COMPLETE": open(S+"S7_ALL_JOBS_COMPLETE","w").write(out["timestamp"]+"\n")
print(json.dumps({k:v for k,v in out.items() if k not in ("missing_units","paired_stats")},indent=1))
print("J4 COMPLETE")
