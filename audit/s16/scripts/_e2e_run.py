"""E2E gate, ONE target per array task (parallel). Assertions run afterwards by
_e2e_check.py reading the JSONs from disk."""
import sys, os
sys.path.insert(0,'/users/3171356m/A-GCL/audit/s16/scripts')
import s16_train as TR, s16_data as DAT
TR.MAX_EPOCHS, TR.MIN_EPOCHS, TR.PATIENCE = 4, 2, 2
import s16_worker as W, s16_grid as G
DAT.folds_orig = DAT.folds
DAT.folds = lambda d,p: (DAT.folds_orig(d,p)[:1] if p=='lab' else [])   # ONE fold

def targets():
    T=[]
    for arm in ("A1","A3","A4","A5","A6","A7"):
        for mode in ("plain","fused"):
            i=next((k for k,u in enumerate(G.MAIN) if u["arm"]==arm and u["mode"]==mode
                    and u["E"]=="signed" and u["seed_idx"]==0), None)
            if i is not None: T.append(("main",i,f"{arm}-{mode}"))
    for c in ("C-RAND","C-PERM","C-SHUF","C-ROI"):
        i=next((k for k,u in enumerate(G.CTRL) if u["control"]==c and u["seed_idx"]==0), None)
        if i is not None: T.append(("ctrl",i,c))
    i=next((k for k,u in enumerate(G.ABL) if u["seed_idx"]==0), None)
    if i is not None: T.append(("abl",i,"ALFF-abl"))
    # also exercise the sparse path, which no signed-only target touches
    i=next((k for k,u in enumerate(G.MAIN) if u["arm"]=="A1" and u["E"]=="pos_zero"
            and u["mode"]=="plain" and u["seed_idx"]==0), None)
    if i is not None: T.append(("main",i,"A1-pos_zero-SPARSE"))
    return T

if __name__=="__main__":
    T=targets(); k=int(sys.argv[1])
    if k>=len(T): print(f"no target {k}"); sys.exit(0)
    b,i,label=T[k]; print(f"E2E target {k}: {label}",flush=True)
    W.run(b,i)
