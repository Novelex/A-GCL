"""E2E gate, ONE target per array task (parallel). Assertions run afterwards by
_e2e_check.py reading the JSONs from disk."""
import sys, os
sys.path.insert(0,'/users/3171356m/A-GCL/audit/s16/scripts')
import s16_train as TR, s16_data as DAT
import s16_worker as W, s16_grid as G

# *** NO IMPORT-TIME MUTATION. ***
# These overrides previously ran at module scope, so merely IMPORTING this file
# rewrote the global fold definition and the epoch budget for every other module in
# the process. That silently collapsed the expected ledger from 9 folds to 1
# (hash 8587b1ca36553408 -> 2dfbce8b946e4a17). They are now applied ONLY inside
# apply_e2e_overrides(), called from __main__.
def apply_e2e_overrides():
    TR.MAX_EPOCHS, TR.MIN_EPOCHS, TR.PATIENCE = 4, 2, 2
    if not hasattr(DAT, "folds_orig"): DAT.folds_orig = DAT.folds
    DAT.folds = lambda d,p: (DAT.folds_orig(d,p)[:1] if p=='lab' else [])

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
    # A7 under ALL FOUR E (its input is the E-transformed edge triangle)
    for e in ("abs","pos_zero","shift"):
        i=next((k for k,u in enumerate(G.MAIN) if u["arm"]=="A7" and u["E"]==e
                and u["mode"]=="plain" and u["seed_idx"]==0), None)
        if i is not None: T.append(("main",i,f"A7-{e}"))
    # SPARSE WGIN path, which no signed-only target touches
    for arm in ("A1","A4"):
        i=next((k for k,u in enumerate(G.MAIN) if u["arm"]==arm and u["E"]=="pos_zero"
                and u["mode"]=="plain" and u["seed_idx"]==0), None)
        if i is not None: T.append(("main",i,f"{arm}-pos_zero-SPARSE-WGIN"))
    # BNT under a non-signed E
    i=next((k for k,u in enumerate(G.MAIN) if u["arm"]=="A6" and u["E"]=="shift"
            and u["mode"]=="fused" and u["seed_idx"]==0), None)
    if i is not None: T.append(("main",i,"A6-shift-fused"))
    # ALL ALFF branches (raw / perband / joint via the ABL units; z is the default)
    for k,u in enumerate(G.ABL):
        if u["seed_idx"]==0: T.append(("abl",k,f"ALFF-{u['alff_mode']}"))
    return T

if __name__=="__main__":
    apply_e2e_overrides()
    T=targets(); k=int(sys.argv[1])
    if k>=len(T): print(f"no target {k}"); sys.exit(0)
    b,i,label=T[k]; print(f"E2E target {k}: {label} [ns=e2e]",flush=True)
    W.run(b,i,ns="e2e")
