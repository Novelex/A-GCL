"""Run ONE real fold of EVERY arm through the REAL worker, short epochs.
This is the gate that was missing before the last two submissions."""
import sys, itertools
sys.path.insert(0,'/users/3171356m/A-GCL/audit/s16/scripts')
import s16_train as TR, s16_data as DAT
TR.MAX_EPOCHS, TR.MIN_EPOCHS, TR.PATIENCE = 4, 2, 2     # short, correctness only
import s16_worker as W, s16_grid as G
_orig = DAT.all_folds if hasattr(DAT,'all_folds') else None
DAT.folds_orig = DAT.folds
DAT.folds = lambda d,p: DAT.folds_orig(d,p)[:1]          # 1 fold per protocol -> 3
targets=[]
for arm in ("A1","A3","A4","A5","A6","A7"):
    for mode in ("plain","fused"):
        i=next((k for k,u in enumerate(G.MAIN) if u["arm"]==arm and u["mode"]==mode
                and u["E"]=="signed" and u["seed_idx"]==0), None)
        if i is not None: targets.append(("main",i,f"{arm}-{mode}"))
for c in ("C-RAND","C-PERM","C-SHUF","C-ROI"):
    i=next((k for k,u in enumerate(G.CTRL) if u["control"]==c and u["seed_idx"]==0), None)
    if i is not None: targets.append(("ctrl",i,c))
i=next((k for k,u in enumerate(G.ABL) if u["seed_idx"]==0), None)
if i is not None: targets.append(("abl",i,"ALFF-abl"))
fails=[]
for branch,idx,label in targets:
    try:
        W.run(branch, idx); print(f"E2E OK   {label}", flush=True)
    except Exception as e:
        fails.append((label,repr(e))); print(f"E2E FAIL {label}: {e!r}", flush=True)
print("\n=== E2E SUMMARY ===")
print(f"{len(targets)-len(fails)}/{len(targets)} arms ran end-to-end")
for l,e in fails: print("  FAIL",l,e[:160])
sys.exit(1 if fails else 0)
