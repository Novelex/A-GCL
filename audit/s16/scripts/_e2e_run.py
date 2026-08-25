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
E2E_NAMESPACE = "e2e"          # HARD-CODED. Never inherited from the environment.

def e2e_policy():
    """The E2E ExecPolicy. The worker derives BOTH the training budget AND the
    provenance record from this same object, so a record can no longer claim 400
    epochs while 4 were run."""
    import s16_policy as PL
    return PL.get(E2E_NAMESPACE)

def targets():
    T=[]
    for arm in ("A1","A3","A4","A5","A6","A7"):
        for mode in ("plain","fused"):
            i=next((k for k,u in enumerate(G.MAIN) if u["arm"]==arm and u["mode"]==mode
                    and u["E"]=="signed" and u["seed_idx"]==0), None)
            if i is not None: T.append(("main",i,f"{arm}-{mode}"))
    # controls for BOTH architectures: C-SHUF/C-ROI act on the graph for WGIN and on
    # node features only for BNT, so their behaviour IS architecture-dependent.
    for c in ("C-RAND","C-PERM","C-SHUF","C-ROI"):
        for arm,archname in (("A6","BNT"),("A4","WGIN")):
            i=next((k for k,u in enumerate(G.CTRL) if u["control"]==c
                    and u["arm"]==arm and u["seed_idx"]==0), None)
            if i is not None: T.append(("ctrl",i,f"{c}-{archname}"))
    # NOTE: no generic "ALFF-abl" entry — it selected the SAME unit as ALFF-raw,
    # giving 26 labels over 25 unique unit IDs. The ALFF modes are enumerated below.
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
    assert_targets_unique(T)
    return T

def assert_targets_unique(T):
    """Labels, (branch,index) pairs, unit IDs and output paths must all be unique.
    A duplicate silently halves coverage while appearing to add a target."""
    import collections
    labels=[l for _,_,l in T]; pairs=[(b,i) for b,i,_ in T]
    uids=[G.unit_id({"main":G.MAIN,"ctrl":G.CTRL,"abl":G.ABL}[b][i]) for b,i,_ in T]
    for name,seq in (("labels",labels),("(branch,index)",pairs),("unit IDs",uids)):
        dup=[x for x,c in collections.Counter(seq).items() if c>1]
        if dup: raise AssertionError(f"duplicate E2E {name}: {dup}")
    paths=[f"{u}__lab0" for u in uids]
    dup=[x for x,c in collections.Counter(paths).items() if c>1]
    if dup: raise AssertionError(f"duplicate E2E output paths: {dup}")
    return True

if __name__=="__main__":
    os.environ["S16_NS"] = E2E_NAMESPACE          # explicit, not inherited
    T=targets(); k=int(sys.argv[1])
    if k>=len(T): print(f"no target {k}"); sys.exit(0)
    b,i,label=T[k]
    pol=e2e_policy()
    print(f"E2E target {k}: {label} [ns={E2E_NAMESPACE} policy={pol.name} "
          f"max_epochs={pol.max_epochs} folds=({pol.n_lab},{pol.n_site},{pol.n_loso})]",
          flush=True)
    W.run(b,i,ns=E2E_NAMESPACE)
