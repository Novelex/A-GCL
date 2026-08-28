"""S17 Wave-1 E2E runner: the four Wave-1 arms at the frozen 4-epoch e2e policy.

Replaces the unusable S16 entrypoints for S17:
  _e2e.py      is a deprecated stub that exits 2
  _e2e_run.py  is hardwired to S16's 29 targets and audit/s16/runs/e2e

Writes ONLY under audit/s17/runs/e2e/. Submits nothing.
Usage:  python s17_e2e.py            # all four targets
        python s17_e2e.py 2          # one target by index
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, "/users/3171356m/A-GCL/audit/s16/scripts")
import s16_grid as G, s16_feat as FT, s16_policy as PL
import s17_worker as W17

NS = "e2e"

def targets():
    """Wave 1: the three input specs, plus C-ROI on the signed spec.

    C-ROI is included because it is the one path that fails SILENTLY if n_profile is
    wrong (90 for signed/abs, 180 for split), so the smoke test must exercise it."""
    T = []
    for arm in ("R1s", "R1a", "R1p"):
        T.append(dict(branch="s17w1", arm=arm, E="signed", arch=G.ARCH[arm],
                      kh=G.KH[G.ARCH[arm]], mode="plain", seed_idx=0,
                      seed=G.SEEDS[0], alff_mode="z", control=None))
    T.append(dict(branch="s17w1", arm="R1s", E="signed", arch=G.ARCH["R1s"],
                  kh=G.KH["ROWMLP"], mode="plain", seed_idx=0, seed=G.SEEDS[0],
                  alff_mode="z", control="C-ROI"))
    return T

def assert_targets_unique(T):
    ids = [W17.unit_id(u) for u in T]
    assert len(set(ids)) == len(ids), f"duplicate unit ids: {ids}"
    return ids

def main():
    T = targets(); ids = assert_targets_unique(T)
    pol = PL.get(NS)
    print(f"S17 Wave 1 E2E | ns={NS} policy={pol.name} max_epochs={pol.max_epochs} "
          f"hash={pol.policy_hash()} folds=({pol.n_lab},{pol.n_site},{pol.n_loso})")
    print(f"root = {W17.root(NS)}")
    assert "/audit/s17/" in W17.root(NS) and "/audit/s16/" not in W17.root(NS)
    print(f"{len(T)} targets: {ids}\n")
    sel = range(len(T))
    if len(sys.argv) > 1: sel = [int(sys.argv[1])]
    out, t0 = [], time.time()
    for k in sel:
        u = T[k]
        print(f"--- target {k}: {W17.unit_id(u)}  spec={FT.ARMS[u['arm']][1]}", flush=True)
        out.extend(W17.run_unit(u, ns=NS))
    print(f"\n{'unit':34s} {'fold':6s} {'D':>4s} {'repr':>5s} {'head':>8s} {'probe':>8s} "
          f"{'ep':>3s} {'mv':>6s}")
    print("-" * 84)
    ok = True
    for r in out:
        h, p_ = r["head"]["auc"], r["probe_honest"]["auc"]
        fin = all(np.isfinite(v) and 0.0 < v < 1.0 for v in (h, p_))
        ok &= fin
        print(f"{r['unit']:34s} {r['fold']:6s} {r['D_in']:4d} {r['repr_dim_used']:5d} "
              f"{h:8.4f} {p_:8.4f} {r['best_epoch']:3d} {r['movement_max']:6.3f}"
              + ("" if fin else "  *** NON-FINITE / OUTSIDE (0,1)"))
    print("-" * 84)
    print(f"{len(out)} folds over {len(list(sel))} arms in {time.time()-t0:.0f}s")
    print(f"every AUC finite and strictly inside (0,1): {ok}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
