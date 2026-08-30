"""Stage A array task. Units = (protocol, fold, representation) + 29 shuffled-label
units. MaxArraySize on this cluster is 1001, so units are PACKED: task k handles
units k, k+NTASK, k+2*NTASK ... Each unit writes ONE atomic npz+json; a unit is
skipped only if its output already exists AND re-validates (requeue-safe, never a
stale-cache trap: outputs are keyed by protocol/fold/rep and contain their own
fold indices, which are re-checked against the frozen folds on load)."""
import os, sys, json, time, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_lib as L

OUT = L.ROOT + "A/"; NTASK = int(os.environ.get("SWEEP_NTASK", "640"))

def units():
    U = []
    for p in L.protocols():
        for k in range(len(L.outer_folds(p))):
            for r in L.rep_catalogue(): U.append((p, k, r.name(), False))
            U.append((p, k, "flat", True))                       # shuffled-label arm
    return U                                                     # 29*43 + 29 = 1276

def path(p, k, rname, shuf): return f"{OUT}{p}/f{k}/{'SHUF_' if shuf else ''}{rname}"

def valid(base, p, k):
    """A unit is reusable only if json+npz are a SEALED pair (json records the npz
    sha), the fold indices match the frozen fold, and the config set is exactly the
    current catalogue - so a stale json next to a fresh npz, or an output from an
    older catalogue, can never be accepted."""
    try:
        j = json.load(open(base + ".json")); z = np.load(base + ".npz")
        te = L.outer_folds(p)[k][2]; cat = {L.cfg_id(kd, hp) for kd, hp in L.clf_catalogue()}
        return (j.get("status") == "OK" and np.array_equal(z["test_idx"], te)
                and set(j["inner_mean"]) == set(z.files) - {"test_idx"} == cat
                and j.get("npz_sha") == L.sha_file(base + ".npz"))
    except Exception: return False

def run_unit(p, k, rname, shuf):
    base = path(p, k, rname, shuf)
    if valid(base, p, k): return "skip"
    rep = next(r for r in L.rep_catalogue() if r.name() == rname)
    y_over = None
    if shuf:
        rng = np.random.default_rng(L.SEED); y_over = L.data()["y"][rng.permutation(954)]
    r = L.eval_rep_on_fold(rep, p, k, y_override=y_over)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    sha = L.anpz(base + ".npz", test_idx=r["test_idx"], **r["test_scores"])
    r2 = {k_: v for k_, v in r.items() if k_ not in ("test_idx", "test_scores")}
    r2.update(status="OK", shuffled=bool(shuf), y_perm_seed=(L.SEED if shuf else None), npz_sha=sha)
    L.aj(r2, base + ".json")
    return f"ok {r['secs_total']}s"

def main():
    k = int(sys.argv[1]); U = units(); mine = U[k::NTASK]
    print(f"task {k}/{NTASK}: {len(mine)} units", flush=True); rc = 0
    for (p, kf, rn, sh) in mine:
        t0 = time.time()
        try: msg = run_unit(p, kf, rn, sh)
        except Exception as e:
            rc = 1; msg = f"FAILED {e!r}"; os.makedirs(OUT + "failed", exist_ok=True)
            open(OUT + f"failed/{p}_f{kf}_{'SHUF_' if sh else ''}{rn}.txt", "w").write(traceback.format_exc())
        print(f"  {p} f{kf} {'SHUF ' if sh else ''}{rn}: {msg} ({time.time()-t0:.0f}s)", flush=True)
    return rc

if __name__ == "__main__": sys.exit(main())
