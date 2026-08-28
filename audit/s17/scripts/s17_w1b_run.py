"""S17 Wave 1b array task: ONE fold per SLURM task. Writes only audit/s17/runs/prod/.

Identical to Wave 1 in arms, seeds, folds and the PROD policy. The ONLY difference
is that probe_honest is reported at three widths (native 2880, PCA-32, PCA-64), with
the PCA fitted on tr_enc only. No gate, threshold or arm changes.

Per-fold try/except, a POISON marker and scancel of the whole array if more than 5%
of the attempted folds have failed. Submits nothing itself.

Usage (from the array):  python s17_w1b_run.py $SLURM_ARRAY_TASK_ID
"""
import os, sys, json, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, "/users/3171356m/A-GCL/audit/s16/scripts")
import s17_worker as W17

NS = "prod"
POISON_FRAC = 0.05                      # >5% of ATTEMPTED folds failed -> wave broken


def poison_path(ns): return W17.root(ns) + "POISON"
def fail_dir(ns):
    d = W17.root(ns) + "failed/"; os.makedirs(d, exist_ok=True); return d


def check_poison(ns):
    """Abort the whole array once the failure rate passes the threshold."""
    import glob
    done = len(glob.glob(W17.jobs_dir(ns) + "*/fold_*.json"))
    bad = len(glob.glob(fail_dir(ns) + "*.json"))
    att = done + bad
    if att >= 20 and bad / att > POISON_FRAC:
        msg = (f"{bad}/{att} attempted folds FAILED ({100*bad/att:.1f}% > "
               f"{100*POISON_FRAC:.0f}%) — the wave is broken")
        open(poison_path(ns), "w").write(msg + "\n")
        aid = os.environ.get("SLURM_ARRAY_JOB_ID")
        print("POISON: " + msg, file=sys.stderr, flush=True)
        if aid:
            print(f"POISON: cancelling array {aid}", file=sys.stderr, flush=True)
            os.system(f"scancel {aid}")
        return True
    return False


def main():
    k = int(sys.argv[1])
    tasks = W17.wave1b_tasks(NS)
    if k >= len(tasks):
        print(f"no task {k} (only {len(tasks)})"); return 0
    u, fold = tasks[k]
    uid = W17.unit_id(u)
    W17.ensure(NS)
    assert "/audit/s17/" in W17.root(NS) and "/audit/s16/" not in W17.root(NS)

    if os.path.exists(poison_path(NS)):
        print(f"REFUSING: POISON present — {open(poison_path(NS)).read().strip()[:160]}",
              file=sys.stderr)
        return 3

    print(f"task {k}: {uid} / {fold}  ns={NS} root={W17.root(NS)}", flush=True)
    t0 = time.time()
    try:
        W17.run_unit(u, ns=NS, only_fold=fold)
        print(f"OK {uid}/{fold} in {time.time()-t0:.0f}s", flush=True)
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        rec = dict(unit=uid, fold=fold, task=k, error=repr(e), traceback=tb[-4000:],
                   node=os.environ.get("SLURMD_NODENAME"), wall_s=round(time.time()-t0, 1))
        with open(fail_dir(NS) + f"{uid}__{fold}.json", "w") as fh:
            json.dump(rec, fh, indent=1, default=str)
        print(f"FAILED {uid}/{fold}: {e!r}\n{tb}", file=sys.stderr, flush=True)
        check_poison(NS)
        return 1


if __name__ == "__main__":
    sys.exit(main())
