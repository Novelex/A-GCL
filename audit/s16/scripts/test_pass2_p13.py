"""Pass 2, P13: the requeue claim must be true.

D45: on SIGUSR1 the worker wrote STATUS state='requeued' and exited 0. SLURM does
     not requeue a job that exited 0 (--requeue covers node failure and preemption,
     not a clean exit), so the unit stopped for good while the record claimed it
     would resume. Worse, --signal=B:USR1@300 delivers to the BATCH SHELL, not to
     the python child, so the handler could never fire at all: the whole
     graceful-stop path was unreachable.

No cluster job is submitted, cancelled or requeued by this test — scontrol is
mocked and its arguments are asserted."""
import sys, os, json, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OK=[]; HERE=os.path.dirname(os.path.abspath(__file__))
def check(c,m): OK.append(bool(c)); print(f"  [{'PASS' if c else 'FAIL'}] {m}")
import s16_worker as W

CAPTURED=[]
class FakeR:
    def __init__(self, rc, err=""): self.returncode=rc; self.stderr=err; self.stdout=""
def fake_run(rc):
    def _f(cmd, **kw):
        CAPTURED.append(list(cmd)); return FakeR(rc, "" if rc==0 else "Invalid job id")
    return _f

def capture_status():
    got={}
    def st(jd, state, done, total, extra=None):
        got.update(state=state, done=done, total=total, extra=extra or {})
    return got, st

print("=== 1. no SLURM_JOB_ID -> no requeue is claimed ===")
for k in ("SLURM_JOB_ID","SLURM_JOBID"): os.environ.pop(k, None)
got, st = capture_status()
ok = W.requeue_self("/tmp", 3, 9, st)
check(ok is False, "returns False")
check(got["state"]=="stopped_not_requeued", f"state={got['state']!r} (never 'requeued')")
check("no requeue was attempted" in got["extra"]["reason"], "the reason is explicit")

print("\n=== 2. scontrol succeeds -> requeue is claimed, with evidence ===")
os.environ["SLURM_JOB_ID"]="12345_7"
CAPTURED.clear(); orig=subprocess.run; subprocess.run=fake_run(0)
try:
    got, st = capture_status(); ok = W.requeue_self("/tmp", 3, 9, st)
finally: subprocess.run=orig
check(ok is True, "returns True")
check(CAPTURED==[["scontrol","requeue","12345_7"]],
      f"issued exactly one explicit requeue for this job: {CAPTURED}")
check(got["state"]=="requeued", f"state={got['state']!r}")
check(got["extra"]["slurm_job_id"]=="12345_7" and got["extra"]["scontrol_rc"]==0,
      f"the job id and scontrol return code are recorded: {got['extra']['slurm_job_id']}, "
      f"rc={got['extra']['scontrol_rc']}")

print("\n=== 3. scontrol FAILS -> the failure is reported, not hidden ===")
CAPTURED.clear(); subprocess.run=fake_run(1)
try:
    got, st = capture_status(); ok = W.requeue_self("/tmp", 3, 9, st)
finally: subprocess.run=orig
check(ok is False, "returns False")
check(got["state"]=="stopped_not_requeued", f"state={got['state']!r}")
check("must be resubmitted by hand" in got["extra"]["reason"],
      "the record says the unit is INCOMPLETE")
check(got["extra"]["scontrol_rc"]==1 and "Invalid job id" in got["extra"]["scontrol_stderr"],
      "scontrol's own error text is preserved")
os.environ.pop("SLURM_JOB_ID", None)

print("\n=== 4. the exit code distinguishes the two outcomes ===")
src=open(f"{HERE}/s16_worker.py").read()
check("sys.exit(0 if ok else 3)" in src,
      "exit 0 only on a real requeue; a failed stop exits 3 so sacct shows it")
check('status(jd,"requeued",done,len(folds)); ev.set(); sys.exit(0)' not in src,
      "the old unconditional 'requeued' + exit 0 is gone")

print("\n=== 5. the signal actually reaches the python child ===")
for f in ("sb_wgin.sh","sb_bnt.sh","sb_ctrlu.sh"):
    s=open(f"{HERE}/{f}").read()
    has=("trap _fwd USR1" in s and 'kill -USR1 "$PY_PID"' in s
         and 'PY_PID=$!' in s and 'exit "$RC"' in s)
    check(has, f"{f} forwards SIGUSR1 to the child and propagates its exit status")
    check(subprocess.run(["bash","-n",f"{HERE}/{f}"]).returncode==0, f"{f} parses")

print(f"\n{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
