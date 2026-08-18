#!/bin/bash
#SBATCH --job-name=s0-cpu-probe
#SBATCH --partition=gpu-l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s0_cpu_probe_%j.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s0_cpu_probe_%j.err

set -uo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
mkdir -p "$PYTHONPYCACHEPREFIX"
VP=/users/3171356m/A-GCL/.venv/bin/python
cd /users/3171356m/A-GCL

echo "=== HOSTNAME ==="; hostname
echo "=== SLURM ALLOCATION ==="
echo "SLURM_JOB_ID           = ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION    = ${SLURM_JOB_PARTITION:-}"
echo "SLURM_CPUS_PER_TASK    = ${SLURM_CPUS_PER_TASK:-}"
echo "SLURM_CPUS_ON_NODE     = ${SLURM_CPUS_ON_NODE:-}"
echo "SLURM_MEM_PER_NODE(MB) = ${SLURM_MEM_PER_NODE:-}"
echo "affinity cpus          = $(nproc)"
echo "cgroup cpuset          = $(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || echo n/a)"
echo "cgroup mem.max         = $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)"
echo "node total RAM         = $(free -g | awk '/^Mem:/{print $2" GiB"}')"

echo "=== GIT BEFORE ==="
git -C /users/3171356m/A-GCL rev-parse HEAD
git -C /users/3171356m/A-GCL status --short | tee /users/3171356m/agcl_audit_s0/git_status_before_cpujob.txt
echo "(blank = clean)"
sha256sum /users/3171356m/agcl_audit_s0/git_status_before_cpujob.txt

echo "=== PYTHON / TORCH ==="
echo "python path: $VP"
$VP -c "
import sys, torch, platform
print('sys.executable :', sys.executable)
print('python.version :', sys.version.replace(chr(10),' '))
print('platform       :', platform.platform())
print('torch          :', torch.__version__)
print('torch.cuda.is_available:', torch.cuda.is_available())
print('torch threads  :', torch.get_num_threads())
"

echo "=== PYTEST (external pycache prefix, cache provider disabled) ==="
echo "PYTHONPYCACHEPREFIX=$PYTHONPYCACHEPREFIX"
$VP -m pytest -v -rA -p no:cacheprovider > /users/3171356m/agcl_audit_s0/pytest_slurm_cpu.txt 2>&1
echo "PYTEST_EXIT_CODE=$?"
tail -1 /users/3171356m/agcl_audit_s0/pytest_slurm_cpu.txt
echo "passed_lines=$(grep -cE '^PASSED ' /users/3171356m/agcl_audit_s0/pytest_slurm_cpu.txt)"
echo "failed_lines=$(grep -cE '^FAILED ' /users/3171356m/agcl_audit_s0/pytest_slurm_cpu.txt)"
echo "error_lines=$(grep -cE '^ERROR ' /users/3171356m/agcl_audit_s0/pytest_slurm_cpu.txt)"
echo "skipped_lines=$(grep -cE '^SKIPPED ' /users/3171356m/agcl_audit_s0/pytest_slurm_cpu.txt)"

echo "=== CPU DETERMINISM (same seed twice, compute node) ==="
$VP - <<'PY'
import torch, numpy as np, random, hashlib
def seeded_calc(device):
    torch.manual_seed(1234); np.random.seed(1234); random.seed(1234)
    g = torch.Generator(device=device).manual_seed(1234)
    x = torch.randn(512, 128, generator=g, device=device)
    w = torch.randn(128, 64,  generator=g, device=device)
    y = torch.relu(x @ w)
    return torch.nn.functional.softmax(y, dim=1)
y1 = seeded_calc("cpu"); y2 = seeded_calc("cpu")
print("torch.equal        :", torch.equal(y1, y2))
print("max_abs_diff       :", (y1-y2).abs().max().item())
print("run1_sum           :", f"{y1.sum().item():.12f}")
print("run2_sum           :", f"{y2.sum().item():.12f}")
print("run1_sha256        :", hashlib.sha256(y1.numpy().tobytes()).hexdigest())
print("run2_sha256        :", hashlib.sha256(y2.numpy().tobytes()).hexdigest())
PY

echo "=== GIT AFTER ==="
git -C /users/3171356m/A-GCL rev-parse HEAD
git -C /users/3171356m/A-GCL status --short | tee /users/3171356m/agcl_audit_s0/git_status_after_cpujob.txt
echo "(blank = clean)"
sha256sum /users/3171356m/agcl_audit_s0/git_status_after_cpujob.txt
diff /users/3171356m/agcl_audit_s0/git_status_before_cpujob.txt /users/3171356m/agcl_audit_s0/git_status_after_cpujob.txt && echo "GIT_STATUS_IDENTICAL=YES" || echo "GIT_STATUS_IDENTICAL=NO"
echo "=== CPU PROBE DONE ==="
