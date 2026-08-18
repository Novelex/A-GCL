#!/bin/bash
#SBATCH --job-name=s0-gpu-probe
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s0_gpu_probe_%j.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s0_gpu_probe_%j.err

set -uo pipefail

echo "=== HOSTNAME ==="; hostname
echo "=== nvidia-smi ==="; nvidia-smi
echo "=== nvidia-smi query ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv

echo "=== TORCH CUDA + GPU DETERMINISM ==="
/users/3171356m/A-GCL/.venv/bin/python - <<'PY'
import torch, platform
print("torch:", torch.__version__, "| torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("cudnn:", torch.backends.cudnn.version())

def seeded_calc(device):
    torch.manual_seed(1234)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(1234)
    g = torch.Generator(device=device).manual_seed(1234)
    x = torch.randn(512, 128, generator=g, device=device)
    w = torch.randn(128, 64,  generator=g, device=device)
    y = torch.relu(x @ w)
    y = torch.nn.functional.softmax(y, dim=1)
    return y.sum(), y

print("--- GPU determinism (same device, twice) ---")
s1, y1 = seeded_calc("cuda")
s2, y2 = seeded_calc("cuda")
same = torch.equal(y1, y2)
print("gpu_run1_sum:", f"{s1.item():.12f}")
print("gpu_run2_sum:", f"{s2.item():.12f}")
print("gpu_bitwise_identical:", same)
print("gpu_max_abs_diff:", (y1-y2).abs().max().item())

print("--- CPU determinism (inside GPU job, for reference) ---")
c1, cy1 = seeded_calc("cpu")
c2, cy2 = seeded_calc("cpu")
print("cpu_run1_sum:", f"{c1.item():.12f}")
print("cpu_run2_sum:", f"{c2.item():.12f}")
print("cpu_bitwise_identical:", torch.equal(cy1, cy2))

print("--- CPU vs GPU cross-device (NOT required to match) ---")
print("cpu_vs_gpu_max_abs_diff:", (cy1.cpu()-y1.cpu()).abs().max().item())
print("cpu_vs_gpu_bitwise_identical:", torch.equal(cy1.cpu(), y1.cpu()))
PY
echo "=== PROBE DONE rc=$? ==="
