#!/bin/bash
#SBATCH --job-name=cuda-diag
#SBATCH --partition=gpu-l40s
#SBATCH --exclude=node07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/users/3171356m/A-GCL/logs/%x_%j.out
#SBATCH --error=/users/3171356m/A-GCL/logs/%x_%j.err

set -euo pipefail

echo "=== nvidia-smi -q ECC section ==="
nvidia-smi -q | grep -A 5 "ECC Errors" || true

/users/3171356m/A-GCL/.venv/bin/python -c "
import torch
print('cuda available:', torch.cuda.is_available())
print('device name:', torch.cuda.get_device_name(0))
x = torch.zeros(5).cuda()
print('plain zeros().cuda() OK:', x)
y = torch.zeros((256,32), requires_grad=True)
print('cpu requires_grad zeros OK')
z = y.to('cuda')
print('to(cuda) with requires_grad OK:', z.shape, z.is_leaf, z.grad_fn)
"
