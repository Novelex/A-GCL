# S0 — Environment & Immutable Baseline (evidence only)
Audit date: 2026-08-18 | Repo: /users/3171356m/A-GCL

## 1. Git
branch                : main
HEAD (full SHA)       : 906a494b076968768573a24c31804c6b0b1dd65b
HEAD subject          : "Updated Layr testing"
HEAD author/date      : Muhammad Hasan Masrur <234311073+Novelex@users.noreply.github.com>, Sun Aug 16 00:42:40 2026 +0100
git status --short    : (empty) — clean before AND after all S0 operations
recent commits        : 906a494, 416de8d, d8df3ed, ce9e785, 408a206
remote 'origin'       : https://github.com/qbmizsj/A-GCL.git (upstream, NOT the operator's fork)
                        local main is 17 commits ahead of origin/main;
                        github.com/Novelex/A-GCL main == 906a494 (in sync)

## 2. Interpreter / OS
python.executable     : /users/3171356m/A-GCL/.venv/bin/python
python.version        : 3.12.13 (Anaconda build, GCC 14.3.0)
platform              : Linux-5.14.0-687.25.1.el9_8.x86_64-x86_64-with-glibc2.34
OS                    : Rocky Linux 9.8 (Blue Onyx)
login host            : login1.cognition.gla.alces.network
scheduler             : SLURM (sinfo/sbatch present)
VIRTUAL_ENV           : /users/3171356m/A-GCL/.venv (active)

## 3. CPU
model                 : AMD EPYC 9334 32-Core Processor
cores                 : 32 (1 socket, 1 thread/core)
RAM                   : 376 GiB total
torch threads         : 32

## 4. GPU
login node            : NO GPU — nvidia-smi absent; torch.cuda.is_available() == False
compute node          : SLURM job 1868790, partition gpu-l40s, node02, State=COMPLETED ExitCode 0:0
GPU model             : NVIDIA L40S (46068 MiB, compute capability 8.9)
driver version        : 595.71.05
driver CUDA version   : 13.2
torch CUDA runtime    : 12.1 (torch 2.5.0+cu121)
cudnn                 : 90100
cuda.is_available     : True (on compute node), device_count = 1

## 5. Package versions (exact)
numpy            1.26.4
scipy            1.17.1
pandas           2.3.3
scikit-learn     1.9.0   (sklearn.__version__ 1.9.0)
torch            2.5.0+cu121
torch_geometric  2.6.1
torch_scatter    2.1.2+pt25cu121
torch_sparse     0.6.18+pt25cu121
(all resolve inside .venv; all match requirements.txt pins)
full snapshot    : /users/3171356m/agcl_audit_s0/pip_freeze.txt (66 packages)

## 6. compileall
command    : PYTHONPYCACHEPREFIX=<audit>/pycache python -m compileall -q .
exit code  : 0
output     : empty (no syntax errors)
tree after : clean

## 7. pytest
command   : PYTHONPYCACHEPREFIX=<audit>/pycache python -m pytest -v -rA -p no:cacheprovider
result    : 65 passed, 0 failed, 0 skipped, 0 errors, 22 warnings in 20.58s (exit 0)
failures  : NONE
warnings  : 20x torch_geometric 'data.DataLoader' deprecated;
            1x RuntimeWarning Mean of empty slice (unsupervised/embedding_evaluation.py:401);
            1x RuntimeWarning Degrees of freedom <= 0
collected : tests/ only (10 files). layertesting/ contributes 0 tests —
            its 10 test_*.py files are standalone main() scripts with no test_ functions.
log       : /users/3171356m/agcl_audit_s0/pytest_full.txt

## 8. Determinism (same seeded calc twice, same device)
CPU  (login1) : bitwise identical = True, max_abs_diff = 0.0
                run1 sum = run2 sum = 511.999969482422
                run1 sha256 = run2 sha256 = 6d50a7e7...0958
                numpy RandomState(1234) reproducible = True
GPU  (node02) : bitwise identical = True, max_abs_diff = 0.0
                run1 sum = run2 sum = 512.000000000000
CPU (in GPU job): bitwise identical = True
cross-device  : NOT identical (max_abs_diff 1.0) — expected and NOT required.
                CPU and CUDA torch.Generator streams differ, so the two runs draw
                different random numbers; this is not numerical drift.

## 9. Raw data inventory  (data -> /mnt/scratch/users/3171356m/A-GCL/data)
                 files  unique_ids  dup_ids  nonconforming_names
ASD_ADJ            455         455        0        0
ASD_NF             455         455        0        0
ASD_DW             455         455        0        0
NC_ADJ             501         501        0        0
NC_NF              501         501        0        0
NC_DW              501         501        0        0
TOTAL subjects   : 956 (455 ASD + 501 NC)
filename matching: ASD ADJ<->NF matched 455/455 (0 missing, 0 extra)
                   ASD ADJ<->DW matched 455/455; ASD triple ADJ&NF&DW = 455
                   NC  ADJ<->NF matched 501/501 (0 missing, 0 extra)
                   NC  ADJ<->DW matched 501/501; NC  triple ADJ&NF&DW = 501
duplicate ids    : 0 in every directory
ASD/NC id overlap: 0 for ADJ, NF, DW
zero-byte files  : 0
duplicate content: 0 (all 2868 raw files have distinct SHA-256)
sites            : ASD 23 sites, NC 22 sites (MaxMun_b present in ASD only, absent in NC)
other inputs     : data/subject_tr.csv (30402 B);
                   data/ALFF_need/rois_aal 968 files + download_manifest.csv (956 rows, no checksum column);
                   data/processed/ 5 derived .pt files (447 MB)
NOTE: no scientific values inspected — counts/names/sizes/hashes only (deferred to S1).

## 10. Hash manifests
location: /users/3171356m/agcl_audit_s0/
  manifest_raw_sha256.txt          2868 entries  (data/raw — the A-GCL raw inputs)
  manifest_alff_need_sha256.txt     968 entries  (data/ALFF_need)
  manifest_subject_tr_sha256.txt      1 entry    subject_tr.csv
                                    = 22112b5d660e8f20733d6684f48095023892b01bc285fae15a178666f29ec59d
  manifest_processed_sha256.txt       5 entries  (derived artefacts, recorded for freeze-detection)
  MANIFEST_ROOT_sha256.txt          anchor over all manifests + pip_freeze
pre-existing manifest: NONE found (ALFF download_manifest.csv is a download log, not checksums)
verification: sha256sum -c on raw manifest  -> exit 0, 0 mismatches
              sha256sum -c on ALFF manifest -> exit 0, 0 mismatches
              subject_tr.csv                -> OK
raw files altered: NO (read-only hashing)

## 11. Deviations from the literal instruction (disclosed)
127 .pyc files under __pycache__/ are TRACKED IN GIT (git ls-files count = 127).
Running `python -m compileall -q .` or pytest with default settings would have
rewritten those tracked files and dirtied the immutable baseline. Both were therefore
run with PYTHONPYCACHEPREFIX pointed outside the repo, and pytest with
-p no:cacheprovider. Commands are otherwise as specified. Working tree verified clean
before and after. All audit artefacts written to /users/3171356m/agcl_audit_s0/
(outside the repo); nothing in the repo or in data/ was created, modified or deleted.
No A-GCL training was run; no later-stage investigation performed.
