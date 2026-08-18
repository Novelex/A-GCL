# S0 ADDENDUM — two gaps closed (evidence only; S1 not begun)
Date 2026-08-18 | HEAD 906a494b076968768573a24c31804c6b0b1dd65b (unchanged)

## GAP 1 — THE TWO ALFF BRANCHES

### Baseline change detected
ALFF_func_proc/ was uploaded into the repo at 2026-08-18 15:40, AFTER the S0 snapshot.
Working tree is therefore no longer clean: `?? ALFF_func_proc/` (untracked).
HEAD unchanged; no tracked file altered. Not covered by .gitignore (which lists .venv/, data/ only).

### Branch A — method 1, ALFF-ROI-first
path            /users/3171356m/A-GCL/ALFF_func_proc/method1/alff_roi_first.npz
size / sha256   2130074 B / 647a1d872398ebea26a0b3113aea35765247fc7fcdebd482c2ecbb9f34cc3cf4
keys            file_ids(954,<U16), alff(954,90,3,float64), tr(954,float64)
subjects        954 rows, 954 unique IDs, 0 duplicates, sorted order
ASD / NC        455 ASD / 499 NC   (by storage location in raw/ASD_ADJ vs raw/NC_ADJ;
                dx_group absent in this file — no diagnosis codes interpreted)

### Branch B — method 2, ALFF-voxel-first
path            /users/3171356m/A-GCL/ALFF_func_proc/method2/alff_voxel_first.npz
size / sha256   2130074 B / 19b95826f4b232fabb4587b806c56d5913a5f1fe09c9350239158608a9680126
keys            file_ids(954,<U16), alff(954,90,3,float64), tr(954,float64)
subjects        954 rows, 954 unique IDs, 0 duplicates, sorted order
ASD / NC        455 ASD / 499 NC (same basis as above)

### A vs B
intersection |A∩B| = 954 ; only-in-A = 0 ; only-in-B = 0 ; ID order identical = True
integrity: files are DISTINCT — different sha256; alff arrays differ in 257577/257580
elements; `tr` arrays byte-identical; file_ids identical. 0 NaN, 0 Inf in both.
(identity check only — no ALFF values interpreted)

### Branch OLD — rois_aal / ".1D by ROI id" route
path            /users/3171356m/A-GCL/data/ALFF_need/alff_new.npz
size / sha256   4200944 B / 69da61aac8411e6cc2a9164a712539735dd6a9aff06bdda055820c27fcc5a22b
keys            file_ids(956), alff(956,90,3), malff(956,90,3), dx_group(956,int64), ok(956,bool)
subjects        956 rows, 956 unique, 0 duplicates, sorted
ASD / NC        455 ASD / 501 NC (by storage)
ok field        956 True / 0 False
source inputs   data/ALFF_need/rois_aal/*.1D = 956 files, 0 missing / 0 extra vs cohort
siblings        alff_new_combat.npz (956; adds `site`, drops malff)
                alff_correlation.npz (r(90,3) + file_ids(956))
structural delta vs new branches: OLD has malff + dx_group + ok and NO tr;
                new branches have tr and NO malff/dx_group/ok. Both alff are (N,90,3).

### Against the frozen 956 ADJ cohort
branch A : matched 954/956 ; missing 2 -> CMU_b_0050669, Leuven_1_0050706 ; extra 0
branch B : matched 954/956 ; missing 2 -> CMU_b_0050669, Leuven_1_0050706 ; extra 0
OLD      : matched 956/956 ; missing 0 ; extra 0
OLD∖A = OLD∖B = {CMU_b_0050669, Leuven_1_0050706} ; A∖OLD = B∖OLD = empty
Both dropped subjects are NC (NC 501 -> 499; ASD unchanged at 455).

### Reason for 956 vs 954 — DISCOVERED (primary source, not inferred)
Generating logs: /users/3171356m/muhammad/GraSTIACL/layer_testing/logs/dual_alff_full_1868526_*.out
  "FAILED: CMU_b_0050669: ROI(s) with zero valid voxels: [87]"
  "FAILED: Leuven_1_0050706: ROI(s) with zero valid voxels: [28]"
  aggregate across chunks: succeeded=954 of total=956; exactly 2 failures.
Both subjects DO have QC-passed inputs (qc_passed_func_proc holds
CMU_b_0050669_func_preproc.nii.gz and Leuven_1_0050706_func_preproc.nii.gz; 956 .nii.gz + 1 csv),
and both downloaded fine on the .1D route (logs/agcl-download-alff-rois_1867348.err
[467/956] and [524/956] "downloaded"). So the exclusion is an atlas-coverage failure at
ALFF computation, not a missing input.
Corroborating narrative: GraSTIACL/docs/STAGE6C_ALFF_THREE_SOURCE_COMPARISON.md —
"the two zero-ROI func_preproc subjects — excluded, NOT imputed, no .1D substitution".
PROVENANCE CAVEAT: both npz files and all the evidence above were produced in a
DIFFERENT project tree (/users/3171356m/muhammad/GraSTIACL) and copied into A-GCL.
No generating script or log for them exists inside the A-GCL repo. A-GCL contains no
reference to 954 anywhere.

## GAP 2 — SLURM CPU COMPUTE-NODE CONFIRMATION
job              1868799, partition gpu-l40s, State COMPLETED, ExitCode 0:0, Elapsed 00:00:20
hostname         node01.cognition.gla.alces.network
allocated CPUs   SLURM_CPUS_PER_TASK=8, CPUS_ON_NODE=8, nproc=8 (node has 376 GiB total)
allocated RAM    SLURM_MEM_PER_NODE = 32768 MB (32 GiB)
python path      /users/3171356m/A-GCL/.venv/bin/python  (same .venv)
python version   3.12.13 (Anaconda, GCC 14.3.0)
platform         Linux-5.14.0-687.25.1.el9_8.x86_64-x86_64-with-glibc2.34
torch            2.5.0+cu121 ; cuda.is_available False (CPU job, as intended) ; threads 8
pytest           PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm,
                 -p no:cacheprovider -> exit 0
                 65 passed, 0 failed, 0 skipped, 0 errors, 22 warnings, 13.75s
determinism      torch.equal = True ; max_abs_diff = 0.0
                 run1_sum = run2_sum = 511.999969482422
                 run1_sha256 = run2_sha256 = 6d50a7e77fe4774a81881d14e5745c6014ab422fca984db750327d0c5d7e0958
                 (byte-identical to the login-node S0 CPU result -> reproducible across hosts)
git before/after HEAD identical; status identical both times ("?? ALFF_func_proc/");
                 status files hash-equal 5e581b3a...593e; GIT_STATUS_IDENTICAL=YES

## Artefacts
/users/3171356m/agcl_audit_s0/
  alff_branch_comparison.txt, manifest_alff_func_proc_sha256.txt (verified OK),
  s0_cpu_probe.slurm, s0_cpu_probe_1868799.out, pytest_slurm_cpu.txt,
  git_status_before_cpujob.txt, git_status_after_cpujob.txt
No code modified. No ALFF scientific values interpreted. S1 not begun.
