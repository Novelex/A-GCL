# S8 START REPORT — View Learner Audit + Training Readiness
date: 2026-08-20T18:50:54+01:00
git HEAD: f6c49d49761ef3bf82b3d59798ee1e8ace6b62dc
working tree: 0 modified files (0 = clean)
production surface vs 8cac2358: 0 non-audit files changed (verified)
python: Python 3.12.13
torch: 2.5.0+cu121
CUDA available on login node: False (GPU work goes via SLURM; cluster GPUs currently saturated)

## INPUT HASHES (verified at start)
cohort sha256 (954 ids)   : aca3d945f7d89ccb1cc6fb46ca01f6036ccf036d81d120d2b09fe4bd0d1d68c9
X_sources.npz             : dc10bf36c4124aa7f214ab6bbc5a89288adc03da747e3950485bb15c0da333a9
M1_B graph cache          : 312266b23ecf1348ce083cb25d9c5e5a51d5595dab9ce5639875a51c12f1f844
M1_C graph cache          : 17338f14073b82f6793beb81c93314a1f94e35f58e92b32488ecba8ff59e0c9d
M1_D graph cache          : 59a4c88a2c3797882727fd1c5cd323fdb6608e808641ac2025cbdc64075fd397
repo processed cache      : verified OK vs S0 manifest (data_dense_v3.pt untouched, NOT used)
raw 2868-file manifest    : exit 0

## THE THREE TRAINING CONTRACTS (proven from source, locked before any result)
P = fork paper_exact profile (agcl_ABIDE_queue.py + training_profiles.py)
    arch: normalize_nodes=F, message_relu=F, post_bn_relu=F, drop=0.0
    mask: ordered Concrete, ASYMMETRIC independent noise per directed edge, T=1.0
    reg : mean(mu), sign MINUS (view_loss = InfoNCE - 2.0*mean(mu) + 0.4*cr)
    bank: PaperMemoryBank_Q(256,32), zero-init counts as negatives, push BEFORE cr (bug-5 literal)
    cr  : calc_regloss_paper, temp 1.0, no same-subject exclusion
    InfoNCE: temp 1.0, sym=False ; model_loss = InfoNCE + 0.4*cr ; eval z
O = ORIGINAL author's agcl_ABIDE_queue.py (commit 08339b7, sjzhang)
    arch: normalize_nodes=T, message_relu=T, post_bn_relu=T, drop=0.3
    mask: sigmoid((logistic_noise+logits)/1.0), ASYMMETRIC, resampled detached in model half
    reg : mean(1-mask) [sampled, not mu], view_loss = InfoNCE - 2.0*reg + 0.4*cr
    bank: zero-init FIFO queue(256), no validity/no subject ids, push in VIEW half BEFORE cr
    cr  : original calc_regloss == positive - logsumexp(all memory), temp 0.1
    InfoNCE: temp 0.2, sym=True ; *** model_loss = InfoNCE - 0.4*cr  (MINUS — original sign) ***
    eval h (passes model.encoder to the evaluator)
C = fork corrected defaults
    arch: same flags as O (T/T/T, drop 0.3)
    mask: symmetrized logits + SHARED symmetric logistic noise, T=1.0
    reg : 'budget' -> view_loss = InfoNCE + 2.0*mean(mu) + 0.4*cr  (sign PLUS)
    bank: MemoryBank_Q with validity mask + same-subject exclusion, push at END of batch
    InfoNCE: temp 0.2, sym=True ; model_loss = InfoNCE + 0.4*cr ; eval z
NOTE 1: the original model update SUBTRACTS the memory term; both fork profiles ADD it.
NOTE 2: O's cr uses exp-then-log (unstable); the audit uses the mathematically identical
        logsumexp form (calc_regloss_paper) — the fork's own documented hygiene policy.
        This is the ONLY deviation from the original lines, and it is recorded.
Common to all: Adam lr 5e-4 (model and view), batch 32, emb 32, proj z 32, GIN 2 layers,
mlp_edge_model_dim 64, reg_lambda 2.0, cr_lambda 0.4, max_length 256.

## RULES
No production modification. Nothing written to data/, processed/, or any production cache.
All outputs under /users/3171356m/agcl_audit_s0/s8/. Every job logs job ID, host, seed,
git hash, environment. TEMP -> validate -> rename -> DONE on every unit.
