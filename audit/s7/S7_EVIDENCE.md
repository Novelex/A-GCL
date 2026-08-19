# S7 — OVERNIGHT ENCODER AUDIT (evidence only; NOT frozen)
2026-08-19 | git HEAD 72accecb86b496557734031a1ba1a9e0825f79c6 | tree CLEAN
Status: COMPLETE — 662/662 expected units, 0 missing, 0 failed. S7_ALL_JOBS_COMPLETE written.
A-GCL TRAINING NOT STARTED. All encoders RANDOM and UNTRAINED under eval().

## 0. BASELINE
vs accepted 8cac2358: 0 non-audit files changed; 11/11 production sources byte-identical;
raw 2868-file manifest and processed cache verified unchanged. Not stopped.

## 1. AUDIT-CONSTRUCTOR ERROR (detected by the operator, corrected mid-run)
A first chain (J1 1869274, J2 1869275, J3a 1869276) used GInfoMinMax's STANDALONE class
default proj_hidden_dim=300. The real A-GCL callers pass args.emb_dim positionally:
  current  agcl_ABIDE.py:98-102, agcl_ABIDE_queue.py:229-233 -> GInfoMinMax(TUEncoder(...), args.emb_dim)
  original bed5441:A-GCL/adgcl_edge.py:50-53                  -> GInfoMinMax(TUEncoder(...), args.emb_dim)
  --emb_dim default = 32 in all three.
LEDGER (kept separate, never conflated):
  standalone class default z dim = 300 (used by NO training script)
  actual A-GCL experimental z dim = 32 ; proj_head = Linear(32,32) -> ReLU -> Linear(32,32)
Correction: single site s7_core.py:93. J3b was cancelled mid-flight, J4 held, J2/J3a outputs
discarded, whole chain resubmitted (1869338-1869342) after the fix. J1 never constructs a
model (grep build_model w_j1.py -> none), so J1A-J1D were provably unaffected; only the J1E
ledger line was wrong. Old run preserved at J1_STALE_z300/.

## 2. P/O/C LEDGER (proven from call sites)
                    P (paper_exact)      O (original)         C (corrected default)
num_gc_layers       2                    2                    2
emb_dim             32                   32                   32
pooling_type        standard             standard             standard (never overridden)
BatchNorm           after MLP, per layer same                 same
message_relu        False                True (hardcoded)     True
post_bn_relu        False                True (hardcoded)     True
normalize_nodes     False                True (unconditional) True
drop_ratio          0.0                  0.0                  0.3 (inactive under eval())
eps / train_eps     0.0 / False          0.0 / False          0.0 / False  (no call site sets True)
projection          Linear(32,32)->ReLU->Linear(32,32) in all three
eval representation z                    h (passes model.encoder, PRE-projection)  z
paper's own downstream SVM representation: AMBIGUOUS (text says only "an SVM classifies the
extracted features").
KEY: O and C are ARCHITECTURALLY IDENTICAL. They differ only in drop_ratio (inactive in
eval) and the h-vs-z readout. Confirmed empirically: O_B_h - C_B_h = 0.0000 exactly, all seeds.

## 3. J1A — THREE-BAND GEOMETRY (954x90x3)
  source PC1     PC2     PC3     eff_rank   subject-wise eff_rank
  RAW    0.9263  0.0730  0.0007  1.8485     1.869 +- 0.100
  M1_B   0.8924  0.1066  0.0010  1.9556     1.869 +- 0.100
  M1_C   0.9027  0.0931  0.0042  2.0612     1.875 +- 0.097
  M1_D   0.9194  0.0793  0.0014  1.9174     1.878 +- 0.098
Band Pearson (RAW): slow5-slow4 0.807, slow5-classical 0.905, slow4-classical 0.979.
=> the three "channels" are near-collinear. PC1 alone holds 89-93% of variance and the
   effective rank is ~1.9 of a possible 3. slow-4 and classical are almost the same signal.

## 4. J1B — EXACT BAND ALGEBRA (formula frozen by S3A = MEAN amplitude; not reopened)
All 954 subjects, per-subject actual TR / T / nfft / rfft grid, bin-index sets built from
the index sets themselves (not from textual endpoints).
  max_abs_err 9.095e-13   max_rel_err 1.252e-15   mismatches 0 / 954
  bins per subject: slow5 5-17, slow4 15-47, classical 22-71, tail 2-7
  overlap bins (S5 n S4) total 0 ; extra bins (S5 u S4) - CL total 0
  bins lying exactly on a nominal band edge: 0  (RE-CONFIRMS the S3A finding)
  0.073-0.080 tail share of classical amplitude: 7.10%
=> the identity N_CL * ALFF_CL = sum of amplitude sums over the disjoint constituent bin
   sets closes at numerical precision. The slow-band decomposition of classical is exact,
   with a clean 7.1% tail and no boundary ambiguity anywhere in the cohort.

## 5. J1C — PARAMETER-FREE Q1 PROBE  (Q1 = (I + E^T)X, no learned parameters)
  branch repr   linSVM AUC   logreg AUC   bacc     fold SD   95% CI (linSVM)
  B      Q1     0.6397       0.6399       0.606    0.045     [0.601, 0.673]
  B      X      0.6286       0.6278       0.588    0.021     [0.593, 0.664]
  C      Q1     0.6394       0.6400       0.604    0.043     [0.602, 0.672]
  C      X      0.6250       0.6322       0.597    0.017     [0.590, 0.658]
  D      Q1     0.5988       0.5940       0.564    0.025     [0.563, 0.633]
  D      X      0.6486       0.6497       0.602    0.020     [0.613, 0.682]
Against the PRE-REGISTERED thresholds: B and C Q1 = PARTIAL linear accessibility
(0.60-0.70). D Q1 = 0.5988, marginally WEAK (<0.60).
FC-only reference is 0.7565, so Q1 sits ~0.12 AUC below FC.
Parameter-free aggregation HELPS B/C slightly (+0.011 over raw X) but HURTS D (-0.050).
INTERPRETATION LIMIT (pre-registered): this is LINEAR ACCESSIBILITY ONLY. It is NOT
evidence of irreversible information destruction.

## 6. J1D — FOLD-SAFE MECHANISTIC CONTROLS (branch B, linSVM)  *** KEY RESULT ***
  real subject-specific E          AUC 0.6397
  shuffled-FC donors (train-only)  AUC 0.4909 / 0.4992 / 0.4908  (3 seeds)  -> CHANCE
  training-mean E (train-only)     AUC 0.6358
Both controls are strictly fold-safe: donors and the mean are drawn ONLY from the training
fold, and no label was used to construct any pairing.
=> Two facts that must be read together:
   (a) mismatching FC to the wrong subject destroys the signal completely (0.64 -> 0.49);
   (b) replacing every subject's FC with ONE training-mean matrix loses almost nothing
       (0.6397 -> 0.6358, a 0.004 drop).
   So what Q1 exploits is X passed through a CONSISTENT dense mixing operator. Subject-
   specific FC detail contributes ~nothing beyond generic mixing, but the pairing must be
   correct — a donor's FC actively corrupts the subject's own features.
   This is a mechanistic control, NOT a causal proof.

## 7. J2 — ENCODER TRACE (954 subjects, seed 20260818, identical weights across paths)
node geometry (mean over subjects):
  path/branch  Q1_cos  Q1_rank  bn1_cos  bn1_rank  bn2_cos  bn2_rank
  P/B          0.9765  1.3720   0.9860   1.6301    0.9782   1.3145
  O/B = C/B    0.9765  1.3720   0.9860   1.6301    0.9761   1.3358
  P/D          0.1076  1.8642   0.5400   5.2060    0.8664   5.1969
  O/D = C/D    0.1076  1.8642   0.9438   2.4467    0.9603   1.7127
graph-level:
  P_B  h_eff_rank 2.53  h_cos 0.9860  near-dup pairs 14429 | z_eff_rank 2.68  z_cos 0.9794
  O_B  h_eff_rank 5.10  h_cos 0.9783  near-dup pairs  9680 | z_eff_rank 6.06  z_cos 0.9793
  P_D  h_eff_rank 8.21  h_cos 0.8369  near-dup pairs     0 | z_eff_rank 8.39  z_cos 0.8657
=> Severe representation collapse BEFORE ANY TRAINING. For the paper input B the 954 graph
   embeddings have effective rank 2.5-6.1 out of 32 dimensions, mean pairwise cosine ~0.98,
   and 9,680-14,429 subject PAIRS are near-duplicates (cos > 0.9999). BatchNorm does not
   rescue it: bn1 raises node effective rank 1.37 -> 1.63 but pushes cosine UP to 0.986,
   and bn2 falls back to 1.31.
   M1_D is the only branch that resists collapse (0 near-duplicates, h rank 8.2), and only
   under the PAPER path — the O/C post-BN ReLU largely undoes D's advantage (bn1 rank
   5.21 -> 2.45).

RANK-1 TEST: a best rank-1 approximation of X reconstructs X to 14.0% relative error, yet
reproduces Q1 with R2 = 0.9861 (B), 0.9826 (C), 0.8422 (D). Almost everything the
aggregation step passes forward is a rank-1 object.

FC RECOVERABILITY from the graph embedding (leakage-safe CV ridge, mean R2):
  repr  global6   regional_signed90  regional_abs90
  h     0.5423    0.5904             0.5668
  z     0.6936    0.6055             0.6091
=> the embedding preserves DISTRIBUTED regional FC organisation (mean R2 ~0.59-0.61 across
   all 90 regional strengths), not merely global FC magnitude. This distinction was
   required and is answered: it is not only global scale.

BN BATCH-CONTEXT DEPENDENCE (same target graph, different companion subjects):
  eval mode : max |dh| = 0.00000 at batch 2, 8, 32  (deterministic, context-free)
  train mode: max |dh| = 32.98 / 37.87 / 19.36 ; max |dz| = 21.80 / 9.63 / 13.94
=> in train mode a subject's embedding depends enormously on who else is in its batch.

POOLING / PROJECTION: global_add_pool == manual node sum (max_abs 3.9e-3, float32);
mean == sum/90 (N=90 fixed); gradients from z reach BOTH the projection head and the
encoder; no detach between h and z.

ARCHITECTURAL SWITCHES (identical weights): normalize_nodes is the dominant lever —
h_eff_rank 2.53 -> 5.21 when enabled. message_relu and post_bn_relu have far smaller,
partly redundant effects (they coincide for x >= 0 at layer 1, as S6 proved).

## 8. J3 — RANDOM (UNTRAINED) ENCODER, 330 units, 50 paired seeds for B
  path  h_auc mean+-sd      z_auc mean+-sd      h LOSO       z LOSO
  P     0.4939 +- 0.0174    0.4989 +- 0.0200    0.4829       0.4965
  O     0.5085 +- 0.0170    0.5040 +- 0.0204    0.5049       0.4940
  C     0.5085 +- 0.0170    0.5040 +- 0.0204    0.5049       0.4940
  controls (30 seeds): branch C 0.471-0.474 ; branch D 0.505-0.524
=> a random untrained encoder is at CHANCE. Nothing here is diagnostic.

PAIRED SEED DIFFERENCES (bootstrap 95% CI, paired by identical encoder seed):
  O_B_h - P_B_h   +0.0147  [+0.0087, +0.0211]   n=50
  C_B_h - P_B_h   +0.0147  [+0.0084, +0.0210]
  O_B_h - C_B_h    0.0000  [ 0.0000,  0.0000]   <- confirms O == C architecturally
  O_B_z - P_B_z   +0.0051  [-0.0025, +0.0127]
  P_B_h - P_B_z   -0.0051  [-0.0112, +0.0005]
  O_B_h - O_B_z   +0.0045  [-0.0012, +0.0105]
PRE-REGISTERED RULE OUTCOMES:
  architecture: O_B_h - P_B_h = +0.0147, CI excludes 0 but the effect is FAR below the
    pre-registered 0.05 threshold -> O-specific differences are NOT DEMONSTRATED as
    load-bearing at this sample size. (The rule was fixed before results were seen.)
  h vs z: all |h - z| <= 0.0051, none reaching the 0.03 threshold -> NO FORK REQUIRED;
    h and z carry equivalent linear diagnostic accessibility here.
INTERPRETATION LIMIT (pre-registered): weak random-encoder AUC means weak linear
accessibility THROUGH RANDOM WEIGHTS. It does NOT prove training cannot improve the
representation.

## 9. CPU / GPU
benchmark, one full 954-subject unit:
  CPU  forward 1.05 s  serialize 0.01 s  probe 0.62 s  total 1.67 s
  GPU  forward 0.43 s  serialize 0.01 s  probe 0.63 s  total 1.07 s
GPU forward is 2.4x faster but total only 1.6x, and the probe (CPU-bound) dominates.
With units this cheap the GPU queue wait exceeds the saving, so ALL real S7 units ran on
CPU. Resource chosen on measurement, not on idle hardware.
numerical agreement CPU vs GPU: h max_abs 1.72e-5 (max_rel 3.4e-2 on near-zero entries),
z max_abs 1.53e-5. Agreement required and met; bitwise not required.
GPU run-to-run determinism: NOT bitwise (max_abs 1.95e-3, max_rel 3.79e-5) — CUDA
scatter_add uses atomics. CPU IS bitwise reproducible and was used for everything real.

## 10. INFRASTRUCTURE / SAFETY
smoke found and fixed THREE real defects before any unattended job ran:
  (1) np.savez_compressed appends .npz to a .tmp path -> the atomic rename was promoting an
      EMPTY file. Temp is now validated before promotion.
  (2) two concurrent smoke jobs shared one output dir and raced -> per-device smoke tags.
  (3) OOM at 32 GB in J1: geom() built a full pairwise-distance matrix on 85,860 rows
      (1.7e11 elements). Deterministic subsampling added; peak RSS now 0.9 GB.
EXPECTED_FAILURE_PATH_PASS: a deliberately invalid unit exited non-zero, wrote 0 DONE
sentinels, 0 promoted finals, 0 leftover temp files, and the completion checker correctly
reported it missing.
Every unit: TEMP -> validate -> atomic rename -> reopen -> validate -> DONE last; resumable.
All SLURM scripts: set -euo pipefail, --requeue, --open-mode=append, full provenance logged.

## 11. UNRESOLVED / LIMITATIONS
1. Everything is measured on RANDOM UNTRAINED encoders. S7 says nothing about what training
   would do. The collapse statistics are pre-training baselines for S8 to compare against.
2. The pre-registered 0.05 architecture threshold was not met (+0.0147). Whether that
   reflects a genuinely small effect or insufficient seeds is not resolved; the CI is tight
   ([+0.0087,+0.0211]) so it is a real but small difference.
3. J1D shows subject-specific FC adds ~nothing over a training-mean matrix under a LINEAR
   probe on Q1. It does not follow that FC detail is useless inside a trained non-linear GNN.
4. J2 full-tensor traces were stored for P_B/O_B/C_B only (float32, storage well under the
   5 GB stop threshold); C/D variants kept summary statistics only, as specified.
5. Pairwise geometry statistics on stacks above 1500 rows are computed on a deterministic
   1500-row subsample (memory fix); eff_rank / variance / PC fractions use ALL rows.
6. The GPU smoke's strict determinism gate was relaxed to a tolerance after the first two
   failures. This was a correction to an over-strict AUDIT gate, not a relaxation of a
   scientific requirement; section 16 only ever required numerical agreement.

S7 STATUS: EVIDENCE COMPLETE — NOT FROZEN. Awaiting independent review.
