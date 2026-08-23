# S13 — BRAIN NETWORK TRANSFORMER ON ABIDE: PRE-REGISTERED PROTOCOL
Branch s13-bnt. Written and committed BEFORE any S13 result exists.
Reference: Kan et al., Brain Network Transformer, NeurIPS 2022.
Amendments go to PROTOCOL_AMENDMENTS.md with timestamp + justification, never silent.

MISSION: replace the WGIN encoder with an architecture that does not destroy
connectivity information, and measure whether it beats the frozen linear baseline.

## FROZEN REFERENCE LINES (printed in every table, NEVER recomputed)
  LinearSVC on 4005 FC edges : OOF 0.7565  LOSO 0.7432
  ridge logistic             : OOF 0.7561  LOSO 0.7406
  best MLP (h64, wd 1e-3)    : OOF 0.7246  LOSO 0.7090
  best WGIN (S12A5 arm A)    : OOF 0.6307

## WHY THIS ARCHITECTURE (S12B measurements, not re-derived)
  B1 rank-3 node features        -> BNT uses the full FC row per node
  B2 Linear(93->32) loses 38%    -> BNT hidden H=128 >= D, no squeeze
  B3 BatchNorm over subjects     -> BNT uses LayerNorm only
  B4 F.normalize kills magnitude -> absent in BNT
  B5 global_add_pool -> chance   -> OCREAD keeps ROI identity
S12B Arm C S0 = 0.747 retention 0.996. That is BNT's input.

## ISOLATION (non-negotiable)
Branch s13-bnt. Creates ONLY audit/s13/**. Imports READ-ONLY: s11_core (K),
s12a1_core (A1), s12a5_core (M). Modifies nothing under audit/s0..audit/s12b and
nothing in agcl_ABIDE*.py, unsupervised/, datasets/. Deleting this branch must
leave the repo byte-identical to main.

## GATE 0 — DATA AND CACHE (BLOCKING)
New namespace only: cache/data_s13_<sha256(S11 manifest)[:16]>.pt. Never load
data.pt / data_dense_v3.pt / any earlier .pt. Existing target file is DELETED and
rebuilt. NO try/except around loading, no fallback, no default cohort.
HARD ASSERTS (exit non-zero on any failure): len==954; ASD/NC == S11 manifest
exactly; FC.shape==(954,90,90); ALFF.shape==(954,90,3); every FC[i] symmetric
<1e-6; diag(FC[i])==1.0 exactly; no NaN/Inf in FC or ALFF; subject-ID order hash
== S11; label hash == S11; fold hash == frozen S3C (LOADED, never regenerated);
FC row-major edge order hash == S11 X_fc; ALFF band order hash == frozen M1_B.
Records all hashes, git SHA, hostname, python/torch/numpy/sklearn, pip freeze,
OMP_NUM_THREADS, timestamp -> GATE0_DATA.md.

## GATE 1 — INSTRUMENT CALIBRATION (BLOCKING)
THE probe for all of S13 is K.probe_pipe, used unchanged.
  R1 raw FC 4005, K.probe_pipe   -> CEILING_PROBE
  R2 raw FC 4005, LinearSVC grid-C (frozen path)
  R3 raw FC 4005, LABELS PERMUTED
  R4 ALFF(3) flattened only      -> node-only floor
PASS: R2 in [0.741,0.772]; R1 >= R2-0.03; R3 in [0.47,0.53]. Any fail -> STOP,
do not train. -> GATE1_INSTRUMENT.md.

## DATA SPECIFICATION (exact)
Loaded via A1.load_gate() + A1.load_tensors(df). FC is NOT re-derived.
FC   : float32 [954,90,90]; diagonal STAYS 1.0; no row z-scoring; no /max|FC|;
       no thresholding, sparsification, or negative-zeroing.
ALFF : float32 [954,90,3] = Xold from A1.load_tensors (the frozen 90-ROI ALFF).
       Default scaling: z-score ACROSS SUBJECTS per (ROI,band), mu/sd computed on
       THE TRAIN INDICES OF EACH FOLD ONLY, never on all 954, sd clamped >=1e-6.
       Arm T4 alternative: per-subject per-band min-max (production v3 recipe).
X    : concat on last dim -> [954,90,D]; D=90 (T1, FC row only);
       D=93 (T2/T4/T5/T6, FC row + ALFF).
y    : [954] int, ASD=1 NC=0. NOTE: original A-GCL code uses ASD=0 HC=1. AUC is
       unaffected but SENSITIVITY AND SPECIFICITY ARE SWAPPED versus the paper's
       table. Every sen/spe column is labelled with its convention.
Folds: K.folds_ordinary() (5 stratified) + K.folds_loso(y) = 24, frozen, never
       resampled. Inner val: train_test_split(test_size=0.20, stratify=y[tr],
       random_state=M.BASE) — identical to train_fold5. Val is used ONLY for early
       stopping and config selection; test is touched exactly once per fold.

## MODEL (bnt_core.py) — exact
Contract: BNTModel(arm, seed, K_clusters, D, H=128); repr_of(batch, edge_vec)
-> [B,K*H]; forward(batch, edge_vec) -> (repr, logits[B]). X travels on the batch.
FIXED: H=128 (assert H>=D), L=2 layers, 4 heads, d_k=32, FFN expansion 2,
attn/ffn/head dropout 0.30, GELU in FFN, LeakyReLU(0.1) in head.
SWEPT (only these two): K_clusters in {2,4,8}; weight_decay in {1e-4,1e-3}.
FORWARD: inp=Linear(D,H) (EXPANSION, never compression) -> 2x PRE-NORM residual
blocks [A=MHSA(LN(Z)); Z=Z+Drop(A); Z=Z+Drop(FFN(LN(Z)))] with Q,K,V=Linear(H,H,
bias=False) reshaped [B,4,90,32], attn=softmax(QK^T/sqrt(32)) then Dropout, out
merged and projected Linear(H,H) -> Z_L=LayerNorm(Z) -> OCREAD -> head.
*** FC edge weights are NEVER added to attention scores. FC enters ONLY through
node features. *** (BrainGB and S12B both show score-injection degrades complete
graphs.)
OCREAD: C=xavier_uniform_(empty(K,H)) once at __init__ (seeded); E=gram_schmidt(C)
with unit-L2 rows; registered as a BUFFER (requires_grad False, in state_dict);
P=softmax(Z_L @ E.T, dim=-1) over K; Z_G=P^T @ Z_L -> [B,K,H]; repr=reshape(B,K*H).
HEAD: LayerNorm(K*H) -> Drop(0.30) -> Linear(K*H,32) -> LeakyReLU(0.1) ->
Drop(0.30) -> Linear(32,1) -> squeeze(-1).
Init: xavier_uniform_ on all Linear weights, bias zero, seeded by the arm seed.
Expected params at H=128,D=93,K=4 ~294,000. PRINTED. If > 1,030,000 (the EdgeMLP
that overfit 96-100% of folds) -> ABORT before running anything.

## GATE 2 — NINE BLOCKING TESTS (all must PASS) -> GATE2_MODEL.md
1 orthonormality max|E E^T - I| < 1e-5; E is a buffer in state_dict, requires_grad
  False. 2 P.sum(-1)==1 (<1e-6); Z_G [B,K,H]; repr [B,K*H]. 3 attention rows sum
  to 1 (<1e-6), shapes [B,4,90,90], 2 layers x 4 heads confirmed by inspection.
4 ROI-PERMUTATION EQUIVARIANCE: permute ROI order AND the connection-profile
  columns identically -> Z_L equal up to that permutation (<1e-4). Catches a
  transposed profile. 5 K=1 degenerate -> OCREAD equals a MEAN readout (<1e-5;
  P is all-ones so Z_G = 90 x mean, compared after dividing by 90 — declared).
6 gradient reaches inp, each MHSA, each FFN, each LayerNorm and the head at step 1
  (non-zero grad norm, no dead branch); E has NO gradient. 7 OVERFIT ONE BATCH:
  32 subjects, all dropout 0.0, wd 0, label smoothing OFF (smoothing floors BCE at
  0.325 — declared), 500 steps -> train AUC 1.000 and loss < 0.01. IF THIS FAILS
  THE MODEL IS BROKEN; stop, nothing else is readable. 8 no NaN/Inf in any
  intermediate for 8 subjects; same seed -> bitwise-identical logits over two CPU
  runs; save/reload -> bitwise-identical repr and logits. 9 Gate-0 hashes
  re-verified at the START OF EVERY JOB, abort on mismatch.

## TRAINING (exact)
loss = BCEWithLogitsLoss on SMOOTHED targets t = y*0.90 + 0.05 (smoothing 0.1);
AUC always reported against RAW y, never t.
optimizer = AdamW(lr=1e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=<swept>)
  — decoupled, NOT Adam+L2 (the old EdgeMLP used Adam+wd, effectively unregularised).
batch_size = 64 (BNT paper; the old harness used 32 — DEVIATION LOGGED).
max_epochs = 200; early stop on VALIDATION AUC, patience 20, min_delta 1e-6;
never validation accuracy, never test. No lr schedule.
grad clip = clip_grad_norm_(1.0); clip events COUNTED AND LOGGED per epoch.
seeds = 20260818, 20260819, 20260820. Device CPU, torch.set_num_threads(4),
OMP_NUM_THREADS=4, torch.use_deterministic_algorithms(True).
Batch order from torch.Generator().manual_seed(seed).
PER-EPOCH: train/val loss, train/val AUC, lr, grad norm per group
(inp/mhsa/head), clip_events, epoch wall seconds.
PER-FOLD: best_epoch, best_val_auc, epochs_run, TRAIN-MINUS-VAL AUC GAP at the
selected epoch, verdict OVERFIT (gap>0.15) / UNDERFIT (train_auc<0.65) / HEALTHY,
parameter L2 movement from init per group.
INTEGRITY per fold: loss finite every epoch; loss(best) < loss(epoch 1); no
NaN/Inf gradient; test never touched before final scoring; checkpoint reload
reproduces best_val_auc bitwise.

## ARMS AND GRID
STAGE 1 (main sweep): T2 = FC row + ALFF(z across subjects), 90 ROIs, D=93.
  K in {2,4,8} x wd in {1e-4,1e-3} x 3 seeds = 18 units x 24 folds = 432 runs.
  Winner chosen by MEAN VALIDATION AUC ONLY; test never used for selection.
STAGE 2 (controls, fixed K=4 wd=1e-4, independent of Stage 1), 3 seeds each:
  T1 FC row only, D=90 (BNT-paper faithful)
  T4 FC row + ALFF, per-subject min-max (Track-3 normalisation ablation)
  T5 FC-row COLUMNS shuffled per subject — NEGATIVE CONTROL, must collapse
  T6 best config with LABELS PERMUTED — LEAKAGE FLOOR, must be 0.50
  = 12 units x 24 folds = 288 runs.
  T3 (116 ROIs, D=119) IS NOT RUN. Reason recorded before any result: the frozen
  ALFF exists only for 90 ROIs (M1 is computed from NIfTI voxels via atlas labels
  1..90; the 26 cerebellar/vermis regions were never computed), and the standing
  instruction for S13 is to use ONLY existing frozen data. Building 116-ROI ALFF
  would create new unfrozen derived data. Consequence, stated plainly: the
  116-vs-90 ROI contribution is NOT MEASURED by this audit.
STAGE 3 (after Stage 1): re-run T1/T4/T5 at the Stage-1 winning (K,wd) if it
  differs from (4,1e-4); ROI-permutation control on the overall winner (must
  collapse to ~0.50).
Total launched wave: 30 units x 24 folds = 720 fold-runs.

## EVALUATION — two independent points, both reported
HEAD  : metrics from the model's own logits on the held-out test fold.
PROBE : K.probe_pipe(repr.astype(float64), y, [(tr,te)], []) on Z_G IMMEDIATELY
        after the encoder. THIS IS THE NUMBER THAT MATTERS — never contaminated by
        the head, and the same instrument that produced 0.7565.
No projection head exists in S13; no post-projection representation is evaluated.
METRICS per fold AND pooled OOF AND per site (LOSO): AUC + 95% bootstrap CI (2000),
AUPRC, accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV, F1, MCC,
confusion counts, Brier, calibration slope and intercept, decision threshold and
how it was chosen. Threshold is score>0 at both eval points (sigmoid 0.5 for the
head; LinearSVC's natural boundary for the probe). DECLARED: LinearSVC decision
scores are uncalibrated, so probe Brier/calibration are computed on sigmoid(score)
and are INDICATIVE ONLY; head Brier/calibration are genuine.
Aggregation: mean +/- std ACROSS FOLDS and ACROSS SEEDS, reported SEPARATELY and
labelled; never collapsed into one number.

## RESOURCES
CPU ONLY, no --gres. Cluster state at launch: 5 GPUs free with 0-4 CPUs attached,
154 CPU cores and 2.4 TB RAM idle, 29+51 jobs pending on the GPU partitions. The
model is ~294K params, ~18 MFLOPs/subject; GPU buys nothing and costs queue time,
and CPU is bitwise deterministic, which this audit requires (same call as S12B
Track 1). SLURM array, --cpus-per-task=4, --mem=8G. One unit = (arm,K,wd,seed),
own directory under jobs/, atomic writes (tmp then os.replace), fold JSON BEFORE
the npz resume marker (S12A4b lesson), resume-safe by skipping folds whose npz
exists. No shared mutable cache. Logs queue wait, wall clock, peak RSS, hostname,
core count.

## PRE-REGISTERED DECISION RULES (written before looking)
PRIMARY BAR: PROBE AUC on S11 ordinary folds > 0.7565 AND LOSO PROBE > 0.7432.
  >= 0.78         strong result; BNT recovers and exceeds
  0.7565 - 0.78   BNT recovers what WGIN destroyed; this is the paper
  0.72 - 0.7565   ties the linear model; "linear ceiling" paper with the
                  strongest deep baseline properly tested
  < 0.72          architecture-limitation result stands, unarguably
  *** IF NO ARM REACHES 0.76: STOP CHANGING ARCHITECTURES. No third family.
      Write up. THIS RULE IS BINDING. ***
CONTROLS: T5 (column-shuffled) must collapse toward 0.50 — if not, the model is
reading global FC magnitude, not connectivity, and that is reported. T6 (permuted
labels) must be 0.50 +/- 0.03; anything above is leakage. ROI-permutation on the
winner must collapse; if not, STOP.
CONTRASTS: T2-T1 = measured ALFF contribution (if ~0, say so plainly; it would
match Track 3's +0.004 and is an honest negative). T4-T2 re-tests Track 3's
normalisation leak on a new architecture. T3-T2 is NOT MEASURED (see Stage 2).
CALIBRATION NOTE for the report: BNT's published 80.2% used Craddock-200 (200
ROIs) and 1009 subjects; we have AAL-90 and 954. A 90-dim connection profile
carries strictly less information. DO NOT target 80.2%.

## DELIVERABLES
GATE0_DATA.md GATE1_INSTRUMENT.md GATE2_MODEL.md PROTOCOL.md s13_results.csv
s13_epochs.csv RESULTS.md DECISION_REPORT.md manifest.json configs/ plots/
PLOTS: (1) probe AUC by arm with the 0.7565 line; (2) K x wd heatmap on VALIDATION
AUC only; (3) train/val curves for the winner; (4) per-site LOSO bar chart;
(5) head AUC vs probe AUC scatter over all folds.
STOP after the report. No contrastive learning, no A-GCL, no third architecture,
no further tuning without explicit authorisation.
