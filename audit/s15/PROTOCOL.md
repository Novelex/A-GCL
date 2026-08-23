# S15 — ARCHITECTURE + OPTIMISATION REPAIR: PRE-REGISTERED PROTOCOL
Branch s15-arch. Written and committed BEFORE any S15 result exists.
Amendments -> PROTOCOL_AMENDMENTS.md, timestamped and justified, never silent.

## SCOPE — FIXED, NOT EXPANDABLE
Architectures: WGIN-R and BNT-R. TWO. Atlas: AAL-90 ONLY, 954 subjects; no
116-ROI build, no new atlas. Modalities: FC and ALFF, both frozen and
hash-verified. FORBIDDEN: MLP baselines, any third architecture, new datasets,
new preprocessing, lambda sweeps, temperature sweeps. ComBat: ONE arm, Stage 3,
on the winner, fitted in-fold.

## AUTHORISATION — the diagnosis acted on (NOT re-derived)
S13 BNT probe 0.6583 vs FC-MLP probe 0.7246 on the same information. A 285K-param
transformer can trivially represent the MLP's solution; scoring 0.066 below it is
an OPTIMISATION failure, not a capacity limit. Corroborated four ways by S13's own
logs: parameter movement 0.016-0.039 (~3% from init); 92% of steps clipped
(median 628, clip_max_norm=1.0 miscalibrated); val AUC peaking at EPOCH 1 (no
warmup); ~500 total optimizer steps vs BNT's paper ~2200.
THE WATERMARK: S12A3 RANDOM WGIN 0.6539 vs S13 "trained" BNT 0.6583 — the same
number. Both are untrained-encoder scores.
DESIGN DEFECT: S13 used K=2,H=128 -> Z_G 256 dims, head squeezed to 32.
4005 -> 256 -> 32 is 125x compression: B5 rebuilt under a new name. Validation
"picked" K=2 inside noise (0.6890 vs 0.6757).
STRUCTURAL FACT EXPLOITED: E is a FROZEN ORTHONORMAL BUFFER, not a Parameter, so
raising K widens the representation at ZERO trainable-parameter cost. K=32,H=128
-> 4096-dim repr on ~264K trainable params: wider than the 4005 FC edges with a
quarter of EdgeMLP's parameters — the LinearSVC regime (wide features, few
params, strong L2).

## FROZEN REFERENCE LINES (printed in EVERY table, NEVER recomputed)
  LinearSVC 4005 FC edges  F-LAB 0.7565  LOSO 0.7432
  ridge logistic           F-LAB 0.7561  LOSO 0.7406
  BNT S13 winner           F-LAB 0.6583  LOSO 0.6619
  WGIN S12A5 arm A         F-LAB 0.6307
  RANDOM WGIN S12A3        F-LAB 0.6539   <- the untrained watermark

## SECTION 1 — ISOLATION
Branch s15-arch, creating ONLY audit/s15/**. READ-ONLY imports: s11_core (K),
s12a1_core (A1), s12a5_core (M), audit/s13 bnt_core. Modifies nothing under
audit/s0..audit/s14, agcl_ABIDE*.py, unsupervised/, datasets/. Verified with
`git diff main --stat -- ':!audit/s15'` printing nothing before submission.
audit/s15/jobs, feat, ckpt, cache, logs are gitignored — a branch switch can
never delete live job output (the failure mode that killed the S13 reruns).

## SECTION 2 — CACHE DISCIPLINE
ONE canonical build by s15_data.py -> cache/s15_data_<sha16>.npz containing FC
float32 [954,90,90], ALFF float32 [954,90,3], y int8 [954], subject_ids,
site_labels, fold_lab, fold_site, fold_loso assignment vectors, plus
CACHE_MANIFEST.json. If the target exists it is DELETED and rebuilt, never
appended. data.pt / data_dense_v3.pt / any earlier .pt or .npz are NEVER loaded.
MANIFEST HASHES (sha256, 16-hex): h_source_manifest, h_fc, h_alff, h_labels,
h_subject_order, h_folds_lab, h_folds_site, h_folds_loso, h_fc_edge_order,
h_alff_band_order, git_sha, builder_version, build_timestamp.
GATE-C ASSERTIONS, run by the builder AND at the START of EVERY job (<3 s):
len==954; ASD/NC == S11 manifest exactly; FC.shape (954,90,90); ALFF.shape
(954,90,3); y.shape (954,); both dtype float32; max|FC[i]-FC[i].T| < 1e-6 for all
i; diag(FC[i]) == 1.0 exactly; |FC| <= 1+1e-6; no NaN/Inf; h_subject_order,
h_labels, h_folds_lab, h_fc_edge_order, h_alff_band_order all == S11 values
(folds LOADED frozen, never regenerated). On mismatch: print expected vs actual,
write a FAILED record, exit non-zero. NO try/except around loading, NO fallback
cohort, NO silent default.
NO SHARED MUTABLE STATE: jobs open the cache READ-ONLY (mmap_mode='r'); no job
writes to cache/; every job writes only to jobs/<unit_id>/; every write is atomic
(<path>.tmp then os.replace); fold JSON is written BEFORE the npz resume marker.
PER-JOB PROVENANCE: every results row carries h_fc, h_labels, h_folds_lab and the
cache filename. At report time, identical hashes across all rows are ASSERTED; if
two differ the whole wave is invalid and is declared so.

## SECTION 3 — JOB SURVIVAL
J1 NO assert may terminate a job for a TRAINING-DYNAMICS observation. "loss did
not decrease", "clipping frequent", "val AUC peaked at epoch 1", "movement small"
are ALL RECORDED FLAG COLUMNS, never exceptions. (The S13/A3 bug killed 8 of 30
units including all six negative controls — which are designed not to learn — and
biased 545 folds by censoring every best_epoch==1 fold. Never again.)
J2 Asserts may halt a job ONLY for DATA/SHAPE violations (Gate-C).
J3 Every fold runs inside try/except; on exception write status='FAILED' with the
full traceback and CONTINUE. A unit NEVER aborts because one fold failed.
J4 Resume-safe: skip any fold whose result npz exists.
J5 SLURM --requeue, --time=48:00:00, --signal=B:USR1@300 with a checkpoint-and-
exit-0 handler; heartbeat touched every 60 s so a stall is distinguishable from a
crash. J6 jobs/<unit_id>/STATUS.json = {queued,running,done,failed} + fold
progress counter, readable with one cat.

## SECTION 4 — SMOKE SUITE (array index 0, one time, gates the wave)
S1 Gate-C in full. S2 INSTRUMENT (K.probe_pipe unchanged is THE probe): R1 raw FC
4005 -> CEILING_PROBE; R2 raw FC LinearSVC grid-C MUST be 0.7565+-0.015 and LOSO
0.7432+-0.015; R3 labels permuted MUST be 0.50+-0.03; R4 ALFF(3) floor recorded.
Any fail -> the ruler is broken, STOP. S3 CAPACITY SWEEP (forward-only, random
init, 3 seeds): BNT K in {2,4,8,16,32,64} x H in {64,128,256}; WGIN hidden in
{32,64,128} x readout in {sum, ROI-concat}. Per config report (a) FC-recon R^2
(ridge, alpha by inner CV on TRAIN only, variance-weighted, held-out subjects),
(b) repr dim, (c) TRAINABLE param count with E.requires_grad==False asserted and
E present as a buffer (never counted), (d) OCREAD mean assignment entropy.
S13's K=2/H=128 R^2 reported EXPLICITLY as the post-mortem. RECORDED, NOT GATING.
S4 OVERFIT-ONE-BATCH both architectures at K=32/H=128 and hidden=128: 32 subjects,
dropout 0, wd 0, no clip, 500 steps -> train AUC 1.000 and loss < 0.01. BOTH
BLOCKING. S5 BNT correctness (all blocking): max|EE^T-I| < 1e-5; E buffer,
requires_grad False, in state_dict; P.sum(-1)==1 (<1e-6); Z_G (B,K,H); attention
rows sum to 1 (<1e-6); 2 layers, 4 heads; ROI-PERMUTATION EQUIVARIANCE (permute
ROI order AND profile columns AND the corresponding inp.weight columns -> Z_L
permutes identically, <1e-4) — see AMENDMENT A1 for why the weight-column
permutation is required and what it does and does not prove; K=1 equals a MEAN
readout (<1e-5, declared factor 90); gradients non-zero in every trainable module
at step 1 and ZERO for E; same seed -> bitwise-identical logits; checkpoint
reload -> bitwise-identical repr and logits. S6 WGIN correctness (all blocking):
hand-verify (I+A.E)H on a 4-node graph with a NEGATIVE edge weight and a
sign-flipping node, <1e-6 float64 and <1e-4 float32; DOCUMENT the self-loop
double count (edge_index contains (i,i) with FC[i,i]=1.0 AND the layer adds
(1+eps)x_r, so own features count TWICE) as a LOGGED FORK, not silently fixed,
because S12A5 ran with it and comparability matters; with FC-row features assert
x[i,:90]==FC_dense[i,:] bitwise for 8 subjects x 8 ROIs. S7 ADAPTIVE-CLIP
CALIBRATION (recorded, not gating): 5 folds x 20 epochs with clipping OFF,
grad-norm p50/p90/p95/p99/max per architecture per module group — the direct S13
post-mortem. S8 Determinism: use_deterministic_algorithms(True), OMP_NUM_THREADS
recorded, two identical runs bitwise identical.
Emits SMOKE.md; exits non-zero on any BLOCKING failure so the dependency cancels
the wave cleanly.

## SECTION 5 — ARCHITECTURES (same contract: forward(batch, edge_vec) -> (repr, logits))
BNT-R: X [B,90,D]; inp Linear(D,H) with H>=D ASSERTED (never a compression);
L=2 PRE-NORM residual blocks [Z = Z + Drop(MHSA(LN(Z)),0.10); Z = Z +
Drop(FFN(LN(Z)),0.10)] with FFN Linear(H,2H)-GELU-Drop(0.10)-Linear(2H,H); MHSA
Q,K,V = Linear(H,H,bias=False), 4 heads, d_k=H/4, attn=softmax(QK^T/sqrt(d_k))
then Dropout(0.10). FC edge weights are NEVER injected into attention scores; FC
enters ONLY through node features. Z_L = LayerNorm(Z). OCREAD: C =
xavier_uniform_(K,H) seeded once at __init__, E = gram_schmidt(C) unit-L2 rows,
register_buffer('E',E); P = softmax(Z_L @ E.T / sqrt(H), dim=-1) SCALED; Z_G =
P^T @ Z_L -> [B,K,H]; repr = reshape(B,K*H); mean assignment entropy logged every
fold. HEAD: LayerNorm(K*H) -> Dropout(0.10) -> Linear(K*H, 1). A SINGLE LINEAR
LAYER, no 32-dim bottleneck (S13's Linear(K*H,32)->Linear(32,1) was a second,
undiagnosed bottleneck).
WGIN-R: WGINConv message passing UNCHANGED from the audit ((I+A.E)H, same eps,
same relu(x_j) in message()) so comparability with S12A5 holds. Repaired around
it: inp Linear(D,hidden) with hidden>=D ASSERTED (fixes B2); LayerNorm not
BatchNorm (B3); F.normalize DISABLED (B4); readout ROI-concat over all 90 nodes
-> 90*hidden dims (B5), sum-pool only as a logged control arm; head LayerNorm ->
Dropout(0.10) -> Linear(90*hidden, 1), single. B1 is addressed by the
node-feature arms W2/W3, not by the encoder.

## SECTION 6 — SIX ARMS (2 x 3 factorial, nothing else)
W1 WGIN, node ALFF(3), edges FC        — the A-GCL paper's exact configuration
W2 WGIN, node FC-row(90), edges FC     — is WGIN's ceiling the rank-3 INPUT or
                                         message passing itself?
W3 WGIN, node FC-row+ALFF(93), edges FC— does ALFF add anything on a graph?
B1 BNT, node FC-row(90)                — the BNT paper's exact configuration
B2 BNT, node FC-row+ALFF(93)           — both modalities on the best backbone
B3 BNT, node ALFF(3)+ROI one-hot(90)=93— does ALFF carry ANY independent signal
                                         given fair positional information?
DATA: FC diagonal STAYS 1.0 (it is the positional signal); no row z-scoring; no
/max|FC| (proven no-op); no sparsify/threshold/negative-zeroing. ALFF z-scored
ACROSS SUBJECTS per (ROI,band) with mu/sd fitted on THAT FOLD'S TRAIN INDICES
ONLY, sd clamped >= 1e-6; per-subject min-max is a Stage-3 ablation only.
LABELS ASD=1 NC=0, STATED IN EVERY TABLE: the original A-GCL code uses ASD=0/HC=1,
so SENSITIVITY AND SPECIFICITY ARE SWAPPED relative to the paper's columns.

## SECTION 7 — FOLD PROTOCOLS (all three, every arm, every time)
F-LAB  5-fold stratified by LABEL = existing S11, THE ANCHOR, never modified.
F-SITE 5-fold stratified by (LABEL, SITE), StratifiedKFold on the composite key,
       seed 20260818, frozen and hashed once by the builder; sites with < 10
       subjects pooled into SMALL_SITE for stratification only, pooled sites
       recorded. NEW: the folds have only ever stratified by label.
F-LOSO leave-one-site-out = existing K.folds_loso(y).
Reported side by side ALWAYS. F-SITE will likely read HIGHER than F-LOSO because
site information becomes exploitable within folds — EXPECTED, STATED, not hidden.
F-LAB remains the number compared to 0.7565. Inner validation:
train_test_split(test_size=0.20, stratify=y[tr], random_state=M.BASE), used ONLY
for early stopping and config selection; test scored EXACTLY ONCE per fold.
DECLARED COUNT: F-LOSO yields 19 evaluable sites (both classes present), so the
per-unit total is 5 + 5 + 19 = 29 folds, not the 27 estimated in the commission.

## SECTION 8 — REPAIRED TRAINING RECIPE (identical for both architectures)
batch_size 32 (was 64) -> ~24 steps/epoch; max_epochs 400 (was 200); min_epochs
80 (EARLY STOPPING CANNOT FIRE BEFORE THIS — kills the epoch-1 selection
pathology); patience 50, min_delta 1e-5 on VALIDATION AUC; warmup linear 0->lr
over the first 10% of total steps (THE SINGLE MOST IMPORTANT ADDITION; S13 had
none); cosine decay to 0.05*lr after warmup; AdamW betas (0.9,0.999) eps 1e-8;
lr SWEPT {3e-4, 1e-3} (NOT 1e-4 — S13 proved it cannot move this model);
weight_decay SWEPT {1e-3, 1e-2} (deliberately strong: a wide linear head on 763
samples needs the SVM's L2).
ADAPTIVE GRADIENT CLIPPING replaces the fixed threshold: a deque of the last 200
total grad norms; steps 1-50 DO NOT CLIP, only record; from step 51 clip_max_norm
= p90 of the deque, recomputed every step; threshold, raw norm and clip flag all
logged. clip_rate reported per fold; > 30% is flagged loudly.
label smoothing 0.05 (was 0.10); dropout 0.10 everywhere (S13 used 0.30 in four
places on 763 samples — crushing, not regularising; weight decay now carries the
load); EMA of weights decay 0.999, EMA and raw both evaluated, selection by
VALIDATION only, both reported with the delta.
LOSS SWEPT, 2 levels: L-BCE = BCEWithLogitsLoss on smoothed targets t=y*0.90+0.05
with AUC reported against RAW y; L-AUC = pairwise ranking surrogate,
mean(softplus(-(s_pos - s_neg))) over all (pos,neg) pairs in the batch — you
MEASURE AUC, this OPTIMISES it.
seeds 20260818/19/20 in the grid, 5 for the winner in Stage 3. Device CPU,
set_num_threads(4), OMP_NUM_THREADS=4, use_deterministic_algorithms(True).
PER-EPOCH LOG: train/val loss, train/val AUC, lr, grad norm per module group,
adaptive_clip_threshold, clip_events, clip_rate, epoch wall seconds.
PER-FOLD: best_epoch, best_val_auc, epochs_run, TOTAL OPTIMIZER STEPS, train-val
gap at the selected epoch, verdict OVERFIT (gap>0.15)/UNDERFIT (train_auc<0.65)/
HEALTHY, PARAMETER L2 MOVEMENT FROM INIT PER MODULE GROUP AS A FIRST-CLASS RESULT
(any fold with movement < 0.10 is flagged UNTRAINED; its AUC is reported but does
NOT count toward any architecture verdict), OCREAD entropy (BNT), EMA-vs-raw
delta, integrity flags with loss_decreased in {pass,fail,n/a} (n/a when
best_epoch==1); arms with arm_type='control' are EXEMPT from all learning checks.

## SECTION 9 — THE GRID (all submitted at once)
BRANCH 1 MAIN: BNT 3 arms x 2 loss x 2 lr x 2 wd x K{8,32} x 3 seeds = 144 units;
WGIN 3 arms x 2 loss x 2 lr x 2 wd x hidden{128,256} x 3 seeds = 144. H=128 fixed
for BNT with H>=D asserted. = 288 units x 29 folds.
BRANCH 2 CONTROLS (fixed mid config, arm_type='control', 3 seeds, all folds):
C-PERM labels permuted -> MUST be 0.50+-0.03; C-SHUF FC-row columns shuffled per
subject -> MUST collapse; C-ROI ROI order shuffled per subject -> MUST collapse;
C-RAND RANDOM-INIT ENCODER, HEAD ONLY TRAINED — THE CRITICAL CONTROL, run for
BOTH architectures at the main config: if a trained arm does not clearly beat its
own random twin, IT DID NOT TRAIN. = 4 x 2 arch x 3 seeds.
BRANCH 3 TRANSDUCTIVE (B2 and W2 only, 3 seeds, three modes kept strictly
separate, never merged or averaged): T1 UNLABELLED PRETRAINING (mask 15% of each
node's FC-row entries, reconstruct, on all 954, NO LABELS EVER, then fine-tune
inside each fold; linear probe reported BEFORE and AFTER — if AFTER < BEFORE,
pretraining destroyed information and that is reported); T2 FEATURE-ONLY COMBAT
(fitted on all 954 FEATURES, no labels, classifier fold-safe; report T2 minus the
in-fold ComBat result as the transductive gain); T3 T-LEAK — SUPERVISED TRAINING
ON ALL 954 THEN CV. THIS IS LABEL LEAKAGE, run as a MEASUREMENT ARM ONLY, always
reported beside its inductive twin with the delta stated, NEVER the headline,
NEVER in an abstract or conclusion, every row marked leakage=True.
BRANCH 4 STAGE-3 ABLATIONS (winner only, dependency on branch 1): A1 ALFF scaling
z-across-subjects vs per-subject min-max; A2 OCREAD softmax scaled vs unscaled
with entropy; A3 EMA vs raw; A4 WGIN readout ROI-concat vs sum-pool; A5 ComBat
fitted IN-FOLD (one arm only); A6 seed ensemble over 5 seeds; A7 score-average of
best WGIN and best BNT; A8 Platt/isotonic calibration on the INNER VALIDATION
SPLIT only.

## SECTION 11 — EVALUATION
HEAD from the model's own logits; PROBE = K.probe_pipe on `repr` IMMEDIATELY
after the encoder, never after any projection head (there is none here).
Metrics per fold, pooled OOF, and per site: AUC + 95% bootstrap CI (2000), AUPRC,
accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV, F1, MCC,
confusion counts, Brier, calibration slope and intercept, decision threshold and
how it was chosen. Mean +/- std ACROSS FOLDS and ACROSS SEEDS reported SEPARATELY
and labelled; never collapsed. Every table prints the frozen reference rows.
DECLARED: LinearSVC decision scores are uncalibrated, so PROBE Brier/calibration
are computed on sigmoid(score) and are INDICATIVE ONLY; HEAD values are genuine.

## SECTION 12 — PRE-REGISTERED DECISION RULES
PRIMARY: PROBE AUC on F-LAB vs 0.7565, F-LOSO vs 0.7432.
VALIDITY GATE, applied BEFORE any architecture claim — a result is INTERPRETABLE
only if, for that arm: parameter movement > 0.10 AND clip_rate < 30% AND the arm
beats its own C-RAND random-encoder twin by >= 0.03. An arm failing ANY of these
is reported as UNTRAINED, not as a verdict. S13 would have failed all three.
R1 Any arm gains >= +0.05 over its S13/S12 number under the repaired recipe ->
   the earlier architecture verdict was CONFOUNDED BY OPTIMISATION; say so
   explicitly and revise S13's conclusion IN WRITING.
R2 All six arms interpretable AND all flat (< +0.02) -> the architecture
   conclusion is CONFIRMED under proper training and is now far stronger than
   S12B or S13 alone. Close the chapter.
R3 W2 >> W1 -> WGIN's ceiling was the rank-3 INPUT (B1). W2 ~ W1 -> message
   passing itself is the limit.
R4 B2 > B1 -> ALFF adds on the transformer. B2 < B1 -> ALFF hurts, confirming
   S13's T1-T2 = +0.0175 at scale. Either reported plainly.
R5 B3 near chance -> ALFF carries NO independent signal and the A-GCL paper's
   node-feature choice is the root cause of its failure. Headline either way.
R6 F-SITE >> F-LAB -> earlier numbers were depressed by site imbalance across
   folds. F-SITE minus F-LOSO measures site exploitation.
R7 L-AUC > L-BCE by >= 0.02 -> the loss function was a real limitation.
R8 K=32 >> K=8 -> representation WIDTH was the S13 defect, corroborated directly
   by the capacity sweep's R^2.
R9 Transductive: T1, T2, T3 reported separately, never merged. T3 is leakage=True
   and is a measurement, never a result.
R10 Best final system = highest MEAN VALIDATION AUC across all branches, scored on
   test EXACTLY ONCE, with bootstrap CI and a DeLong test against LinearSVC.
R11 BINDING: if nothing beats 0.7565 on F-LAB after this wave, the project CLOSES
   with LinearSVC as the final answer and the architecture-limitation result as
   the contribution. No S16. No seventh idea. Not negotiable.
R12 Realistic expectation fixed NOW so the goalposts cannot move: 0.76-0.80 is a
   good outcome on AAL-90; multi-site ABIDE ASD/NC has a published ceiling around
   0.70-0.80. DO NOT target 0.85.

## SECTION 13 — DELIVERABLES
SMOKE.md, CACHE_MANIFEST.json, PROTOCOL.md, s15_results.csv (one row per branch/
arm/config/loss/lr/wd/K-or-hidden/seed/fold/fold_protocol/eval_point with every
metric plus n_params, repr_dim, fc_recon_r2, movement_*, clip_rate,
adaptive_clip_final, total_steps, ocread_entropy, ema_delta, verdict,
integrity_*, leakage, status, h_fc, h_labels, h_folds_lab, cache_file, node,
wall_s, peak_rss_mb, git_sha, ckpt_sha), s15_epochs.csv, RESULTS.md,
DECISION_REPORT.md, manifest.json, configs/, plots/ (8 mandated plots).
STOP after the report. No new architecture, atlas, dataset or tuning without
explicit authorisation.
