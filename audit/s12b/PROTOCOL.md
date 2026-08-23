# S12B — ENCODER AUTOPSY: PRE-REGISTERED PROTOCOL
Written 2026-08-23 BEFORE any S12B number was computed. Frozen on pre-registration
commit; results may not alter this file (amendments go to PROTOCOL_AMENDMENTS.md
with timestamps and justification, never silently).

MISSION: localize, to a single named operation, where the ASD/NC FC signal dies
inside the A-GCL encoder. Candidate serial bottlenecks:
  B1 agg = X + FC@X (rank-3 FC projection when X is 90x3)
  B2 Linear(d_in->emb) per-node compression      (never tested)
  B3 BatchNorm1d batch-dependent rescaling       (never tested vs LN/none)
  B4 F.normalize(x, dim=1) node-magnitude delete (never tested in isolation)
  B5 global_add_pool ROI-identity delete         (known: S12A2)

ANCHORS (frozen, not re-derived): FC-SVM 0.7565/0.7432 LOSO; WGIN+ROI-readout
0.63-0.65; +edge-skip 0.6789; edge MLP 0.7124; random enc 0.6539; trained enc
0.6429; InfoNCE 3.14->5.04 (> ln(31)=3.434); keep 0.44->0.033.

HARD RULES (from the commissioning order): no training in Track 1 (no optimizer/
epochs/loss/ES/A-GCL/InfoNCE/ViewLearner/masks); no tuning of lambda/T/mask/LR
anywhere; no modification of S0-S12A5 outputs or production files; frozen dataset/
folds/ROI order; NEVER evaluate projection-head z; all outputs under s12b only;
pre-registered criteria before results.

## GATE 0 — DATA/CACHE INTEGRITY (BLOCKING)
New cache namespace: s12b/cache/data_s12b_<sha256(S11 subject_manifest.csv)[:16]>.pt
built fresh from canonical sources (frozen S11 manifest -> .mat FC + X_sources M1).
If the file exists it is DELETED and rebuilt. No try/except around loading, no
fallback. HARD ASSERTS: n=954; ASD/NC counts == S11 manifest; 90 nodes and 8100
directed edges per graph; subject-ID order sha == S11 manifest; label sha == S11;
fold sha == frozen S3C splits.json (28fed44d...8916); FC edge-order sha == S11
X_fc sha; ALFF band-order sha == frozen M1_B. Records git SHA, hostname, python/
torch/CUDA, GPU, pip freeze, timestamp. Any failure -> STOP the entire run.

## GATE 1 — INSTRUMENT CALIBRATION (BLOCKING)
THE PROBE (single instrument for the whole audit): per fold —
  StandardScaler(fit TRAIN) -> PCA(n=min(200, d, n_train-1), svd randomized,
  random_state=20260818, fit TRAIN) -> LogisticRegression(l2, lbfgs, max_iter
  5000), C in {1e-3,1e-2,1e-1,1,10,100} by GridSearchCV(StratifiedKFold(5,
  shuffle, rs=20260818), scoring=roc_auc) on TRAIN only.
Decision threshold: p=0.5 on OOF probability (fixed, pre-registered).
Bootstrap: 2000 resamples, rs=20260818, on pooled OOF.
References: R1 = probe on raw FC 4005 (CEILING_PROBE). R2 = S11 LinearSVC
harness on FC (frozen code path). R3 = probe on FC with labels permuted by
rng(20260818). R4 = probe on flattened joint-minmax ALFF (270).
PASS: R2 in [0.74,0.77]; R1 >= R2-0.03; R3 in [0.47,0.53]. Else STOP.
RETENTION RATIO := (AUC-0.5)/(R1-0.5), used for every Track-1 number.

## GATE 2 — FORWARD CORRECTNESS (BLOCKING)
(1) WGINConv hand-verification on a 4-node toy graph with a negative edge weight
and a sign-flipping node: <1e-6 float64, <1e-4 float32, message_relu both ways.
(2) Self-loop fork documented: edge set includes (i,i) with FC[i,i]=1 AND WGINConv
adds (1+eps)*x_r -> own features counted twice. ALL arms KEEP the diagonal
(production behaviour); logged as a fork, not fixed.
(3) Arm C feature check: x[i,3:93]==FC[i,roi,:] bitwise, 8 random subjects/ROIs.
(4) Finiteness of every stage tensor, 8 subjects. (5) Same seed -> bitwise-equal
stage tensors, CPU and GPU (cudnn.deterministic=True). (6) Save/reload bitwise.
(7) PARITY: S12BEncoder(bn, mrelu=T, nn=T, emb32) with weights copied from a
production TUEncoder must reproduce its (xpool, node) outputs to <=1e-5 (fp32).
(8) A1 assert: param-free X+FC^T@X equals conv.propagate+(1+eps)x with mrelu=F.

## TRACK 1 — FORWARD-ONLY AUTOPSY
ARMS (node features; edge weights are ALWAYS the true FC — controls act on the
feature pathway only):
  A ALFF(3, joint-minmax frozen M1_B) | B ALFF+I90 (93) | C ALFF+FC-row (93)
  D ALFF+I90+FC-row (183)
  C-shuf: FC-row block with ONE column permutation per subject,
          perm_s = rng(20260818+row_index).permutation(90), same for all nodes.
  B-rand: I90 replaced by FIXED codes G = rng(20260818).standard_normal((90,90))
          /sqrt(90), identical for every subject.
GRID: emb_dim {32,64,128} x norm {bn,ln,none} x message_relu {T,F} x seeds
{20260818..20260822}; normalize_nodes {T,F} is DERIVED from the same forward
(S4=F.normalize(H2)); post_bn_relu FIXED at production True; dropout inert
(eval). 6 arms x 3 x 3 x 2 x 5 = 1080 forward families; the commissioned 720 =
the 4 main arms; controls add 360.
ENCODER: audit replica S12BEncoder importing production WGINConv verbatim;
2 layers; layer0 MLP Linear(d,emb)-ReLU-Linear(emb,emb); norm in {BatchNorm1d,
LayerNorm, Identity}; forward order conv->norm->(ReLU if not last)->(last: none).
BN PROTOCOL (explicit choice): BatchNorm1d(momentum=None) => cumulative running
stats. Per fold: fresh init -> model.train() -> ONE pass over that fold's TRAIN
subjects only, manifest order, batch 128 -> model.eval() -> extract. Test
subjects never touch BN stats. LN/none are fold-independent. NOTE: this differs
from S12A1-S12A5 (init stats); the delta is itself reported.
STAGES: S0 input(flat 90xd); S1 A1=X+FC^T@X param-free, message_relu=F
(paper-literal; mrelu enters at S2), diag kept; S2 H1; S3 H2; S4 normalize(H2)
(normalize=T only); S5 global_add_pool(S4 or H2); S6 flatten 90xemb (ROI-aware).
S0/S1 depend on arm only (no params): computed once per arm.
INSTRUMENTS at every stage: I1 = Gate-1 probe, frozen 8 ordinary folds + LOSO,
full metric block. I2 = FC reconstruction: RidgeCV(alphas logspace(-3,3,7),
fit TRAIN) multi-target from the SAME per-fold scaler+PCA projections onto the
4005 FC vector; OOF variance-weighted R2, per-edge R2 distribution, frac
R2>0.5. I3 = confounds from the same projections: site 17-class multinomial LR
(C=1 fixed) -> macro-F1+acc; age, sex, func_mean_fd (motion), mean|FC|, total
FC strength -> RidgeCV OOF R2. I2/I3 on ordinary folds (pre-registered scope);
I1 also on LOSO. Instrument note: I2/I3 measured through the same PCA-200
bottleneck as I1 — consistent across stages by construction.
DEDUP (declared, not silent): S0/S1 shared across the grid; S2/S3 shared across
normalize_nodes; S4-S6 computed per branch. information_audit.csv contains the
FULL expanded factorial rows; dedup is computational only.
STORAGE: per eval — metrics row, OOF probabilities, OOF PCA projections (f32).
Full stage tensors ONLY for top-5 configs by S6 retention + arm A emb32/bn/
mrelu=T/nn=T baseline. Disk usage logged.
CONTROLS: C-shuf and B-rand run over the full grid. P-lab (labels permuted by
rng(20260818)) and P-roi (per-subject ROI permutation applied consistently to
features AND FC, S12A5 method) run on the BEST config = highest S6 I1 pooled
ordinary OOF AUC (tie-break: higher S6 I2 R2). PASS for both = AUC in
[0.45, 0.55].

## TRACK 2 — FC EDGE-MODEL CEILING (trains; separate question)
MLP: Linear(4005,h)-ReLU-Dropout(0.3)-Linear(h,32) + head Linear(32,1);
h in {64,256,512}; AdamW(lr 5e-4, weight_decay in {1e-5,1e-4,1e-3,1e-2})
(decoupled decay, NOT Adam+L2); seeds {20260818..20}; batch 32; inner val =
20% stratified split of outer-train (rs 20260818); early stop on val AUC,
patience 20, max 200 epochs; frozen 8 ordinary + LOSO folds.
Classical on identical folds: LinearSVC (S11 harness, = R2), ridge-logistic
(l2 LR, probe C-grid, scaled raw 4005), elastic-net logistic (saga,
l1_ratio {0.2,0.5,0.8} x C {1e-3..10}, max_iter 5000).
Per-epoch diagnostics: train/val loss, train/val AUC, lr, grad norm per group,
param L2 movement from init per group (trainable only). VERDICT RULE
(pre-registered): OVERFIT if (train_auc - val_auc) at best epoch > 0.10;
UNDERFIT if train_auc at best epoch < 0.70; else HEALTHY.
TRAINING_INTEGRITY asserts: finite loss every epoch; loss(best) < loss(ep1);
grads finite; selection never touches test; checkpoint reload reproduces
selected val AUC bitwise.

## TRACK 3 — ALFF NORMALISATION (CPU)
Documented fact (locked pre-run): production abideDataset.py v3 normalizes
per subject PER BAND (x.min(dim=0), lines 71-74); the frozen audit M1_B uses
per-subject JOINT min-max over 90x3 (bitwise-verified vs the S5 cache in
S12A1). Both are per-subject rescalings that delete between-subject amplitude.
ARMS (Gate-1 probe, ordinary + LOSO): raw ALFF 270; joint min-max 270 (=M1_B);
per-band min-max 270 (v3 formula); z-across-subjects 270 (declared: identical
to raw under the probe's per-feature scaler — run as an instrument-invariance
check, expected equal to raw within noise); ALFF(raw)+FC 4275; FC alone (=R1).
VERDICT: if raw/across-subject beats per-subject variants beyond noise, the
per-subject normalization is an independent information leak.

## TRACK 4 — PROTOCOL INFLATION (existing backbone, unchanged)
Backbone: production corrected-C stack (s8_core CFG 'C': TUEncoder(3,emb32,
nn=T,mr=T,pbr=T), GInfoMinMax positional, ViewLearner, verbatim train_step),
ALFF(3) inputs, global_add_pool. Representation h ONLY (z banned).
T-ind: SSL on each ordinary fold's TRAIN subjects only, 100 epochs, extract
+probe h every 5 epochs (20 evals). T-trans: SSL on all 954, same schedule,
classifier still fold-safe. 3 seeds {20260818..20}.
E-best = best-of-20 evals selected ON TEST (the inflationary practice, measured
deliberately); E-final = epoch-100 eval.
REPORT: delta_trans = T-trans - T-ind and delta_best = E-best - E-final, each
at matched settings, mean +/- sd over seeds. Deltas reported plainly; no claims
about the paper's number.

## METRICS (every I1 evaluation)
Per fold, pooled OOF, and LOSO: AUC + 95% bootstrap CI (2000), AUPRC, acc,
balanced acc, sens, spec, PPV, NPV, F1, MCC, confusion counts, Brier,
calibration slope+intercept (logistic fit of y on logit(p), p clipped 1e-6),
threshold 0.5. Across-fold and across-seed dispersion reported SEPARATELY.
Per run: config JSON, seed, param count, dims per stage, wall clock, peak GPU
mem, node, git SHA, artifact SHA-256.

## PRE-REGISTERED DECISION RULES (verbatim from commissioning order)
1 Dead at S1 for ALL arms at random init -> B1 confirmed; message passing is
  the wall; do not train. 2 Arm A dies at S1, C/D survive -> B1 confirmed and
  fixable by node representation. 3 Survives S1, dies at S2, recovers with
  emb 32->128 -> B2 confirmed (new finding). 4 Recovers with LN/none -> B3
  confirmed. 5 Recovers with normalize_nodes=F -> B4 confirmed. 6 Dies only at
  S5, survives S6 -> B5 (known). 7 Any config retention >=0.95 through S6
  (~AUC>=0.74) -> encoder CAN carry signal; training/objective is the wall.
8 No config anywhere >0.85 (~0.70) -> this family cannot carry the signal;
  stop modifying WGIN. 9 C wins and C-shuf wins -> global strength, not
  connectivity. 10 B-rand matches B -> width, not anatomy.
"Dies/survives" quantitative: dies at stage k = retention drops by >=0.25
between k-1 and k AND stays <0.70 at k; survives = retention >=0.85 at k.
(Band 0.70-0.85: partial loss, named and reported as such.)

## RESOURCES
Track 1: 18 SLURM array tasks (6 arms x 3 emb), gpu-l40s, 1 GPU + 6 CPU each.
Track 2: 36 tasks (h x wd x seed) + 1 classical task, CPU-oriented.
Track 3: 1 CPU task. Track 4: 3 GPU tasks (one per seed), queued after T1.
Every job re-verifies Gate-0 hashes at startup and aborts on mismatch. No
shared mutable cache. Atomic TEMP->validate->rename; SKIP-keyed resume.
Queue wait and wall clock logged.

## OUTPUTS
audit/s12b/: GATE0_DATA.md GATE1_INSTRUMENT.md GATE2_SMOKE.md PROTOCOL.md
information_audit.csv training_diagnostics.csv RESULTS.md DECISION_REPORT.md
manifest.json configs/ plots/ (6 mandated plots). Final report answers the 12
commissioned questions in order. STOP after the report.
