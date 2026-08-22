# S12A4 PROTOCOL — pre-registered BEFORE any training
git 7886277 (clean, main) | python 3.12.13 | torch 2.5.0+cu121 | CUDA 12.1
Goal: can the repaired pipeline LEARN diagnosis-relevant representations?

FIXED STACK (all arms): M1_B frozen cache; S11 order; frozen S3C folds; identity input
X_id=[M1_B|I90] (93 feats); WGIN 2 layers emb32 (message_relu=T, post_bn_relu=T, drop 0.3);
normalize_nodes=FALSE (pre-registered from S12A3: L2 norm proven to discard ~0.03 AUC;
S12A3's registered next action names this stack); ROI-aware readout h=Linear(2880->32) on
flattened nodes (S12A2-proven; replaces global_add_pool); z=proj_head(h) (32->32).
Only the OBJECTIVE differs between arms. No other changes; production files read-only.

ARMS (3 seeds each: 20260818/19/20; Adam 5e-4, batch 32 drop_last=True, max 200 epochs):
 1 CE only:      BCEWithLogits on head Linear(32->1). ES: val AUC (head), patience 20.
 2 CE+InfoNCE:   arm1 + 1.0*InfoNCE(z, z_aug), temp 0.2 sym=True (audited C temps);
                 aug = edge_weight * Bernoulli(keep=0.8) per directed edge (seeded).
 3 Full corrected A-GCL (S8 "C" contract verbatim): ViewLearner, symmetric Concrete mask
                 (scale, NOT replace), budget reg +2.0, memory-bank cr +0.4 (late/validity
                 bank, 256x32), temps 0.2/0.1, alternating view-ascent/model-descent.
                 Unsupervised. ES: val AUC via LinearSVC(C=1) on h every 5 epochs, patience 4.
 4 Arm3 + HARD TOP-K mask, keep=0.80 per graph (straight-through; soft sigmoid backward);
                 budget reg removed (keep structurally enforced). actual_keep_rate logged
                 every epoch; run INVALID unless within 0.75-0.85.

FOLD PROTOCOL per (arm,seed) job: loop 7 ordinary folds + all LOSO sites. Inner 80/20
stratified split (seed 20260818) of outer-train for early stopping. Supervised arms train
on inner-train; unsupervised arms train on outer-train (no labels used) with ES-SVM fit on
inner-train h / scored on inner-val. Test fold NEVER touches training. Early stopping ONLY
on validation AUC (accuracy never used). Per-fold best checkpoint saved.

METRICS (pre-committed): cross-arm comparison metric = S11-harness pooled OOF AUC on h
(per-fold models; scaler+LinearSVC C-grid inside folds). CE decision-band metric (arms 1/2)
= pooled OOF head-logit AUC. Secondary: flatten-2880 probe, z probe, acc/bacc/sens/spec,
train-vs-val AUC gap (overfit indicator). LOSO reported for all arms.

DECISION BANDS (verbatim from protocol): CE>=0.70 architecture can approach FC baseline;
CE 0.60-0.68 encoder compression limits; CE<0.60 GNN design unsuitable.
A-GCL > CE+InfoNCE > CE => A-GCL works after repair. CE > A-GCL => objective hurts.
Hard-mask A-GCL > normal A-GCL => mask collapse was limiting factor.
(Gap bands 0.68-0.70 pre-registered as: borderline, read toward the nearer band.)

WINNER VALIDATION: label-permutation control (RNG(20260818) global permutation of y),
full pipeline rerun for the best arm, 1 seed. PASS if pooled OOF AUC <= 0.55.
Known limits disclosed: CUDA scatter_add nondeterminism ~2e-3 (S7 precedent) — seeds+
configs+hashes recorded for reproducibility; CPU smoke is the determinism reference.
STOP after decision report. No tuning, no extra sweeps, no architecture changes.

## PRE-WAVE CORRECTIONS (recorded before any training result exists)
1. Fold counts: the frozen S3C ordinary split is StratifiedKFold(5) => 5 ordinary folds
   (191/191/191/191/190) + 19 LOSO sites, not "7 ordinary + ~20 LOSO" as typed above.
   Caught by adversarial review; folds themselves are the frozen ones, unchanged.
2. Review blocker fixed: per-fold JSON now written atomically BEFORE the npz resume
   marker (crash re-runs fold instead of losing its record).
3. Disclosed arm4 properties (review-verified, accepted): hard top-k keep is 0.80 exactly
   by construction so the 0.75-0.85 gate is structural, not informative — mask-learning
   health is read from logged soft keep_mu instead; tie-splitting at the k-th boundary can
   break symmetry for ~1 undirected pair per graph (<=0.025% of edges); straight-through
   forward values binary to float32 rounding (6e-8).
Leakage lens: ALL-CLEAR (full trace in review record; test folds provably never touch
training, ES, scalers, BN stats, memory bank, or checkpoint selection).
