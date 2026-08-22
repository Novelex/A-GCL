# S12A5 PROTOCOL — pre-registered BEFORE any training (git c0dd9f2)
QUESTION: where is FC 0.7565 -> WGIN-stack ~0.64 information lost?
H1 WGIN destroys edge identity | H2 insufficient regularization | H3 A-GCL failed on weak backbone.

FIXED: 954 cohort, S11 order, frozen S3C folds (5 ord + 19 LOSO), identity input
[M1_B|I90], WGIN emb32 norm=F mr=T pbr=T drop0.3, ROI readout Linear(2880->32) (= S12A4
stack), edge input = canonical X_fc upper triangle [954,4005] (sha 5e0780c9, frozen order).
Seeds 20260818/19/20. ES on validation AUC ONLY, patience 20, max 200 epochs, Adam 5e-4.
NEW vs S12A4 (recorded, not tuned): weight_decay=1e-4 all Wave-1 arms; edge-MLP dropout 0.3;
classifier dropout 0.0; per-epoch logging adds val loss + encoder/classifier grad norms.

WAVE 1 (supervised CE, 3 seeds x 24 folds each):
 A  WGIN control (S12A4 arm1 + weight decay). Q: does regularization beat 0.6383/0.6502?
 B  WGIN + FC residual skip: edge branch MLP(4005->256->ReLU->Drop0.3->32); concat(h_WGIN
    32, edge 32) -> head(64->1). Q: does restoring edge identity recover the loss?
 C  pure edge MLP: MLP(4005->256->ReLU->Drop0.3->32) -> head(32->1). Learnable edge
    baseline vs FC-SVM 0.7565.
WAVE 2 (parallel, independent; corrected-C A-GCL contract on the S12A4 ROI stack, NO
weight decay - audited contract unchanged):
 - INDUCTIVE = S12A4 arm 3 (identical stack/folds/seeds, pre-registered there): cited,
   not re-run — burn no GPU on a bitwise-archived duplicate. h 0.5672 / z 0.5055.
 - TRANSDUCTIVE (paper-style): SSL on ALL 954 unlabeled, fixed 200 epochs (S9 precedent,
   no ES since no labels may be used), 3 seeds; downstream S11 SVM probes stay fold-safe
   (fit inside frozen folds) on h/z/flatten at epoch 200; epoch-0 also probed. Keep rate,
   InfoNCE, memory loss, view loss, grad norms logged every epoch. Results NEVER mixed
   with inductive.
METRICS: pooled OOF AUC, acc, bacc, sens, spec, F1; LOSO + per-site AUC; per-epoch train
loss/AUC + val loss/AUC; train-val gap; grad norms; init-vs-best parameter movement
(TRAINABLE params only — S12A4b BN-buffer lesson); epoch-0 + final embeddings + ckpts saved.
WINNER VALIDATION: per-subject ROI-order permutation (RNG 20260818; X rows and FC
rows+cols permuted consistently), winner arm retrained seed0 on ordinary folds. Expect
collapse (<=0.55) proving anatomical identity usage.
PRE-REGISTERED INTERPRETATION (on 3-seed mean pooled OOF, ordinary):
 B >> A (>= +0.05)            -> WGIN loses edge identity; next: edge-aware A-GCL.
 C >> A and C >> B (>= +0.05) -> graph propagation unnecessary.
 A~B~C ~0.64 (within 0.02)    -> representation ceiling confirmed.
 Transductive >> inductive (>= +0.05 on h) -> prior failure was backbone/regime mismatch.
 Transductive still ~<=0.60   -> objective does not help even with proper representation.
FORBIDDEN: lambda/temperature tuning, sweeps, fold changes, test-epoch selection, old caches.
Every result carries git SHA, config JSON, seed, hostname, runtime, checkpoint hash.
STOP after decision report.

## PRE-WAVE REVIEW DISCLOSURES (recorded before any result; 3-lens review: 0 blockers)
- Wave-2 z0 probe + grad-norm logging added (registered metrics were missing from code).
- grad-norm grouping: roi.readout counted on the CLASSIFIER side (encoder = WGIN convs/bns
  only); disclosed here so curves are read correctly.
- Wave-1 proj_head receives no gradients (z discarded in CE path) — z must NEVER be probed
  from Wave-1 checkpoints; Wave-2 z is the only trained z.
- Winner ROI-permutation driver will permute Xid/FC/Xe GLOBALLY at load (not via the
  train_fold5 perm_roi arg) so training AND extraction see identical permuted arrays.
