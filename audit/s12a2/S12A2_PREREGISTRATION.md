# S12A2 PREREGISTRATION — written BEFORE any result
Input (fixed): S12A1 identity final_postnorm nodes [954,90,32], seeds 20260818/19/20.
Frozen S3C folds; S11 harness (scaler + LinearSVC C-grid inside inner StratifiedKFold(5,
seed BASE), scoring roc_auc); all transforms inside training folds.
ARMS (per seed):
  P  current readout: global_add_pool(nodes) -> [32]  -> S11 harness
  F  ROI-aware LEARNED readout: flatten [2880] -> Linear(2880,32) trained per OUTER fold
     (with Linear(32,1) CE head, Adam 1e-3, batch 64, max 200 epochs, EARLY STOPPING ON
     VALIDATION AUC (patience 20; inner 80/20 stratified split of outer-train, seed BASE);
     validation ACCURACY explicitly NOT used) -> frozen 32-d features -> S11 harness.
     Head's direct test AUC also recorded (diagnostic).
  C  diagnostic: flatten -> PCA(32) inside pipeline -> LinearSVM
  X  ceiling: flatten [2880] -> LinearSVM (== S12A1 final_postnorm probe)
LOSO: arms P and F only (primary comparison), all 3 seeds.
PRE-REGISTERED DECISION (immutable):
  READOUT-BOTTLENECK CONFIRMED if: mean3(F) - mean3(P) >= +0.05 AND F>P in every seed.
  RECOVERY QUANTIFIED by F vs X: F >= X - 0.02 -> a learned 32-d ROI-aware readout
  recovers (nearly) all node-level accessible signal; F << X -> 32-d learned readout
  insufficient, S12A3 must consider wider readout.
  C vs F separates supervised vs unsupervised 32-d compression of the same nodes.
No early stopping on accuracy anywhere. No leakage: readout trained per outer fold on
outer-train only; early-stop split inside outer-train; PCA/scaler inside pipelines.
