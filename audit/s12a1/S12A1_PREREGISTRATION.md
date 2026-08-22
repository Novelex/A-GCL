# S12A1 PREREGISTRATION — written BEFORE any probe result exists
date 2026-08-22 | BASE=20260818 | seeds BASE+0/+1/+2 | one intervention: X_id=[M1_B | I90] (90x93)
Corrected-C MAIN encoder only, instantiated values RECORDED (not assumed):
  emb_dim=32, num_gc_layers=2, normalize_nodes=True, message_relu=True, post_bn_relu=True,
  drop_ratio=0.3 (inert under eval), eps=0.0 buffer (train_eps never set), pooling standard,
  GInfoMinMax(enc, 32) positional (real call-site contract). No ViewLearner. No training,
  no gradients (torch.no_grad + eval). Encoder state_dict hashes recorded per seed; donor
  reuses the EXACT same 93-input state as its real-FC condition.
PLUMBING GATES (blocking):
  P1 identity algebra: max|Q1[:,3:] - (I90 + FC^T)| <= 1e-5 (float32), via the ACTUAL
     conv.propagate (S6 method); mismatching subjects = 0/954; identity-block diag == 2.
  P2 recovered 4005 FC (S11 pair order, transpose retained) == canonical S11 X_fc within
     float32 representation tolerance.
  P3 Q1-FC plumbing SVM reproduces the S11 raw-FC result: |AUC - 0.7565| <= 0.005
     (float32-conversion tolerance, declared here). PLUMBING ONLY, not the decision.
PRIMARY REPRESENTATION: H1_BN = BatchNorm(Linear(93,32)->ReLU->Linear(32,32) applied to Q1),
  captured BEFORE post-BN ReLU/dropout, flattened [90,32]->[2880], canonical ROI order.
  NOTE (recorded now): under eval() with untrained BN (running stats 0/1, affine 1/0),
  H1_BN ~= H1_preBN / sqrt(1+1e-5); their probes are expected to be near-identical.
PRIMARY PASS RULE (all required; thresholds immutable):
  A. mean 3-seed AUC(identity H1_BN) >= 0.65
  B. mean(identity - old H1_BN) >= +0.02 AND positive in ALL 3 seeds
  C. DONOR CONTROL (primary causal evidence): real-FC identity H1_BN minus
     training-mean-FC donor H1_BN >= +0.03 IN EVERY SEED, with the fold-safe donor
     protocol of section J (inner-train mean for C selection; outer-train mean for the
     final fit; never a global mean; donor never sees held-out FC).
EVALUATION: exact S11 probe (scaler+LinearSVC, C grid inside inner StratifiedKFold(5,
  seed BASE), scoring roc_auc), frozen S3C ordinary folds; LOSO only for H1_BN x
  {old, identity-real, identity-donor} after ordinary CV completes.
OUTCOME MAP: per instruction section N (no loop; single next stage per outcome).
