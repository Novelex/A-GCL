# S12A3 PREREGISTRATION — written BEFORE any result
Question: is the residual loss (S12A2 ceiling 0.6217 vs FC 0.7565) caused by WGIN-stack
operations (L2 node normalize / post-BN ReLU / width) or does a compression bottleneck remain?
Input (fixed): X_id=[M1_B|I90] (93 feats), FC edges, frozen S3C/S11 folds+order.
Encoders: RANDOM (seeded), real TUEncoder/GInfoMinMax code path, eval+no_grad,
num_gc_layers=2, message_relu=True, drop 0.3 (inert), seeds 20260818/19/20.
ARMS: A=baseline(norm=T,pbr=T,emb32)  B=norm F  C=pbr F  D=both F  E=emb64  F=emb128
Readout for scoring (pre-committed): flatten node output [954, 90*emb] -> S11 harness
(scaler+LinearSVC C-grid inner CV, pooled OOF AUC). Justification: S12A2 proved learned
ROI-aware readout = flatten ceiling within 0.018; flatten is the assumption-free retention
measure. NO global_add_pool anywhere.
Node output = 2nd return of production TUEncoder.forward (flags respected by real code).
DECISION (per protocol, on best arm's 3-seed mean ordinary AUC):
  >=0.68            -> encoder retention solved
  0.60-0.63         -> compression bottleneck remains (encoder-op changes do not recover)
  0.63-0.68 (gap)   -> pre-registered reading: partial retention gain, not solved
  <0.60             -> arm degrades retention
LOSO on all arms = corroboration only, never overrides ordinary-CV decision.
SENTINELS: plumbing FC probe must return 0.7565; arm A must reproduce S12A2 arm X
per-seed (0.6046/0.6224/0.6381) and match S12A1 final_postnorm embedding hashes.
Only after this report: consider S12A4 training. No training in S12A3.
