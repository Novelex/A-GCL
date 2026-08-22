# S12A3 DECISION REPORT
Best arm: D (normalize_nodes=F, post_bn_relu=F, emb32), 3-seed mean 0.6549.
PRE-REGISTERED BAND: 0.63 < 0.6549 < 0.68 -> **PARTIAL RETENTION GAIN — NOT SOLVED**.
Neither protocol endpoint fired: not >=0.68 (solved), not 0.60-0.63 (pure compression
bottleneck). Reading per preregistration: encoder operations ARE a real, causal part of the
residual loss — the L2 node-normalization alone discards ~0.03 AUC — but removing every
tested op still leaves ~0.10 vs the frozen FC baseline (0.7565). The remaining gap sits in
the random WGIN propagation itself (mixing/BN-affine at init), not in norm/ReLU/width knobs.
Chain now: FC 0.7565 -> best random encoder retention 0.6549 -> baseline encoder 0.6217
-> sum pool 0.5498.
Constraints honored: no training; no tuning of lambda/temperature/mask/InfoNCE; production
code untouched; only pre-registered arms A-F run (candidate combo norm=F+e128 NOT run —
noted for S12A4, not executed).
NEXT (single, not implemented, awaiting authorization): S12A4 — TRAINING with the proven
stack (identity input + ROI-aware readout + norm=F), to test whether learned weights close
the remaining ~0.10 retention gap that random propagation cannot.
S12A3_COMPLETE
