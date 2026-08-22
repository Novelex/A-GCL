# S12A2 DECISION REPORT
VERDICT (pre-registered, immutable): READOUT-BOTTLENECK **CONFIRMED**.
global_add_pool costs ~0.05-0.07 AUC relative to an ROI-aware readout on identical nodes,
in ordinary CV and LOSO, every seed. A learned 32-d ROI-aware readout recovers essentially
the entire node-accessible signal (F within 0.018 of the 2880-d ceiling X).
Caveats stated up front: (1) the primary margin is marginal (+0.0538 vs +0.05 threshold;
seed-1 gap +0.0163); LOSO (+0.0615, all seeds positive) is the stronger corroboration.
(2) C ~= F: PCA-32 matches the supervised readout — the gain is from PRESERVING ROI
identity, not from learning.
LIMIT: even the ceiling X = 0.6217 << frozen FC baseline 0.7565. Readout was A bottleneck,
not THE bottleneck: the (untrained, identity-input) encoder itself still discards ~0.13 AUC
before any readout. Chain now fully localized:
  raw FC 0.7565 -> encoder layer stack (identity input) 0.6217 -> sum pool 0.5498.
NEXT ACTION (single, not implemented, awaiting authorization):
S12A3 — ENCODER RETENTION: with identity input and ROI-aware readout both fixed (proven),
test whether trained (vs random) encoder weights close the 0.62 -> 0.76 gap, or whether
the WGIN layer stack itself (message_relu / L2 norm / BN) is the residual information sink.
No implementation performed. S0-S12A1 evidence untouched; production files untouched.
