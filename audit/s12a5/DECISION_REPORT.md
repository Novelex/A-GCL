# S12A5 DECISION REPORT — seven answers
1. WAS DATA CORRECT? Yes. All hashes verified (dataset/FC/folds/order/features bitwise at
   source); 954 exact; no 956 path; gate report S12A5_DATA_GATE.md.
2. WAS TRAINING CORRECT? Yes. Grad norms finite + logged; ES on val AUC only; leakage
   lens ALL-CLEAR pre-launch; ROI-permutation control collapses to 0.502 (no artifact).
3. DID REGULARIZATION FIX OVERFIT? No. weight_decay 1e-4 left gaps ~+0.20-0.23 and arm A
   at 0.63-0.65 (== S12A4 unregularized). H2 REJECTED: regularization was not the issue.
4. DID EDGE SKIP RECOVER LOST INFORMATION? Yes, partially. B - A = +0.048 head / +0.056
   repr (pre-registered B>>A threshold met on repr; head at threshold). H1 CONFIRMED:
   WGIN propagation loses edge identity that a parallel FC branch restores.
5. DID FC MLP BEAT WGIN? Decisively. C - A = +0.082 head / +0.075 repr; C - B = +0.034/
   +0.020; C 0.7124/0.7239 approaches FC-SVM 0.7565 and beats every graph model in the
   entire audit. Pre-registered verdict branch: GRAPH PROPAGATION UNNECESSARY for this
   task/data — the diagnosis signal lives in ROI-indexed edge VALUES, and message passing
   only degrades access to it (movement analysis: even in B, the edge branch does the
   learning, enc 0.12 vs edge 0.90).
6. DID TRANSDUCTIVE A-GCL HELP? Not meaningfully. +0.035 vs inductive (< +0.05 threshold),
   still ~0.14 below arm C; SSL again REDUCED retention (flat 0.6408 < 0.6539 epoch-0) and
   the mask collapsed (keep 0.02-0.03). Pre-registered branch: THE OBJECTIVE DOES NOT HELP
   EVEN WITH THE PROPER REPRESENTATION REGIME. H3 REJECTED (failure is not backbone
   mismatch; it is the objective + propagation).
7. NEXT SCIENTIFICALLY JUSTIFIED STEP: close the remaining C-vs-SVM gap honestly
   (0.7239 vs 0.7565: the linear SVM on raw edges still wins — deep edge models buy
   nothing yet) BEFORE any further graph work. Single candidate experiment (NOT run,
   awaiting authorization): S12A6 — regularization-strength study on the pure edge model
   ONLY (one knob, pre-registered grid) OR accept FC-SVM as the final production model
   and close the audit. No edge-aware A-GCL is justified while a linear model beats every
   learned representation.
STOP: no additional tuning performed. Winner control PASS. All results carry git SHA
(c0dd9f2), configs, seeds, hostnames, runtimes, checkpoint hashes.
