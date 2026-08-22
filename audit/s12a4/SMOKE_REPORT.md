# S12A4 SMOKE REPORT — PASS (8 subjects, 2 epochs, seed 20260818, CPU)
All protocol checks:
1 forward finite (no NaN/Inf) all 4 arms. 2 edge_weight not None + finite asserted.
3 ROI identity features verified present (X_id[:,:,3:] == I90 bitwise).
4 readout shapes: nodes [B*90,32] -> flatten [B,2880] -> h [B,32], z [B,32].
5 losses finite: CE 0.4539; CE+InfoNCE (nce -0.3991, legal: -log ratio>1 at T=0.2);
  arm3 view 2.6597 / model 3.5954 / cr 3.3345; arm4 view 1.501 / model -0.383.
6 checkpoint save->reload: outputs allclose atol 1e-6 (eval mode), all arms.
7 production tree unchanged (git clean outside audit/).
Mini end-to-end train_fold (BATCH=4 smoke-only override): ES machinery, val AUC logging OK.
SCIENTIFIC NOTE (pre-wave, not a defect): arm3 soft Concrete mask initializes at ~0 keep
(untrained ViewLearner emits extreme negative logits on the norm=F identity stack). Arm4's
hard 80% top-k control is exactly the intervention that isolates this; keep-rate is logged
every epoch in both arms. No tuning performed (protocol forbids it).
Grad-finite asserted every step; early stopping uses validation AUC ONLY.
