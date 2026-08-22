# S12A5 SMOKE REPORT — PASS (8 subjects, 1 epoch, seed 20260818, CPU)
Forward/backward/loss finite for arms A (loss 0.7509), B (0.7332), C (0.7065); gradient
norms finite and logged per group (arm C encoder-grad = 0 correctly: no encoder exists).
Checkpoint save->reload: outputs allclose 1e-6 all arms. Mini train_fold5 end-to-end OK
(val loss + val AUC + grad norms per epoch; epoch-0 embeddings captured; BATCH=4 smoke-only
override). Edge ordering & FC values: bitwise vs canonical at float64 source (0.0);
float32 tensor cast quantization <= 3e-8 (same precision class as graph edge weights).
Wave-2 transductive step (corrected-C, S12A4 step math): grad finite, keep_mu 0.3713 at
init. ROI-permutation control path exercised. Production tree unchanged. ES = val AUC only.
