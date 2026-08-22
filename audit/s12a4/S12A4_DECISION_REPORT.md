# S12A4 DECISION REPORT — Controlled Training Wave
288/288 fold-trainings complete (12 GPU jobs; 4 arms x 3 seeds x [5 ordinary + 19 LOSO]
frozen folds). All grad-finite gates passed; arm4 keep-rate valid; production untouched.
Adversarial pre-wave review: leakage lens ALL-CLEAR; 1 blocker fixed before launch.

## Results (pooled OOF AUC, frozen folds, S11 harness for h/z/flat; head = model's own OOF)
| arm | head | h (SVM) | z (SVM) | LOSO (h) |
|---|---|---|---|---|
| 1 CE                  | 0.6383 | **0.6502** | 0.6491 | 0.6594 |
| 2 CE+InfoNCE          | 0.6348 | 0.6480 | 0.5778 | 0.6318 |
| 3 A-GCL corrected     |   —    | 0.5672 | 0.5055 | 0.5803 |
| 4 A-GCL hard-mask 0.8 |   —    | 0.5665 | 0.5203 | 0.5539 |
Winner (CE) detail: acc 0.5968, bacc 0.5963, sens 0.5853, spec 0.6072; flatten-2880 probe
0.6429 (0.6366/0.6298/0.6623). Train-val gaps: CE +0.31, CE+NCE +0.27 (strong overfitting
yet capped at ~0.65); A-GCL +0.12/+0.08. Mean best epochs 20-31.

## The four pre-registered questions
1. CAN SUPERVISED LEARNING RECOVER SIGNAL? Partially. CE head OOF 0.6383 -> band
   **"ENCODER COMPRESSION LIMITS PERFORMANCE"** (0.60-0.68). Best supervised readout
   0.6502 / LOSO 0.6594, still ~0.11 below the frozen FC baseline 0.7565. Decisive extra:
   trained-encoder flatten retention (0.6429) does NOT exceed the RANDOM norm=F encoder
   (0.6539, S12A3 arm B) — supervision improves the readout, not WGIN retention itself.
2. DOES INFONCE HELP? No. 0.6480 vs 0.6502 (h); head 0.6348 vs 0.6383; z drops to 0.5778.
3. DOES FULL A-GCL WORK AFTER REPAIR? No. 0.5672 (h), z 0.5055 ~ chance.
   Pre-registered ordering test: A-GCL > CE+NCE > CE = FALSE; CE > A-GCL = TRUE
   -> **the contrastive objective actively hurts.**
4. WAS MASK COLLAPSE THE REASON? **No.** Collapse genuinely occurs (soft keep 0.47->0.03
   in arm3; arm4's soft mu 0.32->0.0004) but the hard 80% control that structurally
   prevents it performs identically (0.5665 vs 0.5672).

## Winner validation
Label-permutation control (arm 1, seed 20260818, RNG(20260818), full pipeline rerun):
head 0.4808, h-probe 0.4728 — both <= 0.55 -> **PASS, no leakage**.

## Information chain (final form after S12A4)
FC baseline 0.7565 > random norm=F flatten 0.6539 >= trained CE h 0.6502 ~ trained flatten
0.6429 > baseline random flatten 0.6217 >> A-GCL trained 0.567 >> trained z-32 (S8) 0.49.
The WGIN propagation layer is the compression bottleneck; no tested objective overcomes it,
and the A-GCL objective subtracts ~0.08 from what the same architecture achieves with CE.
Per protocol: STOP. No tuning, no extra sweeps, no architecture changes performed.
