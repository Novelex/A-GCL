# S10 DECISION REPORT — ROI-AWARE READOUT x MASK-BUDGET FACTORIAL
2026-08-20 | git HEAD f6c49d49761ef3bf82b3d59798ee1e8ace6b62dc | production untouched
2x2 arms x 3 seeds (20260818/19/20); arm A seed 20260818 reused from S9. 11 new trainings,
all COMPLETED on node07 L40S (~3 min each), 33 probes on frozen S3C splits, S7.5 probe code.
Data/cache gate passed in every run (dataset sha 312266b2..., splits sha 28fed44d...,
ROI manifest sha a7632cd9..., x rows asserted equal to frozen M1_B). Gradients finite and
masks EXACTLY symmetric (max err 0.0) in all 11 runs.

## RESULTS (across-seed mean AUC, n=3; full per-seed table in S10_RESULTS.md)
  representation      A        B        C        D
  post_norm_nodes   0.5936   0.5913   0.5868   0.5924    (training-side signal: unchanged by arms)
  h (readout)       0.4771   0.5213   0.4931   0.5065
  z (PRIMARY)       0.4889   0.4857   0.4779   0.4940

## PRE-REGISTERED VERDICT (primary endpoint: final-epoch z, across-seed means)
  B - A = -0.0033   (rule >= +0.05)  NOT MET
  C - A = -0.0111   (rule >= +0.05)  NOT MET
  D - max(B,C) = +0.0083 (rule >= +0.03)  NOT MET
  best arm z = 0.4940 (rule >= 0.60)  NOT MET
  ALL MODIFIED ARMS z < 0.55  ->  REGISTERED OUTCOME:
     STOP node-only A-GCL modifications.
     Recommend an edge-aware model preserving fixed FC-pair identity.
(AUC throughout; no accuracy claim is made or implied. Accuracies are reported alongside
in S10_RESULTS.md and sit near chance for h/z in all arms.)

## TREATMENT-FIDELITY CAVEAT (stated plainly, affects interpretation of C/D)
The 80% budget did NOT hold during training: final expected keep was 0.08-0.25 (C) and
0.02-0.03 (D), despite the pre-training unit test proving the budget gradient points
toward 0.80 from both sides. With the mandated single setting (target 0.80, lambda 2.0,
no tuning), the adversarial InfoNCE pressure overwhelms the squared-error budget term.
Therefore C/D test "target_keep(0.80, lambda=2) as an intervention" — which fails — and
NOT "an enforced 80% keep rate", which was never achieved. Any claim that an enforced
keep rate would not help is NOT licensed by S10.

## SECONDARY OBSERVATIONS (not verdict-bearing)
1. The ROI-aware readout DOES move h in the expected direction: h(B)-h(A) = +0.044
   (0.5213 vs 0.4771, consistent across all 3 seeds). But the projection/contrastive
   stage erases it again (z(B) = 0.4857). The bottleneck is therefore not readout
   CAPACITY — the flat-node probe (0.59) proves a linear map can extract the signal —
   but the TRAINING OBJECTIVE, which does not select for diagnosis-relevant structure
   in any arm.
2. post-norm node accessibility is arm-invariant (0.587-0.594): neither the readout nor
   the budget changes what the encoder itself retains.
3. The ROI-aware arms drove masks MORE extreme (B keep -> 0.01-0.02): with a higher-
   capacity readout the adversary sparsifies harder.
4. Deterministic final mu saved for the 20 pre-registered subjects in every arm/seed.

## LIMITATIONS
3 seeds; single mandated budget setting; AAL90/954 dataset; linear probes; GPU training
(CUDA nondeterminism documented; state_dict reload verified bitwise in every run).

## VERDICT (exactly one)
All modified arms < 0.55 z-AUC -> node-only modifications of A-GCL are exhausted under
the registered rules: neither an ROI-identity-preserving readout, nor the mask budget,
nor both together move the end-to-end representation off chance, while the same trained
encoders demonstrably still carry ~0.59 node-level signal.

## SINGLE NEXT ACTION (from the registered decision table)
Design an edge-aware model preserving fixed FC-pair identity (the S5.5/S7.5 evidence
locates the discriminative signal in identity-aligned FC edges: 4005-edge linear model
0.7565 / LOSO 0.7432, destroyed by ROI permutation). Await authorization before
implementing.

STOP. Nothing further implemented.
