# S12B DECISION REPORT — the twelve commissioned questions

## 1. Was the data correct and the cache clean?
Yes. Fresh namespace `data_s12b_4f01b0abc1b160bd.pt`, delete-and-rebuilt, never
reusing data_dense_v3.pt or any earlier .pt. n=954 (455 ASD / 499 NC), 90 nodes and
8100 directed edges per graph. Every Gate-0 item is an EXECUTED assert (after the
pre-launch review found several were only being printed): FC rebuilt from the .mat
files equals the frozen X_fc BITWISE; .mat-vs-S5-cache mismatches = 0; symmetry
2.2e-16; diag deviation 2.2e-16; node features == canonical M1_B bitwise; X_fc
sha256 == frozen S11 sha; splits sha 28fed44d == S3C authority. No try/except and
no fallback anywhere in the load path.

## 2. Was the measuring instrument valid?
Yes, and this matters because a weak probe would have manufactured the answer.
R2 (frozen LinearSVC path) = 0.7565 — the pre-existing anchor reproduced EXACTLY.
R1 (the audit probe) = 0.7481, only 0.008 below the SVC, so the probe is not the
bottleneck. R3 = 0.4877 mean over 10 permutations (0.465-0.514) -> no leakage.
R4 ALFF floor = 0.6315. NOTE the amendment trail: the FIRST Gate-1 run FAILED on a
single-draw R3 of 0.4647; rather than waive it I diagnosed the criterion as a
+-1.6 SE test (~11% false-stop rate) failing on the below-chance side, where
leakage cannot reach, and replaced it with a STRICTER mean-of-10 + max<=0.55 rule
(amendment A1, written before the re-run).

## 3. Was the forward pass correct?
Yes, 14/14 Gate-2 checks. WGINConv hand-verified against a 4-node toy graph with a
negative edge weight and a sign-flipping node, both message_relu branches, f64
<1e-6 and f32 <1e-4. Replica-vs-production parity <1e-5 on node and pooled outputs.
A1 identity verified against conv.propagate. Save/reload bitwise. Gate 2 also
CAUGHT A REAL DEFECT: CUDA run-to-run non-determinism (~6e-7 relative) from the
atomicAdd-ordered scatter in propagate. Instead of weakening the criterion I
measured it and fixed it — deterministic algorithms + cuBLAS workspace pin give
bitwise 0.0 — so the whole audit is now exactly reproducible on GPU.
SELF-LOOP FORK (documented, not silently fixed): edge_index contains (i,i) with
FC[i,i]=1 AND WGINConv adds (1+eps)*x_r, so a node's own features enter with total
coefficient 2. All arms KEEP the diagonal, matching production.

## 4. At which NAMED operation does the FC signal die?
TWO serial operations, and the pre-registered hypothesis about which one dominates
was WRONG in an informative way.

(a) THE FIRST WGIN BLOCK — `Linear(d_in -> emb_dim)` + `BatchNorm1d` + `ReLU`
    (tu_encoder.py lines 51/57/72, applied through WGINConv.forward -> self.lin).
    Arm B ENTERS this block at retention 0.967 with FC-reconstruction R2 = 0.753
    and LEAVES it at 0.590 (production emb32/BN) or 0.707 (best emb128/LN), with
    FC R2 falling to 0.564. That is a PERMANENT loss of 0.26-0.38 retention in a
    single operation; no downstream stage ever recovers it.
(b) `global_add_pool` (tu_encoder.py line 80). Retention collapses to -0.05..0.12
    — chance. But this loss is RECOVERABLE: ROI-flatten restores +0.274.

So the readout is the louder failure and the first block is the fatal one.

## 5. Which of B1-B5 are confirmed, with numbers?
B1 (agg = X + FC@X is rank-3): PARTIALLY CONFIRMED — for arm A only. ALFF(3) gains
   just +0.03 retention through A1 (0.530->0.564). But given ANY full-rank node
   basis the same aggregation reaches 0.928-0.967. The aggregation is NOT the wall;
   the INPUT RANK was.
B2 (emb_dim compression): CONFIRMED, modest. 32=0.490, 64=0.503, 128=0.552 (+0.062).
B3 (BatchNorm): CONFIRMED, the largest single knob. bn=0.450, ln=0.537, none=0.557.
   BatchNorm is the worst normalisation of the three (-0.107 vs none).
B4 (F.normalize): REJECTED. 0.510 vs 0.515 — within noise. Node magnitude was not
   where the signal lived.
B5 (global_add_pool): CONFIRMED. +0.274 from switching to ROI-aware readout.

## 6. Does any configuration recover retention >= 0.95?
NO. Best S6 (encoder output) over all 1080 configurations = 0.690. Best at ANY
encoder stage = 0.834 (arm D, emb32, BN, S2). Meanwhile the signal is demonstrably
PRESENT at 0.967 immediately before the encoder (arm B, S1, param-free).
=> Pre-registered decision rule 8 FIRES: no config anywhere exceeds 0.85 at the
encoder output. THIS ARCHITECTURE FAMILY CANNOT CARRY THIS SIGNAL. Stop modifying
WGIN. Rule 7 (training/objective is the wall) does NOT fire.

## 7. What is the representation encoding INSTEAD?
Mean FC strength. Its R2 stays at 0.945 -> 0.943 -> 0.908 -> 0.890 across S1, S2,
S3 and even after pooling, while diagnosis retention falls 0.803 -> 0.057. Site
macro-F1 also decays (0.668 -> 0.100), so the representation is NOT primarily
encoding scanner; age R2 0.408->0.010 and motion R2 0.328->0.122 likewise decay.
The encoder converges to a global-connectivity-strength detector: it keeps the one
scalar property that survives sum-pooling and discards the edge-identity structure
that carries diagnosis.

## 8. What is the honest ceiling from Track 2?
0.7565 ordinary / 0.7432 LOSO — and it belongs to a LINEAR model. Best MLP over
36 runs (3 hidden widths x 4 weight decays x 3 seeds) = 0.7246 / 0.7090, i.e. ~0.03
BELOW LinearSVC and ridge logistic (0.7561). 96-100% of folds are OVERFIT by the
pre-registered rule; capacity HURTS (h512 worse than h64); decoupled weight decay
across four orders of magnitude barely moves the result. With 763 training subjects
and 4005 features, no learned representation in this audit has beaten a linear
model on raw edges.

## 9. Was ALFF normalisation a second leak?
Yes, small but real and independent. Raw ALFF 0.6423 / LOSO 0.6174 beats both
per-subject variants: production v3 per-band min-max 0.6322 / 0.6119 and the frozen
joint min-max 0.6315 / 0.6062. Cost ~0.010 ordinary, ~0.011 LOSO. The per-subject
rescaling in datasets/abideDataset.py (lines 71-74) deletes between-subject
amplitude before the graph is built, and that information is worth ~1 AUC point.
Instrument invariance verified exactly (|raw - z-across-subjects| = 0.000000).
Separately: ALFF contributes only +0.004 on top of FC.

## 10. How large is the protocol inflation?
E-best minus E-final = +0.0442 +- 0.0060 (inductive), +0.0180 +- 0.0100
(transductive). T-trans minus T-ind = +0.0176 +- 0.0139. Reported as measured
deltas on the unchanged production backbone; no claim is made about the paper's
published number. Context that makes these deltas alarming: the backbone itself
sits AT CHANCE (0.479-0.510), so best-of-20-evals reporting can lift a chance-level
model by more than 4 AUC points.

## 11. Mapping to the A-GCL paper / code
- tu_encoder.py:51 `Linear(num_dataset_features, emb_dim)` and :57
  `BatchNorm1d(emb_dim)`, and :72 the post-BN ReLU — THE fatal compression. With
  ALFF(3) inputs the block receives a rank-3 signal; with a full-rank basis it
  receives 0.967 retention and still discards a third of it.
- tu_encoder.py:80 `global_add_pool` — sum over 90 nodes destroys ROI identity;
  the one loss that a readout change fully recovers (+0.274).
- wgin_conv.py:46 `out += (1 + self.eps) * x_r` combined with FC[i,i]=1 self-loops
  double-counts each node's own features (documented fork).
- wgin_conv.py:52 `F.relu(x_j)` inside message() — message_relu is near-neutral
  in our grid (the paper-literal branch is not the problem).
- datasets/abideDataset.py:71-74 per-subject per-band min-max — the ~0.010 leak.
- The paper's node feature choice (ALFF, 3 channels) is the root enabler of B1:
  arm A never rises above 0.564 retention at ANY stage.

## 12. The single next scientifically justified step
NONE of the remaining options is "improve WGIN". Decision rule 8 has fired: the
information is present at 0.967 before the encoder and no configuration of this
architecture family retains more than 0.69 of it at the output, while a linear
model on the raw edges reaches 0.7565.

RECOMMENDED SINGLE STEP: write up A-GCL-on-ABIDE as a NEGATIVE METHODS RESULT.
The audit now has, with pre-registered criteria and passing controls, all four
components a methods contribution needs: (i) the signal is linearly accessible at
0.7565; (ii) it survives to 0.967 through the raw graph aggregation; (iii) it is
destroyed at two NAMED operations, one of them permanently; (iv) the published
protocol inflates by up to +0.044 on a backbone that is otherwise at chance.
No further experiment is required to support that claim, and no new architecture
should be built until it is written down.

STOP. No new architecture, no A-GCL run, no tuning without explicit authorisation.
