# S6 — WGINConv MATHEMATICS + COMPLETE-GRAPH MAGNITUDE / COLLAPSE AUDIT
2026-08-19 | frozen baseline 8cac2358ff12bcfa7452c38c4f4ef5e058814289 | tree CLEAN
No training, no tuning, no graph-cache change, no production-code change, S7 not begun.

## 1. EXACT MATHEMATICAL OPERATOR (proven, not inferred)
For one layer, with eps = 0 fixed and MLP = self.lin:
      Q = (I + E^T) X                      (message_relu = False, paper-literal)
      Q = X + E^T relu(X)                  (message_relu = True, original behaviour)
      H = MLP(Q)
Per node:  q_v = (1+eps) x_v + sum_u E[u,v] x_u , eps = 0 -> coefficient exactly 1.
NOTE THE TRANSPOSE. It is E^T, not E. See item 5.

## 2. PAPER / ORIGINAL / CURRENT COMPARISON
                       PAPER (Sec 2.2.1)        ORIGINAL qbmizsj         CURRENT fork
aggregation            sum                      aggr='add'               aggr='add'
degree normalization   none                     NONE                     NONE
softmax                none                     NONE                     NONE
edge weights           raw e_uv                 raw                      raw
message ReLU           ABSENT from the equation F.relu(x_j) ALWAYS,      message_relu flag,
                                                no flag                  default True
eps                    absent (no epsilon)      eps=0., train_eps=False  same
matrix form            H^(k-1/2)=(I+A∘E)H       (I+E^T)relu-weighted     same, flag-dependent
Files: current unsupervised/convs/wgin_conv.py:13-53, unsupervised/encoder/tu_encoder.py:54;
original git bed5441:A-GCL/unsupervised/convs/wgin_conv.py:13-48, .../tu_encoder.py:48.
The current fork's ONLY divergence from the original is the added message_relu switch.
Default True reproduces the original exactly; False is the paper-literal path.

## 3. eps / train_eps VERDICT — PROVEN FROM ALL CALL SITES
Every WGINConv construction found in either tree:
  current  tu_encoder.py:54  WGINConv(nn, message_relu=self.message_relu)
           uni_encoder.py:52 WGINConv(nn)
           gatv2_encoder.py:53 commented out
           tests/test_paper_gin.py:24,38,49 and layertesting/... (message_relu only)
  original tu_encoder.py:48  WGINConv(nn)
           uni_encoder.py:53 WGINConv(nn)
           gatv2_encoder.py:53 commented out
grep for train_eps across both trees returns ONLY the two __init__ definitions
(wgin_conv.py and gine_conv.py). NO CALL SITE ANYWHERE SETS train_eps=True.
=> eps is a registered BUFFER fixed at 0.0, non-learnable, in both trees. The residual
   coefficient (1+eps) is exactly 1.0, matching the paper, which has no learnable epsilon.

## 4. HAND-vs-CODE ERROR (tiny 4-node, 2-feature, float64, asymmetric E)
Graph deliberately contains positive, negative, fractional, ZERO and explicit self edges,
x has both signs, and E is genuinely asymmetric (max|E - E^T| = 3.5).
  max_abs_error vs  E X + X  : 6.175e+00      (WRONG operator)
  max_abs_error vs E^T X + X : 0.000e+00      (CORRECT, exact)
  message_relu=True vs E^T relu(X) + X : 0.000e+00
Requirement was < 1e-6; achieved exactly 0. Node-0 scalar check reproduced by hand.

## 5. SOURCE / TARGET ORIENTATION  ***RECORD FOR S8***
Loader layout: edge k = i*N + j has edge_index[0] = i, edge_index[1] = j, weight = E[i,j].
PyG MessagePassing default flow='source_to_target' gathers x_j := x[edge_index[0]] and
scatters into out[edge_index[1]]. Therefore
      out[j] = sum_i E[i,j] x_i = (E^T X)[j].
Because real FC is SYMMETRIC (S4: max|FC-FC^T| = 2.22e-16) this is invisible on real data.
A learned Bernoulli edge mask in S8 CAN be asymmetric, at which point the implemented
operator uses the TRANSPOSE of the intended matrix. This must be checked in S8.

## 6. DOUBLE-SELF COEFFICIENT
With explicit e_vv = 1 (S4 proved diag(FC) = 1.0 exactly for all 954) and eps = 0:
  residual self      = (1+eps) x_v = x_v
  explicit FC self   = e_vv x_v    = x_v
  TOTAL SELF         = 2 x_v
Tiny graph: q - offdiag == 2x exactly (True); with the diagonal removed it becomes exactly
x_v; max|q_with - q_without - x| = 0.000e+00.
Real graphs, all 954: |resid_norm - expl_norm| <= 8.9e-16 and total_self/resid = 2.000000
for every branch B, C, D.
ALL THREE (paper matrix equation with A all-ones and PCC E_ii=1, original code, current
code) produce the literal 2*x_v behaviour. The paper's neighbourhood notation
"sum_{u in N_v}" is textually ambiguous about whether v is in N_v, but the PRINTED MATRIX
EQUATION H^(k-1/2) = (I + A∘E)H is unambiguous and is treated here as the contract: with
A all-ones and E_ii = 1 it yields (1 + 1) x_v.

## 7-8. NEGATIVE EDGES AND message_relu
Tiny graph, message_relu False vs True differ by max 2.0 (ReLU is not a no-op there).
Worked example E[1,0] = 0.25, x_1 = [-0.5, 3.0]:
  A relu=False: e*x_u       = [-0.125, 0.75]   (sign preserved, can subtract)
  B relu=True : e*relu(x_u) = [ 0.0  , 0.75]   (negative component zeroed BEFORE weighting)
So with relu=True a negative feature contributes nothing regardless of edge sign; the
edge's sign then only ever acts on the non-negative part of x.
REAL DATA (verified empirically, every 6th subject, layer 1 pre-BN):
  branch   min x        n(x<0)     max|relu=T - relu=F|   identical
  M1_B     0.000000          0          0.000e+00         True
  M1_C     0.000000          0          0.000e+00         True
  M1_D    -2.819696      24611          2.576e+01         False
Confirmed: for B and C (min-max to [0,1], x >= 0) message_relu is EXACTLY a no-op at
layer 1. For D it bites immediately.
CAVEAT: this no-op holds ONLY at layer 1 pre-BN. After BatchNorm the layer-2 input is
centred and contains negatives for ALL branches, so message_relu will matter at layer 2
for B and C too. That belongs to S7.
No judgement is made here about which setting is better.

## 9. GRADIENT CHECK (float64, autograd vs central differences, h = 1e-6)
  x               max_abs_err 3.16e-09   max_rel_err 3.78e-10
  edge_weight     max_abs_err 6.07e-09   max_rel_err 4.77e-09
  Linear.weight   max_abs_err 4.01e-09   max_rel_err 7.59e-11
  Linear.bias     max_abs_err 8.17e-10   max_rel_err 1.47e-10
All finite; no NaN, no Inf. Gradients flow correctly to edge weights (needed for S8).

## 10. REAL-GRAPH MAGNITUDE DECOMPOSITION — ALL 954 GRAPHS  *** KEY RESULT ***
q_v = residual_self + explicit_FC_self + offdiag ; norms averaged per subject.
  branch  resid  explicit  total_self  offdiag       q    off/self (median)  self/q (median)
  B      0.5041   0.5041     1.0083   14.8940  15.8814        20.02             0.0776
  C      0.5946   0.5946     1.1891   17.4691  18.6322        27.71             0.0777
  D      1.4076   1.4076     2.8152    5.0263   6.5004         2.84             0.6282
THE OFF-DIAGONAL AGGREGATE DOMINATES. For the paper input B the 89 neighbours contribute
about 20x the combined self term, and the node's own identity is only ~8% of q_v. For D it
is 2.8x and ~63%.
HONESTY NOTE: the MEAN off/self ratio for C is astronomically large (4.7e10) because C's
per-band min-max sets the minimum ROI of each band to EXACTLY 0, so a node that is the
per-band minimum can have a near-zero ||x_v|| and the ratio diverges. The mean is a
degenerate-denominator artifact; the MEDIAN quoted above is the honest statistic. (No node
had ||x_v|| < 1e-12 exactly, so nothing is undefined, only ill-conditioned.)

## 11. SIGN-CANCELLATION DISTRIBUTION
cancellation = 1 - ||sum_{u!=v} e_uv x_u|| / (sum_{u!=v} |e_uv| ||x_u||)
  branch   mean     sd      min     25%     50%     75%     max
  B       0.1038  0.0980  0.0040  0.0326  0.0724  0.1414  0.5699
  C       0.1055  0.0981  0.0046  0.0340  0.0734  0.1435  0.5728
  D       0.8680  0.0625  0.5249  0.8353  0.8751  0.9134  0.9735
DECISIVE ANSWER to "is dense 89-neighbour aggregation truly enormous or mostly cancelled":
For B and C it is TRULY ENORMOUS — only ~10% cancels, because x >= 0 and 91% of FC edges
are positive (S4), so messages overwhelmingly ADD. For D it is MOSTLY CANCELLED (87%),
because per-band z-scoring makes x zero-mean so the signed sum largely destructively
interferes. Node-level (85860 nodes, B): mean 0.1038, sd 0.1658, range 0.0031-0.9933.
Per-site means span 0.040 (CMU) to 0.181 (YALE) — cancellation is itself site-structured.

## 12. X -> Q -> WGIN1  (identical deterministic seeded MLP for every condition)
  [M1_B]   metric              X         Q      WGIN1
           across_node_var  0.0253    6.1139    0.2170
           mean_cos         0.9636    0.9765    0.9743
           eff_rank         1.8694    1.3720    1.9105
           frac_var_sv1     0.9159    0.9894    0.9752
           mean_pair_dist   0.3142    4.5887    2.8529
  [M1_C]   mean_cos 0.9477 -> 0.9763 -> 0.9754 ; eff_rank 1.8746 -> 1.3959 -> 1.9063
           frac_var_sv1 0.9206 -> 0.9888 -> 0.9741
  [M1_D]   mean_cos 0.0152 -> 0.1076 -> 0.4712 ; eff_rank 1.8782 -> 1.8642 -> 5.9642
           frac_var_sv1 0.9196 -> 0.8891 -> 0.6984
READ CAREFULLY, and NOT from cosine alone:
 - For B and C the aggregation step MAKES NODES MORE ALIKE on every metric at once:
   cosine rises, effective rank FALLS 1.87 -> 1.37, and the first singular component grows
   to hold 98.9% of the variance. Three independent metrics agree, so this is a genuine
   rank-collapse toward an almost rank-1 node representation BEFORE any training.
 - For D the same operator does NOT collapse: eff_rank is preserved (1.878 -> 1.864),
   frac_var_sv1 falls slightly, cosine stays low (0.015 -> 0.108).
 - The first MLP partially restores rank (B/C to ~1.91, D to 5.96) but for B/C mean cosine
   stays at 0.974, i.e. the MLP does not undo the collapse, it re-expands a nearly
   collinear input into a higher-dimensional but still nearly collinear representation.

## 13. IS WGIN ALREADY AN FC-STRENGTH ENCODER? (node level, M1_B, 85860 nodes)
Correlations with node signed strength s_v, absolute strength a_v, positive/negative parts:
  target     r(s_v)  r(a_v)  r(pos)  r(neg)  r(||x_v||)  rho(a_v)
  q_norm     0.8926  0.8850  0.8946  0.5484    0.2703     0.8917
  w1_norm    0.8919  0.8887  0.8958  0.5330    0.2675     0.8937
  q0/q1/q2   0.899 / 0.817 / 0.867 with s_v
Variance explained:
  target     R2 FC-strength only   R2 ALFF x_v only   R2 both   gain from adding ALFF
  q_norm           0.8010               0.0731        0.8506          +0.0497
  w1_norm          0.8026               0.0716        0.8501          +0.0475
=> ~80% of the first-layer node representation is already explained by pure FC connectivity
   strength, versus ~7% by the node's own ALFF. Adding ALFF on top of FC strength buys
   about 5 percentage points. At layer 1 the WGIN node embedding is predominantly an
   FC-strength encoder.

## 14. GRAPH-LEVEL PRE-POOLING DIAGNOSTIC (sum over 90 nodes, 954 subjects, NO labels)
Correlation of each summed dimension with simple global FC summaries:
  max |r| over all graph dims = 0.8868 (Qsum0 with fc_mean); median max|r| = 0.8607
  median R2 from 6 FC summaries alone = 0.7426 (max 0.7887)
  dominant drivers: fc_mean and fc_pos; W1sum_norm r = 0.8819 with fc_pos, R2 = 0.7801
=> Before any training, a summed graph-level representation is already ~74-79% explained by
   trivial global FC summary statistics. Adding total ALFF level raises R2 to 0.91-0.98 for
   several dimensions, but FC summaries alone carry most of it.

## 15. DIAGNOSTIC EDGE ABLATIONS (controls only — NOT paper or original variants)
First 200 subjects, node features M1_B, identical seeded MLP.
  ablation          q_norm  Q_mean_cos  Q_eff_rank  Q_frac_var_sv1  Q_node_var  W1_eff_rank
  1_raw_signed     16.2948     0.9643      1.3493       0.9907        7.3505       1.9487
  2_abs            17.3102     0.9993      1.3821       0.9883        5.3022       1.5946
  3_pos_only       16.7802     0.9989      1.3670       0.9894        6.1208       1.6347
  4_all_ones       45.6329     1.0000      1.8523       0.9228        0.0260       1.9938
  5_offdiag_only   15.7984     0.9581      1.3097       0.9926        7.1890       1.9174
Interpretation, separating the four effects:
 - DENSE CONNECTIVITY is the dominant collapse driver: all-ones E gives mean cosine exactly
   1.0000 and across-node variance 0.026 — every node becomes the same vector. Its higher
   eff_rank (1.85) is an artifact of near-zero variance, not diversity; read it together
   with Q_node_var.
 - EDGE MAGNITUDES matter: real magnitudes (raw/abs/pos) keep node variance ~5-7 vs 0.03
   for all-ones.
 - SIGN is what preserves the most diversity: raw signed E has the LOWEST cosine (0.9643)
   of all real-magnitude variants; abs(E) 0.9993 and positive-only 0.9989 are far more
   collapsed. Signed FC is protective, not harmful, for node diversity.
 - The EXPLICIT DIAGONAL contributes little to collapse: removing it (5) changes cosine
   only 0.9643 -> 0.9581 and eff_rank 1.349 -> 1.310.
DO NOT read ablation 3 as "original A-GCL". The original applies ReLU to x, not to E.

## 16. B / C / D DIFFERENCES (summary)
                                   M1_B         M1_C         M1_D
off/self ratio (median)            20.02        27.71         2.84
self share of q (median)           0.0776       0.0777        0.6282
sign cancellation (mean)           0.104        0.106         0.868
eff_rank X -> Q                 1.87 -> 1.37  1.87 -> 1.40  1.88 -> 1.86
mean cosine X -> Q              0.964 -> 0.977 0.948 -> 0.976 0.015 -> 0.108
message_relu no-op at layer 1      YES          YES           NO
B and C behave almost identically at this layer; D is qualitatively different on every
magnitude and geometry metric. No judgement of which is preferable is made in S6.

## 17. CPU / GPU AGREEMENT (Part 12)
Login node (CPU), SLURM CPU compute node (job 1869260), and SLURM GPU node16
NVIDIA H100 NVL 95830 MiB (job 1869261, 4 s) all ran the identical tiny float64 test.
  q (message_relu=False): max_abs_diff CPU vs GPU = 0.000e+00, bitwise_identical = True
  q (message_relu=True) : max_abs_diff CPU vs GPU = 0.000e+00, bitwise_identical = True
Only numerical agreement was required; in float64 the results are bitwise identical.
Scheduling note: an initial GPU request on gpu-l40s with 4 CPUs could not be scheduled
(every l40s node holding a free GPU had 0-1 idle cores); it was cancelled and resubmitted
to gpu-h100 with 1 CPU / 4 GB, which started immediately.

## 18. REAL-GRAPH SANITY (Part 13)
output shape (N,out_dim), all finite; edge_weight = 0 gives q == x exactly (err 0.0e+00);
E = Identity gives q == 2x exactly; repeated forward with fixed weights is BITWISE
identical; negative edges are demonstrably consumed (clamping E at 0 changes the output);
no hidden normalization or softmax exists anywhere in the message or aggregation path.

## PRE-BN BASELINE HANDED TO S7
S7 must ask what BatchNorm, the second WGIN layer, F.normalize and global_add_pool change
relative to THESE pre-BN numbers:
  M1_B  Q: across_node_var 6.1139, mean_cos 0.9765, eff_rank 1.3720, frac_var_sv1 0.9894
        WGIN1: across_node_var 0.2170, mean_cos 0.9743, eff_rank 1.9105, frac_var_sv1 0.9752
  M1_C  Q: mean_cos 0.9763, eff_rank 1.3959   WGIN1: mean_cos 0.9754, eff_rank 1.9063
  M1_D  Q: mean_cos 0.1076, eff_rank 1.8642   WGIN1: mean_cos 0.4712, eff_rank 5.9642
Recorded for S7: the ORIGINAL TUEncoder applies, per layer, conv -> BatchNorm1d -> (ReLU on
all but the last layer) -> dropout, then an UNCONDITIONAL F.normalize(x, dim=1) before
global_add_pool. The current fork gates the post-BN ReLU behind post_bn_relu and the final
normalize behind normalize_nodes (argparse default True; the paper-exact profile sets
normalize_nodes False). The paper-literal path differs from both.

## UNRESOLVED ISSUES
1. The E vs E^T transpose is harmless today only because FC is symmetric. If S8's Bernoulli
   mask is applied asymmetrically the implemented operator silently transposes it. NOT a
   defect yet; a hazard to test in S8.
2. The paper's "sum_{u in N_v}" text does not state whether v in N_v; only the matrix
   equation disambiguates it. The 2*x_v behaviour follows from the printed equation plus
   PCC E_ii = 1, but the authors may not have intended a doubled self term.
3. Rank collapse at Q for B/C is measured PRE-BatchNorm. BatchNorm rescales per feature
   across the batch and may substantially change these numbers; that is S7's question, and
   nothing here should be read as a statement about the trained model.
4. Ablations were run on the first 200 subjects (representative, not random) with M1_B only.
5. The seeded MLP is one deterministic draw; rank statistics after WGIN1 depend on that
   initialization. The X -> Q comparison is initialization-free and is the robust part.
6. All analysis is single-layer. Two-layer behaviour is deliberately out of scope.

## GIT STATE
HEAD 8cac2358ff12bcfa7452c38c4f4ef5e058814289, branch main, working tree CLEAN, in sync
with origin/main. Nothing written inside the repo. All S6 evidence in
/users/3171356m/agcl_audit_s0/s6/ : tiny_test.py, real_graph.py, s6_magnitude.csv,
s6_nodes.csv, s6_graph.csv, s6_ablation.csv, s6_parts6_8.txt, s6_parts9_11.txt,
s6_relu_real.txt, tiny_cpu.pt, tiny_cuda.pt, gpu_1869261.out, cpunode_1869260.out.
SLURM: 1869260 (CPU node) and 1869261 (H100) both COMPLETED; 1869259 cancelled unscheduled.

S6 STATUS: EVIDENCE COMPLETE — nothing fixed, nothing trained, S7 not begun.
