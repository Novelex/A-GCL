# S3B — ALFF NORMALIZATION AUDIT (mathematical behaviour only)
2026-08-18 | HEAD 906a494b… | tree ?? ALFF_func_proc/ | no code/data modified, no ALFF overwritten
Frozen: 954 subjects / 90 ROIs / 3 bands / M1, M2, OLD raw ALFF
No A-GCL training, no ComBat, no diagnosis labels read at any point, no method ranking.

## 1. EXACT FORMULAE AUDITED  (X[s,r,b], s=954, r=90, b=3)
A RAW    Y = X
B SUBJ-JOINT MIN-MAX   Y[s,r,b] = (X[s,r,b] - min_{r,b}X[s]) / (max_{r,b}X[s] - min_{r,b}X[s])
C SUBJ-BAND MIN-MAX    Y[s,r,b] = (X[s,r,b] - min_r X[s,:,b]) / (max_r X[s,:,b] - min_r X[s,:,b])
D SUBJ-BAND Z-SCORE    Y[s,r,b] = (X[s,r,b] - mean_r X[s,:,b]) / std_r X[s,:,b]
E TRAINFOLD-BAND MM    Y[:,:,b] = (X[:,:,b] - min_{s in TRAIN, r}) / (max_{s in TRAIN, r} - min_{s in TRAIN, r})
F TRAINFOLD-BAND Z     Y[:,:,b] = (X[:,:,b] - mean_{s in TRAIN, r}) / std_{s in TRAIN, r}
A-D are per-subject (no cross-subject statistic -> structurally leakage-free).
E-F fit on TRAIN subjects only (leakage-free by construction; proven in §11).

## 2. OUTPUT min / max / mean / std     (full 18-row table: s3b_norm_matrix.csv)
  M1  A raw   0.4609 / 62.0829 / 5.4787 / 2.5916      OLD A raw  0.0018 / 51.9378 / 4.4521 / 2.5017
  M1  B       0.0000 /  1.0000 / 0.2829 / 0.1756      M2  A raw  0.5396 /126.1842 /13.2849 / 4.4649
  M1  C       0.0000 /  1.0000 / 0.3371 / 0.2089      M2  B      0.0000 / 1.0000 / 0.3650 / 0.1911
  M1  D      -2.8197 /  7.1365 / 0.0000 / 1.0000      M2  D     -4.8445 / 6.5556 / 0.0000 / 1.0000
  M1  E       0.0000 /  1.0000 / 0.0943 / 0.0470      M2  E      0.0000 / 1.0000 / 0.1175 / 0.0416
  M1  F      -2.0780 / 19.9251 / 0.0021 / 0.9904      M2  F     -2.9112 /22.4054 / 0.0005 / 0.9796
  OLD E      -0.0002 /  1.0000 / 0.1098 / 0.0607   <- NEGATIVE: a non-train subject below the
                                                      train min. Correct, expected, and is itself
                                                      evidence of train-only fitting (§11).

## 3. NaN / Inf / ZERO-VARIANCE
Zero across ALL 18 source x transform combinations: n_nan = 0, n_inf = 0.
Degeneracy margins (min over 2862 subject-bands):
  source                     min span (min-max)   min sd (z)   n(span=0)  n(sd=0)
  M1                         3.190724             0.751785     0          0
  M2                         4.954804             1.002454     0          0
  OLD                        3.449912             0.788377     0          0
  norm_matrix (loader input) 3.724640             1.000000     0          0
Joint (B) min span: M1 5.033, M2 5.771, OLD 4.795 — all far from 0.
=> every transform is well-defined for THIS cohort. Not a general guarantee: see REJECTIONS.

## 4. ROI ORDERING / RANKING PRESERVATION  (min Spearman vs RAW)
                    within a subject-BAND (90 ROIs)   within a subject, all 270 values
  A raw             1.000000                          1.000000
  B subj-joint mm   1.000000                          1.000000   <- ONLY B preserves the full
  C subj-band mm    1.000000                          0.568759      270-element ordering
  D subj-band z     1.000000                          0.764355
  E fold-band mm    1.000000                          0.661729
  F fold-band z     1.000000                          0.885304
(worst value across the 3 sources shown). Every transform is a positive affine map WITHIN a
band, so within-band ROI ranking is mathematically exact for all six. Cross-band ranking
survives only under B, because B is a single affine map over the whole 90x3 matrix.

## 5-6, 8-9. SCALE / VARIANCE STRUCTURE  (scale-invariant; s3b_scaleinvariant.csv)
r_subjlevel = Pearson(subject mean before, subject mean after) -> 1 = between-subject scale
intact, ~0 = destroyed. share_* = fraction of total variance (immune to uniform rescaling).
  src transform         share_subj  share_band  share_roi   r_subjlevel
  M1  A raw                0.33030     0.04319    0.19569       1.00000
  M1  B subj-joint mm      0.09980     0.05803    0.29682       0.05872
  M1  C subj-band mm       0.06806     0.00079    0.35134       0.06615
  M1  D subj-band z        0.00000     0.00000    0.39017       0.00000
  M1  E fold-band mm       0.35601     0.00088    0.21129       0.99969
  M1  F fold-band z        0.35818     0.00000    0.21267       0.99940
  M2  A raw                0.58779     0.01022    0.10163       1.00000
  M2  D subj-band z        0.00000     0.00000    0.34927       0.00000
  M2  E fold-band mm       0.57682     0.03849    0.09877       0.99988
  M2  F fold-band z        0.59967     0.00000    0.10297       0.99994
  OLD A raw                0.25536     0.02984    0.35343       1.00000
  OLD D subj-band z        0.00000     0.00000    0.57882       0.00000
  OLD F fold-band z        0.27111     0.00000    0.37549       0.99952
BETWEEN-SUBJECT absolute scale: PRESERVED by E and F (r = 0.999+). DESTROYED by B, C
  (r = 0.03-0.07, and negative for M2) and exactly annihilated by D (r = 0, share = 0).
BETWEEN-BAND scale: PRESERVED by B (share_band rises slightly 0.043 -> 0.058; band-level
  r = 1.000). DESTROYED by C (share_band 0.043 -> 0.0008), by D (exactly 0), and by E/F
  (0.0009 / 0.0000) because each band receives its own independent transform.
=> B and {E,F} are complementary: B keeps band structure and discards subject scale;
   E/F keep subject scale and discard band structure. D discards both. Only A keeps both.
CAUTION recorded: my first pass used raw variance ratios, which conflate a uniform
rescaling with genuine structure loss (e.g. E's between-subject variance ratio reads
0.00035 purely because it divides by a span of ~61). All conclusions above use the
scale-invariant shares and r_subjlevel instead.

## 7. CORRELATION WITH RAW  (all 954x90x3)
                  M1 pearson/spearman    M2                    OLD
  B subj-joint mm  0.7444 / 0.7641       0.5013 / 0.5131       0.7877 / 0.8137
  C subj-band mm   0.6944 / 0.7277       0.4860 / 0.4974       0.7503 / 0.7858
  D subj-band z    0.7289 / 0.7564       0.5722 / 0.5632       0.7906 / 0.8236
  E fold-band mm   0.9701 / 0.9725       0.9502 / 0.9434       0.9270 / 0.9329
  F fold-band z    0.9661 / 0.9775       0.9920 / 0.9940       0.9747 / 0.9850
Per-subject transforms move the data much further from RAW than train-fold transforms,
and the effect is largest for M2 (whose raw between-subject variance share is 0.588).

## 10. TR / SITE SCALE PATTERNS  (NO diagnosis labels used)
site_eta2 = fraction of between-subject variance in subject mean level explained by SITE_ID.
  source  transform        corr(level,TR)  corr(level,N_VOL)  site_eta2
  M1      A raw               -0.1235          0.1421           0.1112
  M1      D subj-band z        0.0239         -0.0248           0.0095
  M1      E fold-band mm      -0.1214          0.1378           0.1114
  M1      F fold-band z       -0.1206          0.1362           0.1114
  M2      A raw                0.0247          0.0767           0.4243  <-- 42% site
  M2      B subj-joint mm      0.0402         -0.2498           0.2506
  M2      C subj-band mm      -0.0151         -0.1769           0.2606
  M2      D subj-band z        0.0385         -0.0089           0.0144
  M2      E fold-band mm       0.0272          0.0704           0.4306
  M2      F fold-band z        0.0265          0.0723           0.4287
  OLD     A raw               -0.0782          0.1401           0.1179
  OLD     D subj-band z        0.0041         -0.0281           0.0119
  OLD     F fold-band z       -0.0753          0.1346           0.1185
FINDING: M2 (voxel-first) carries a far larger site-linked absolute-scale component than
M1 or OLD (eta2 0.424 vs 0.111 / 0.118). E and F, being global affine maps, leave that
site structure fully intact (0.431 / 0.429). Only per-subject transforms attenuate it, and
D removes it almost entirely (0.0144). B/C attenuate it partially (0.25 / 0.26).
This is a structural tension for later stages, stated WITHOUT any diagnosis comparison:
the transforms that best preserve real between-subject signal also best preserve the site
confound; the transform that kills the site confound also kills all between-subject scale.
Also note B/C introduce a NEGATIVE correlation with scan length for M2 (-0.25 / -0.18)
that RAW does not have (+0.077) — a normalization-induced artefact, not a data property.

## 11. TRAIN-ONLY FITTING PROVED, NO TEST LEAKAGE
Design: outer KFold(5, shuffle, random_state=0) -> test; inner 80/20 of train -> val.
KFold is UNSTRATIFIED, so no diagnosis label is read anywhere in the procedure.
  fold n_train n_val n_test  E:test outside[0,1]  E:val outside  E:test outside if GLOBAL fit
   0     610    153   191            0                  3                   0
   1     610    153   191            0                  0                   0
   2     610    153   191            6                  0                   0
   3     610    153   191            9                  9                   0
   4     611    153   190            0                  0                   0
Four independent signatures of train-only fitting:
 (i)   splits asserted pairwise disjoint in all 5 folds (train n val n test = empty).
 (ii)  train-fitted parameters DIFFER from all-954 parameters: E band-max delta up to
       20.5690 (62.08 -> 46.35, a 33% change); F sd delta up to 0.0678, mean delta 0.0434.
 (iii) TEST values fall OUTSIDE [0,1] in 2/5 folds (15 values total) under train-only
       fitting, and in 0/5 folds under all-954 fitting. A test value outside [0,1] is only
       possible if the test extremes were never used to fit -> positive proof of train-only
       fitting; the 0 under global fitting is the observable SIGNATURE OF LEAKAGE.
 (iv)  F test statistics deviate from exactly (0,1): mean up to |0.0294|, std 0.9544-1.0974.
       Test mean/std of exactly (0,1) would itself indicate the test set was used.
Fold-to-fold parameter spread for E: 15.74 / 20.57 / 17.06 across the three bands.
Files: s3b_leakage_full.txt, s3b_leakage.json.

## WHAT NORMALIZATION EACH AUTHORITY ACTUALLY PERFORMS
### (a) CURRENT A-GCL LOADER — datasets/abideDataset.py  (verified by direct reading)
Node features, lines 71-74:
    x_min = x.min(dim=0, keepdim=True).values
    x_max = x.max(dim=0, keepdim=True).values
    span  = x_max - x_min
    x = torch.where(span > 0, (x - x_min) / span, x)
x is [90, 3]; dim=0 is the ROI axis; this sits inside the per-subject loop.
=> exactly transformation C: PER-SUBJECT, PER-BAND min-max to [0,1]. Fitted only on that
   subject's own 90 values -> NO train/test leakage. Applied inside Dataset.process(), so
   it is BAKED INTO the cached processed/data.pt (v3 cache).
Edge weights, lines 85-87: fc = fc / np.abs(fc).max() per subject -> matches the paper.
Lines 34-36 record the v2 -> v3 change: node normalization was switched FROM joint (B)
TO per-band (C).
No further input-feature normalization exists downstream: batch.x is passed straight into
the model. --normalize_nodes (default True in argparse, False in the paper-exact profile)
L2-normalizes node EMBEDDINGS inside the GIN, not the input ALFF.
unsupervised/embedding_evaluation.py wraps StandardScaler INSIDE a sklearn Pipeline that is
fit only on train_emb (lines 65-70, 182-186), with an explicit comment that placing it
outside would leak validation-fold statistics — correct practice, on embeddings not ALFF.

### (b) THE PAPER — Zhang et al. 2023, Sec. 2.1  (verbatim)
"The node features are normalized to [0,1] by subtracting the minimum from all the 3
channels and dividing the result by the difference between the maximum and the minimum.
The edge weights are normalized to [-1,1] by dividing each weight by the maximum of the
absolute values."
=> "the minimum from all the 3 channels" = ONE min/max over the whole 90x3 matrix
   = transformation B (SUBJ-JOINT MIN-MAX). Per-subject. Says NOTHING about train-only
   fitting or leakage control. Edge rule matches the loader exactly.
=> the CURRENT LOADER (C) DIFFERS from the PAPER (B). The repo documents this deliberately.

### (c) THE ORIGINAL RELEASED CODE — recovered from git history (commit bed5441)
    x = nf['alff_value_cache']
    x = np.nan_to_num(x)
    x = torch.Tensor(x)
and for edges: edge_index = adj['corr_each_sub']; nan_to_num; coo_matrix -> edge_weight.
=> the original released code performs NO normalization of node features and NO
   normalization of edge weights. It therefore does not implement its own paper's Sec 2.1.
   Whatever normalization existed was baked into the unreleased alff_value_cache .mat
   files -> the original's actual node-feature normalization is UNKNOWN.

### (d) A FOURTH ALFF PRODUCT IS WHAT THE PIPELINE ACTUALLY CONSUMES  (new finding)
The loader reads data/raw/*_NF/*.mat key norm_matrix — NOT M1, M2 or OLD.
norm_matrix over the 954 cohort is ALREADY per-subject per-band Z-SCORED:
  954/954 subjects have per-band mean = 0 and sd = 1 for all three bands
  (global min -3.5994, max 5.6307, mean -0.000000, std 1.000000)
Its relation to the audited sources: pearson vs OLD alff +0.5911, vs OLD malff +0.7364,
vs M1 +0.3516; within-subject-band spearman vs OLD alff mean 0.6969, min 0.4554.
So the ACTIVE pipeline applies D upstream and then C in the loader. PROVEN ALGEBRAIC
CONSEQUENCE (verified numerically):
  max| C(D(X)) - C(X) | = 5.55e-16 (M1), 4.44e-16 (OLD)  -> IDENTICAL
because min-max is invariant to any prior positive per-subject-per-band affine map.
  => the upstream z-score is mathematically ERASED by the loader's per-band min-max.
     The double normalization is harmless, but the z-score step is redundant.
  CONTRAST: max| B(D(X)) - B(X) | = 6.31e-01 (M1), 6.21e-01 (OLD) -> NOT identical.
     If the paper's JOINT normalization B were used, the upstream z-score would materially
     change the result by pre-equalizing the three bands. This is exactly the interaction
     the loader's own comment describes, and it means B and the existing norm_matrix
     upstream pipeline are NOT independent choices.
Idempotence checks: C(C(X)) = C(X) exactly; D(D(X)) = D(X) to 4.4e-15.

## NORMALIZATIONS THAT SHOULD BE REJECTED (mathematical / leakage grounds only)
REJECT 1 — ANY dataset-global min-max or z-score fitted on all 954 subjects.
  Grounds: test-set leakage, demonstrated in §11 (iii): fitting on all 954 makes test
  values exactly bounded in [0,1] in 5/5 folds, whereas honest train-only fitting lets
  15 test values escape. The all-954 parameters differ from train-fold parameters by up
  to 20.57 in band max. This is NOT what the current loader does (it is per-subject), but
  it is the naive default and must be ruled out explicitly before any predictive stage.
REJECT 2 — E (train-fold per-band MIN-MAX), on robustness grounds, not leakage.
  Grounds: the transform is determined by exactly 2 of 85,860 training values. Removing
  one subject from the training fold changed the slow-4 band max from 48.91 to 28.34
  (delta 20.57, a 42% change); fold-to-fold parameter spread was 15.74 / 20.57 / 17.06
  across bands. A normalization whose scale is set by a single extreme subject is not a
  stable estimator, and S3A already showed 0.29-0.65% of values are >3xIQR outliers.
  F (train-fold z-score) is the stable counterpart: sd delta <= 0.0678 across folds.
REJECT 3 — applying B (the paper's joint min-max) on top of the existing norm_matrix.
  Grounds: proven above, B does NOT commute with the upstream per-band z-score
  (delta 0.63). Using B while the upstream z-score remains in place would silently
  implement neither the paper's transform nor a clean one. B is only meaningful if
  applied to genuinely raw per-band ALFF (M1/M2/OLD), not to norm_matrix.
NOT REJECTED, but flagged:
  - C (current loader): leakage-free and well-defined, but discards both between-subject
    and between-band scale and breaks cross-band ROI ordering (Spearman down to 0.569).
  - D: leakage-free, maximal site-confound removal (eta2 0.424 -> 0.014 for M2), but
    annihilates ALL between-subject and between-band information (share = exactly 0).
  - The loader's span > 0 guard is the only code-level degeneracy protection, and on a
    zero-span band it passes x through UNNORMALIZED (torch.where returns raw x), which
    would mix raw and [0,1] scales inside one batch. No such subject exists in this
    cohort (min span 3.19), so it is latent, not active.

## UNRESOLVED / CARRIED FORWARD
1. norm_matrix (what the pipeline actually consumes) is a FOURTH ALFF product whose
   provenance was never established in S1/S2/S3A. It correlates only 0.59 with OLD alff.
   Until its generator is found, the audited M1/M2/OLD chain does not describe the data
   the model is currently trained on.
2. The adversarial cross-check workflow I launched for (a)/(b)/(c) FAILED — all three
   agents died on transient API 529 errors and returned nothing; a retry is running.
   The provenance findings above are therefore my own direct file reading, verified by
   quoting the exact lines, but not yet independently re-checked by a second reviewer.
3. Choice among B / C / D / F is NOT made here: it requires the diagnosis-aware comparison
   that belongs to S3C, plus a decision on whether between-subject scale is signal or
   site confound (§10 shows it is substantially site for M2).

S3B STATUS: EVIDENCE COMPLETE

================================================================================
# S3B ADDENDUM — ADVERSARIAL SELF-VERIFICATION OF THE PROVENANCE CLAIMS
The subagent cross-check workflow failed twice (all 3 agents, API 529 Overloaded, zero
results returned both runs). I therefore performed the refutation pass myself, targeting
the same failure modes the agents were instructed to hunt for.

## CHECK 1 — are the quoted lines actually at the cited locations?
sed -n '34,36p;71,74p;85,87p' datasets/abideDataset.py reproduces all three quoted blocks
verbatim and in order. CONFIRMED, no paraphrase, no line drift.

## CHECK 2 — is dim=0 really the ROI axis? (the scope-error trap)
Lines 57-59:  x = nf['norm_matrix'] ; x = np.nan_to_num(x) ; x = torch.Tensor(x)
norm_matrix is (90, 3) (S1, verified for all 954). There is NO transpose/permute/reshape
between load and the min-max. Therefore x.min(dim=0) reduces over the 90 ROIs and yields
shape (1,3) = one min per BAND, inside the per-subject loop.
CONFIRMED: per-subject, per-band. NOT a global reduction. Claim C stands.

## CHECK 3 — was any normalization MISSED elsewhere in the loader?
Exhaustive grep of the whole file for min|max|mean|std|norm|scal|/|*|log|sqrt|clip|clamp|
transpose|permute returns only lines 71-74 (node features) and 85-87 (edge weights) as
transforms. CONFIRMED: those are the only two.

## CHECK 4 — the access-time transform (a genuine gap in my first pass)
I had not checked `transform=my_transforms`, which PyG applies on every __getitem__.
agcl_ABIDE.py:58 and agcl_ABIDE_queue.py:183:
    my_transforms = Compose([set_tu_dataset_y_shape])
It reshapes the LABEL y only; it does not touch x. pre_transform/pre_filter are never
passed (default None), so lines 106-109 are inert.
CONFIRMED: no additional input-feature normalization. Original claim survives, but it was
under-evidenced before this check.

## CHECK 5 — a global min-max DOES exist in the original tree (found, then dismissed)
Original A-GCL/unsupervised/embedding_evaluation.py:38-39:
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
np.min(data, 0) reduces over SAMPLES, i.e. a dataset-global per-feature min-max — which
would be a leakage path if it sat in the evaluation path. It does NOT: it is the body of
    def plot_embedding(data, label, title)
a t-SNE/matplotlib visualization helper that builds a figure and returns it. It is not
called from the training or scoring path.
VERDICT: not a leakage path, but it is the closest thing to one in the original code and
is recorded here so it is not re-discovered later and misread as a pipeline transform.

## CHECK 6 — embedding-level normalization in the original (adjacent, not input)
Original A-GCL/unsupervised/encoder/tu_encoder.py:68 has an ACTIVE (uncommented)
    x = F.normalize(x, dim=1)
applied to node embeddings after conv -> BatchNorm -> dropout, before pooling. The encoder
also applies self.bns[i](x) to hidden activations. Neither touches the input ALFF.
This is the direct ancestor of the current --normalize_nodes flag (argparse default True;
set to False by the paper-exact profile in unsupervised/training_profiles.py:23).
Recorded to keep the input/embedding distinction explicit.

## NET RESULT OF THE SELF-VERIFICATION
All three provenance verdicts SURVIVE:
  current loader  = C (per-subject per-band min-max), leakage-free, baked into the v3 cache
  paper           = B (per-subject joint min-max across all 3 channels)
  original code   = NO node-feature and NO edge-weight normalization; the real transform
                    was baked into the unreleased alff_value_cache -> UNKNOWN
Two under-evidenced points are now closed (CHECK 4, CHECK 2) and two adjacent
normalizations are now explicitly bounded as non-input (CHECK 5, CHECK 6).
RESIDUAL LIMITATION: this remains single-reviewer verification. It is my own refutation
pass against my own claims, not an independent second reader, because subagent capacity
returned 529 on both attempts.
