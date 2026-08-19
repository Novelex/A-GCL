# S5 — FINAL M1 + FC GRAPH CONSTRUCTION (evidence only)
2026-08-18 | baseline commit 8cac2358ff12bcfa7452c38c4f4ef5e058814289 | tree CLEAN
No A-GCL training, no ComBat, no GNN. data_dense_v3.pt NOT reused and NOT overwritten.
norm_matrix NOT used. No subject excluded beyond the frozen S0 exclusions.

## INGREDIENTS (all hash-verified before use)
frozen raw M1 ROI-first ALFF   X_sources.npz sha256
  dc10bf36c4124aa7f214ab6bbc5a89288adc03da747e3950485bb15c0da333a9  (asserted at build time)
frozen FC                      data/raw/{ASD,NC}_ADJ/*_adj.mat, key cropped_matrix
954 subjects, 455 ASD / 499 NC, 90 AAL ROIs (frozen S1 axis), 3 bands slow5/slow4/classical

## SUBJECT ORDERING AND IDENTITY
Ordering deliberately matches datasets/abideDataset.py process() so the caches are
drop-in comparable: ASD block (sorted by *_adj.mat filename) then NC block (sorted).
  455 ASD then 499 NC = 954, all unique, every ID present in the frozen cohort.
  subject_id = 0..953 in that order (identical convention to the repo loader).
  mapping table: s5/subject_id_map.csv with columns
     subject_id, FILE_ID, y, dx_storage, DX_GROUP_upstream
  Label convention: PyG y = 1 for ASD, 0 for NC (loader convention), while the documented
  upstream ABIDE coding DX_GROUP 1=ASD / 2=NC is preserved verbatim in DX_GROUP_upstream.
  subject ordering sha256      71ccbdab06227533ccd9f8c98617d3ae47e05fd68963000d1b542d73f47fa37e
  subject_id->FILE_ID sha256   64fcc09f1e7e3944cf9aa8faebcb168ae2178410ba1be749bc069b04c39921ec

## THREE BRANCHES, EACH BUILT DIRECTLY FROM FROZEN RAW M1 (never chained)
  B  paper joint    x = (X - min_{all 90x3} X) / (max_{all 90x3} X - min_{all 90x3} X)
  C  per-band       x = (X - min_r X[:,b]) / (max_r X[:,b] - min_r X[:,b])   per band b
  D  per-band z     x = (X - mean_r X[:,b]) / std_r X[:,b]                    per band b
D-then-C was NOT applied anywhere; each branch calls norm(RAW, k) on the raw tensor.

## VERIFICATION — ALL 954 GRAPHS, ALL THREE BRANCHES
                                                        M1_B      M1_C      M1_D
 1  id/diagnosis mismatches vs frozen cohort               0         0         0
 2  x shape != (90,3)                                      0         0         0
    non-finite x                                           0         0         0
 3  round-trip raw->expected vs cached x, mismatches       0         0         0
    round-trip max_abs_error                          0.0e+00   0.0e+00   0.0e+00
 4  B implements paper JOINT [0,1] — violations            0         -         -
    observed global range                        [0.000000,1.000000]
 5  C implements PER-BAND [0,1] — violations               -         0         -
    observed global range                                  [0.000000,1.000000]
 6  D per-band mean~0 / std~1 — violations                 -         -         0
    mean range [-7.09e-08, +8.94e-08]; std range [0.99999970, 1.00000024]
 7  ROI/band ordering changes (Spearman<1 vs raw,
    2862 subject-band checks per branch)                   0         0         0
 8  FC != frozen S4 matrix — mismatches                    0         0         0
    FC max_abs_error                                  0.0e+00   0.0e+00   0.0e+00
 9  edge_index != canonical 8100 layout                    0         0         0
10  edge_weight length != 8100                             0         0         0
11  labels outside {0,1}                                   0         0         0
    ASD->1 / NC->0 mismatches                              0         0         0
12  subject_id mismatches                                  0         0         0
Round-trip error is identically zero (not merely small): the cached tensors are the
bit-for-bit float32 result of normalising the frozen raw M1.

## EDGE STRUCTURE (item 9 detail) — canonical, shared by all three branches
  total directed entries        8100 == 90 x 90                      True
  self-loops (u == v)           90                                    True
  all ordered pairs exactly once                                      True
  reverse edge present for every edge                                 True
  edge_weight order == FC.reshape(-1) (row-major)                     True (item 8, err 0)
  edge payload identical across B/C/D                                 True
  total edges per branch        7,727,400 == 954 x 8100               True
Edge rule applied is the loader's own fc / max(abs(fc)); S4 proved max|FC| = 1.0 exactly
for all 954, so this is a verified no-op and the stored weights equal the raw signed PCC.

## CACHE ISOLATION PROOF (item 13)
(a) branch roots live OUTSIDE the repo and hold only their own file:
      s5/M1_B/processed -> ['M1_B_v1.pt']
      s5/M1_C/processed -> ['M1_C_v1.pt']
      s5/M1_D/processed -> ['M1_D_v1.pt']
(b) repo data/processed still holds exactly its original five files.
(c) data_dense_v3.pt sha256 = a63db36dec759f2ffe3c1ebbe0aaf13d44470a5d9639fb39ab3ddcb41ffc5969
    identical to the S0 manifest value -> UNCHANGED, not overwritten.
(d) zero files named data.pt / data_dense_v2.pt / data_dense_v3.pt exist under any branch root.
(e) PyG resolves a cache by root + processed_file_names. The branches use distinct roots
    AND distinct filenames, so none can resolve to the repo cache or to each other.
(f) content-level proof: none of the branches matches v3. ASD-block comparison
      M1_B vs v3 max_abs_diff 1.0000 corr +0.3859
      M1_C vs v3 max_abs_diff 1.0000 corr +0.4299
      M1_D vs v3 max_abs_diff 6.0547 corr +0.4303
(g) branches mutually distinct: B/C max_abs_diff 0.6798, B/D and C/D 6.1365.
(h) raw inputs untouched: sha256sum -c over all 2868 raw files -> exit 0.

## NEW FINDING — THE ACTIVE v3 CACHE HOLDS 956 SUBJECTS, NOT 954
data_dense_v3.pt contains 956 graphs (x rows 86040 = 956 x 90), y distribution
ASD = 455, NC = 501. Our frozen cohort is 954 (455 / 499).
The two extra graphs are the S0-excluded subjects CMU_b_0050669 and Leuven_1_0050706
(both NC, which is exactly why v3 has 501 NC). This follows from the loader iterating over
every *_adj.mat in the ADJ folders (956) and pairing with norm_matrix, which exists for all
956 — the ALFF-side exclusion never propagated to the graph dataset.
Consequence: any previously reported A-GCL result computed from data_dense_v3.pt was
trained on 956 subjects, a different cohort from the one S1-S3C audited. Recorded as fact;
no action taken, no cache modified.
It also gives a second, independent isolation guarantee: the graph counts differ (956 vs
954), so a stale cache cannot be silently substituted for a branch without detection.

## HASHES
  subject ordering (954 ids)   71ccbdab06227533ccd9f8c98617d3ae47e05fd68963000d1b542d73f47fa37e
  subject_id -> FILE_ID map    64fcc09f1e7e3944cf9aa8faebcb168ae2178410ba1be749bc069b04c39921ec
  frozen raw M1 (X_sources)    dc10bf36c4124aa7f214ab6bbc5a89288adc03da747e3950485bb15c0da333a9
  M1_B  312266b23ecf1348ce083cb25d9c5e5a51d5595dab9ce5639875a51c12f1f844  155,637,176 B  n=954
  M1_C  17338f14073b82f6793beb81c93314a1f94e35f58e92b32488ecba8ff59e0c9d  155,637,176 B  n=954
  M1_D  59a4c88a2c3797882727fd1c5cd323fdb6608e808641ac2025cbdc64075fd397  155,637,176 B  n=954
  repo data_dense_v3.pt (untouched) a63db36dec759f2ffe3c1ebbe0aaf13d44470a5d9639fb39ab3ddcb41ffc5969

## MISMATCHES
None. Every one of the 13 required checks returned 0 mismatches on all 954 graphs in all
three branches, with round-trip and FC errors identically 0.0.

## GIT STATE
HEAD 8cac2358ff12bcfa7452c38c4f4ef5e058814289, branch main, working tree CLEAN,
in sync with origin/main (github.com/Novelex/A-GCL). S5 wrote nothing inside the repo:
all three caches and all evidence live in /users/3171356m/agcl_audit_s0/s5/ (446 MB).
Raw data and the repo processed cache re-verified unchanged.

## UNRESOLVED / CARRIED FORWARD
1. The 956-vs-954 discrepancy in the active cache (above) is the most consequential item;
   it means the repo's current graph dataset is not the audited cohort.
2. Branch selection among B / C / D is NOT made here, as instructed.
3. The dynamic-window FC (*_DW, 3x(90,90)) is not part of these graphs; only the static
   cropped_matrix is used, matching the loader. Its values remain unaudited (S4 item 4).
4. Node features and edges still derive from different C-PAC strategies (M1 nofilt_noglobal,
   FC filt_noglobal) — carried forward from S2/S4, not resolved here.
5. These caches were written with PyG 2.6.1 / torch 2.5.0+cu121; the (data, slices) format
   matches the repo loader's torch.save(self.collate(...)) convention, but they were NOT
   produced by the repo loader class itself, so a future consumer must point at the branch
   root with the matching processed_file_names.

S5 STATUS: EVIDENCE COMPLETE
