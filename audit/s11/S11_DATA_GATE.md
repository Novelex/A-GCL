# S11 DATA GATE — ALL PASS
2026-08-20 | git HEAD f6c49d49761ef3bf82b3d59798ee1e8ace6b62dc | production untouched
Authoritative manifest: audit/s11/manifest/subject_manifest.csv (sha 4f01b0ab...), exact
FILE_ID lookup only — no substring matching, no glob, no fallback path anywhere in the loader.

| check | result | detail |
|---|---|---|
| cohort | PASS | N=954, ASD=455, NC=499, ROI=90; 954 unique FILE_IDs and paths |
| exclusions | PASS | CMU_b_0050669 and Leuven_1_0050706 absent |
| labels | PASS | all 954 equal the frozen S1 mapping (1=ASD/2=NC upstream preserved) |
| subject order | PASS* | rows in the SPLIT-DEFINING S3C order; see finding below |
| frozen hashes | PASS | dataset 312266b2..., splits 28fed44d..., ROI manifest a7632cd9... |
| historical cache exclusion | PASS | frozen 954 cache only; NOT data_dense_v3.pt, NOT any 956 cache, NOT norm_matrix |
| FC source validity (all 954) | PASS | shape/finite; sym max 2.2e-16; |diag-1| max 2.2e-16; signed (min -0.760) |
| FC source vs graph round-trip (all 954) | PASS | max_abs 2.98e-08 (float32 rounding), 0 mismatching subjects, 0 entries |
| ALFF x verification (all 954) | PASS | x == frozen M1_B, max_abs 0.00e+00; no M2/OLD fallback possible |
| path audit | PASS | only data/raw/ASD_ADJ and data/raw/NC_ADJ ever opened |
| split membership | PASS | frozen folds cover exactly rows 0..953, disjoint |
| 4005 pair-map | PASS | k=1 upper triangle, ROI names attached, sha aba8e09f... |
| X_fc vs frozen S5.5 representation | PASS | max_abs 0.00e+00 (BITWISE) |
| X_fc_source vs X_fc_graph | PASS | max_abs 2.98e-08 over all 954x4005 |
| raw-FC baseline reproduction | PASS | ord 0.7565 (d=-1.6e-5), LOSO 0.7432 (d=-3.5e-5) |
X_fc sha256: 5e0780c9d99b238a... (asserted by every downstream worker before computing).

## *GATE FINDING — ordering discrepancy caught and corrected (first gate run FAILED by design)
The S5 graph-cache subject order (ASD block then NC block) is a PERMUTATION of the
split-defining S3C order (sorted cohort). The first gate run built X_fc in S5 order and the
S5.5-representation check failed at max_abs 1.47 — the gate STOPPED, as designed.
Consequence documented for the record: S8-S10 probes applied S3C fold indices to S5-ordered
arrays. Labels remained correctly paired with features throughout (internally valid,
leakage-safe CV), but fold ASD counts drifted to 79-101 instead of the frozen 91 per fold.
This does not alter any S8-S10 conclusion (all end-to-end results were at chance and the
node-level estimates match S7.5, which used the correct order), but those stages' claim of
"exact frozen splits" was inexact. S11 rows are in the S3C order; the S5 permutation is
recorded per subject in the manifest (s5_cache_index column).
