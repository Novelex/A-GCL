# S1 — 954-SUBJECT ID + DIAGNOSIS + ROI ALIGNMENT AUDIT (evidence only)
2026-08-18 | HEAD 906a494b076968768573a24c31804c6b0b1dd65b | tree: ?? ALFF_func_proc/ (unchanged from S0 freeze)
No ALFF statistical comparison, no normalization, no ComBat, no training. No source data/code altered.

## AUDIT TABLE
/users/3171356m/agcl_audit_s0/s1_audit_table.csv — 954 rows x 22 cols
columns: subject_id, m1_row, m2_row, old_row, old_subset_row, fc_adj_file, fc_index,
dx_storage, dx_subject_tr, dx_manifest, dx_phenotypic, dx_alff_npz, dsm_iv_tr,
site_tr, site_combat, site_pheno, site_prefix, n_roi_m1, n_roi_m2, n_roi_old, n_roi_fc,
mismatch_sources

## 1. EXACT 954 ID HASH
sorted 954 ids, newline-joined, SHA-256:
  aca3d945f7d89ccb1cc6fb46ca01f6036ccf036d81d120d2b09fe4bd0d1d68c9
A-order 954 ids SHA-256: identical (A is already sorted)
|A|=954 |B|=954 |A∩B|=954, all unique.  Checks 1,3: PASS-shaped evidence.

## 2. ASD/NC COUNTS (frozen storage membership)
ASD 455 / NC 499 = 954   (target met)
ASD∩NC storage overlap = 0 (check 4)
duplicate ids: A=0 B=0 OLD=0 FC=0 (check 3)

## 5. SOURCE ID MISMATCH COUNT = 0
A ids == B ids                              : True
cohort ⊆ FC(956)                            : True
OLD ⊇ cohort, OLD∖cohort = the 2 excluded   : True
OLD-subset(954) == cohort                   : True
=> method1 / method2 / OLD-subset / FC all carry exactly the same 954 ids. 0 mismatches.

## 6. ROW-ORDER FINDINGS
A order == B order                          : True
A order == sorted(cohort)                   : True  (both sources are sorted)
OLD full order is sorted                    : True
OLD order restricted to the 954             : True — identical sequence to A
alff_new_combat.npz order == OLD order      : True
FC directory listing (sorted) on cohort     : True
IMPORTANT: OLD *absolute row indices* are NOT equal to A/B row indices. The two excluded
subjects sit inside the sorted sequence, so indices shift. First divergence at A position 20
(Caltech_0051456 -> OLD index 21). Any OLD subsetting MUST be done by ID, never by position.
Both index columns are carried in the audit table (old_row vs old_subset_row).

## 7. DIAGNOSIS CODING SCHEMES (derived from storage evidence, nothing assumed)
crosstabs computed over all 956 storage-labelled subjects:
  subject_tr.csv DX_GROUP        : 1 -> ASD (455/455 pure) ; 2 -> NC (501/501 pure)
  download_manifest.csv label    : "ASD" -> ASD (455 pure) ; "NC" -> NC (501 pure)
  phenotypic_filtered_v2 DX_GROUP: 1 -> ASD (455 pure)     ; 2 -> NC (501 pure)
  alff_new.npz dx_group          : 1 -> ASD (455 pure)     ; 2 -> NC (501 pure)
=> a single consistent binary scheme across every numeric source: 1=ASD, 2=NC
   (ABIDE convention), plus one string scheme ASD/NC. No 0/1 and no ASD/HC scheme is
   present anywhere in this repository's metadata.
NOT a diagnosis field — documented explicitly:
  phenotypic DSM_IV_TR: values {-9999,0,1,2,3,4}; NOT pure w.r.t. storage
    (0 -> 481 NC + 18 ASD; -9999 -> 23 ASD + 20 NC; 1/2/3/4 -> ASD only).
    This is an ABIDE DSM-IV-TR subtype/missing field, not a binary label. It is carried in
    the audit table for completeness and MUST NOT be used as a diagnosis.

## 8. DIAGNOSIS MISMATCH COUNT = 0
Per-source mismatches vs frozen storage over the 954 cohort: ZERO in all four sources.
Subjects with ANY mismatch: 0 / 954.  Required target met.
Site cross-check: site_tr == site_pheno for 954/954.
  Granularity note (not a mismatch): site_tr / site_combat / site_pheno use 19 ABIDE
  SITE_ID values; the filename prefix gives 23 (MaxMun_a/b/c/d -> MAX_MUN, Leuven_1/2,
  UCLA_1/2, UM_1/2, CMU_a/b collapse). Both are recorded per subject.

## 9. ROI-ORDER FINDINGS (actual labels, not just shape==90)
method1 / method2 (ALFF_func_proc):
  atlas = DPABI Templates/AAL_61x73x61_YCG.nii (integer labels 0..116).
  generator layer_testing/dual_alff_recompute.py: N_ROIS=90, asserts labels 1..90 all
  present, builds roi_masks for label in range(1,91) -> ROI axis = AAL index 1..90 ascending.
  run log: "90 ROI masks loaded, all validated non-empty".
  authoritative names from Templates/aal_Labels.mat (117 rows):
    idx1=Precentral_L, idx2=Precentral_R, idx29=Insula_L, idx30=Insula_R,
    idx89=Temporal_Inf_L, idx90=Temporal_Inf_R, idx91=Cerebelum_Crus1_L (first excluded).
OLD (alff_new.npz):
  source rois_aal.1D carries an explicit AAL-code header, byte-identical across all 956
  files (distinct header lines = 1). 116 columns.
  scripts/compute_alff.py keeps labels < 9001 -> exactly 90 columns, and those are exactly
  columns 1..90 (0 codes >=9001 inside the first 90; 0 codes <9001 after column 90).
  header codes: col1=2001, col2=2002, col29=3001, col30=3002, col89=8301, col90=8302,
  col91=9001 (cerebellum begins).
CROSS-SOURCE ALIGNMENT (3 independent anchors, positional):
  pos 1/2   : code 2001/2002  vs  idx 1/2   = Precentral_L/R
  pos 29/30 : code 3001/3002  vs  idx 29/30 = Insula_L/R
  pos 89/90 : code 8301/8302  vs  idx 89/90 = Temporal_Inf_L/R
  boundary  : pos 91 = 9001   vs  idx 91    = Cerebelum_Crus1_L
  => method1, method2 and OLD share the same AAL90 ROI axis in the same order.
FC / ADJ (data/raw/*_ADJ/*.mat):  [RESOLVED — see addendum below]
  key `cropped_matrix`, shape (90,90) — uniform across all 954 (single key+shape class).
  node features `norm_matrix` (90,3) uniform across all 954.
  DW `correlation_matrices` = (3,1) MATLAB cell, each (90,90).
  GENERATOR FOUND: /users/3171356m/muhammad/GraSTIACL/notebooks/02_local_global_pcc.ipynb
    def load_rois_aal90(file_id):
        labels   = parse '#'-prefixed AAL codes from the .1D header line
        keep_mask= labels < AAL90_LABEL_CUTOFF (=9001)
        ts90     = ts_full[:, keep_mask]        # order-preserving boolean mask
        assert ts90.shape[1] == 90
        W = np.corrcoef(ts90, rowvar=False)     # -> 90x90 in that same ROI order
        savemat(out_dir/f"{file_id}_adj.mat", {"cropped_matrix": W})
    notebook markdown states: "Parsed directly per file rather than assumed from atlas ordering."
  => FC/ADJ, FC/DW and OLD ALFF are built from the SAME .1D files, with the SAME cutoff
     constant 9001, via an order-preserving column mask. FC ROI axis is therefore PROVEN
     identical to the OLD ROI axis, i.e. .1D columns 1..90 = AAL90 ascending, which the
     4 anchors above tie to method1/method2's atlas index 1..90.
     A-GCL's own scripts/compute_alff.py uses the identical constant
     (CEREBELLUM_VERMIS_THRESHOLD = 9001) and the identical rule.

## 10. EXCLUDED SUBJECTS (explicitly recorded)
  CMU_b_0050669    : in_cohort=False in_A=False in_B=False in_OLD=True in_FC=True storage=NC
  Leuven_1_0050706 : in_cohort=False in_A=False in_B=False in_OLD=True in_FC=True storage=NC
  cohort contains neither: True. Both are NC (hence 501 -> 499; ASD unchanged 455).
  S0-established cause (primary logs): ROI(s) with zero valid voxels — ROI 87 and ROI 28.

## CORRECTION TO THE FIRST S1 PASS
The initial pass reported "no FC build script found". That was read off an incomplete
background search. The completed search returned 9 files referencing `cropped_matrix`;
notebooks/02_local_global_pcc.ipynb is the generator. Ambiguity 1 is RESOLVED (above).

## NEW FINDING — DIAGNOSIS EVIDENCE IS NOT INDEPENDENT (circularity)
The same notebook routes output by diagnosis:
    out_dir = ASD_ADJ_DIR if dx_group == 1 else NC_ADJ_DIR      # global PCC
    out_dir = ASD_DW_DIR  if dx_group == 1 else NC_DW_DIR       # dynamic PCC
with dx_group read from phenotypic DX_GROUP. So "frozen storage membership" (which
directory a subject's .mat sits in) was DERIVED FROM DX_GROUP — it is not an independent
observation of diagnosis.
Consequences for check 8:
  - The zero-mismatch result between storage and DX_GROUP-derived fields
    (subject_tr.csv, phenotypic_filtered_v2, alff_new.npz dx_group) is partly TAUTOLOGICAL:
    it proves the routing was applied consistently and nothing was mis-filed or shuffled —
    a real and necessary integrity result — but it does NOT independently corroborate the
    diagnosis label itself.
  - All four sources ultimately trace to one upstream authority: the ABIDE phenotypic
    DX_GROUP column. download_manifest.csv's ASD/NC string is the only differently-typed
    encoding, and it too was generated from that column.
  - The mapping 1=ASD / 2=NC is now confirmed twice: derived empirically from the crosstabs
    AND stated explicitly in generator code (`if dx_group == 1` -> ASD path).
  - No external/independent diagnosis source exists in any searched tree to cross-validate.

## UNRESOLVED AMBIGUITIES (revised)
1. RESOLVED — FC/ADJ ROI order is proven identical to OLD (same .1D, same 9001 cutoff,
   order-preserving mask). No longer an assumption.
2. method1/method2 ROI axis alignment to OLD/FC is established positionally (atlas index n
   vs .1D column n) with 4 anchor points, not by an explicit AAL-code -> name lookup table;
   aal_Labels.mat indexes by integer 1..116 and carries no 2001-style codes.
3. Both new .npz files were generated outside A-GCL (GraSTIACL tree); no in-repo provenance.
4. DSM_IV_TR contains -9999 for 43 subjects (23 ASD, 20 NC) — missing-data sentinel,
   carried verbatim, not interpreted.
5. NEW: diagnosis labelling has a single upstream authority (ABIDE DX_GROUP) and storage
   membership is derived from it, so check 8 cannot be treated as independent
   corroboration of diagnosis correctness — only of internal consistency.
6. NEW: the notebook wrote to ../data/GraSTIACL_ABIDE_979/raw, not to A-GCL's data/raw.
   The .mat files were copied across projects; byte-level provenance of the copy step is
   not documented. Their key/shape/routing all match the generator's contract.

S1 STATUS: EVIDENCE COMPLETE

================================================================================
# S1 CLOSE-OUT — FULL 90-ROI ALIGNMENT (all positions, not anchors)
2026-08-18 | HEAD 906a494b… | tree ?? ALFF_func_proc/ | no data/code modified

## RESULT
90/90 positions aligned across M1, M2, OLD, FC.  MATCH=yes 90 · MATCH=no 0 · UNPROVEN 0
0 permutations · 0 missing ROIs · 0 duplicate ROI identities · identity order = AAL idx 1..90
Table: /users/3171356m/agcl_audit_s0/s1_roi_alignment_90.csv (90 rows x 10 cols:
position, m1_roi, m2_roi, old_roi, fc_roi, aal_code, aal_name, aal_index,
overlap_purity, match)

## HOW EACH SOURCE'S ORDER WAS ESTABLISHED (provenance, not inference)
M1, M2  : layer_testing/dual_alff_recompute.py — N_ROIS=90; asserts labels 1..90 present
          in AAL_61x73x61_YCG.nii; roi_masks = {l: atlas==l for l in range(1,91)};
          output ROI axis = YCG integer label ascending, index0..89 -> label 1..90.
          Run log: "90 ROI masks loaded, all validated non-empty".
OLD     : scripts/compute_alff.py — parses the '#' header of rois_aal.1D per file, keeps
          codes < 9001 (CEREBELLUM_VERMIS_THRESHOLD), order-preserving.
FC/DW   : notebooks/02_local_global_pcc.ipynb — same .1D, AAL90_LABEL_CUTOFF=9001,
          ts90 = ts_full[:, keep_mask] then np.corrcoef -> identical ROI axis to OLD.
.1D     : header byte-identical across all 956 files (1 distinct header line).
VERIFIED: sorted non-zero labels of the official C-PAC atlas aal_mask_pad.nii.gz
          (the atlas that generated rois_aal) == the .1D header code sequence, exactly
          (116/116, tested programmatically). Kept codes <9001 = exactly columns 1..90.

## PROVING YCG index 1..90  <->  AAL code 2001..8302 (the only real gap)
Neither aal_Labels.mat nor AAL3v1_1mm_Labels.mat carries the 2001-style codes, so the
mapping was DERIVED from the atlas files by three independent methods:
  M1 voxel overlap  : resample AAL_61x73x61_YCG onto the aal_mask_pad grid (nearest
                      neighbour, order=0); 54754 co-labelled voxels; dominant C-PAC code
                      per YCG index. Result: 116/116 mapped, bijective (0 codes claimed by
                      two indices), 0 unmapped.
  M2 positional     : YCG index i vs i-th ascending C-PAC code. 0 disagreements over 116.
  M3 centroid       : aal_Labels.mat centroid for index i vs C-PAC voxel centroid of the
                      mapped code, in MNI world coords. max 7.82 mm, mean 2.84 mm; the
                      nearest-centroid code equals the overlap-mapped code for all 116
                      (0 disagreements).
All three agree everywhere. Anchors confirmed by name: 1 Precentral_L, 29 Insula_L,
89/90 Temporal_Inf_L/R, 91 Cerebelum_Crus1_L (first dropped region).

## HONEST WEAKNESS (single-method only; does not change the verdict)
Voxel-overlap purity varies because the two atlas builds have slightly different
boundaries and were resampled. 17 of 90 positions have dominant-code purity < 0.80;
lowest: Heschl_L 0.500, Olfactory_L 0.529, Amygdala_R 0.671, Heschl_R 0.685,
Occipital_Sup_R 0.699, Frontal_Inf_Oper_R 0.705. These are small/thin regions where
3 mm resampling smears edges. At every one of them the positional and centroid methods
independently give the same answer, so no position rests on overlap alone. Recorded as
evidence quality, not as an unproven mapping.

## USING THE PROVEN MAPPING — the two excluded subjects
  CMU_b_0050669    ROI 87 zero valid voxels -> AAL idx 87 = Temporal_Pole_Mid_L
  Leuven_1_0050706 ROI 28 zero valid voxels -> AAL idx 28 = Rectus_R

## PRESERVED S1 FACTS (unchanged)
  common cohort = 954 ; sha256(sorted ids) = aca3d945f7d89ccb1cc6fb46ca01f6036ccf036d81d120d2b09fe4bd0d1d68c9
  ASD = 455, NC = 499
  DX_GROUP mapping = 1 ASD / 2 NC (empirical crosstabs + generator code `if dx_group == 1`)
  diagnosis mismatch = 0, but this is INTERNAL-CONSISTENCY evidence only, not independent
    validation: storage membership was itself produced by routing on DX_GROUP.
  excluded subjects = CMU_b_0050669, Leuven_1_0050706 (both NC, neither in the 954)

## TOOLING NOTE
nibabel is absent from the A-GCL .venv, so the atlas reads used /users/3171356m/miniconda3/
bin/python (nibabel 5.4.0), read-only. Nothing installed, no file written inside the repo
or data/. Atlas files opened read-only.

S1 STATUS: EVIDENCE COMPLETE
