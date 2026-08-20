# FUTURE FEASIBILITY NOTE — 116-ROI (AAL1 full) BRANCH
STATUS: NOT IMPLEMENTED. Written in S7.5 as a plan only, per instruction.
The primary experiment remains FROZEN at 954 subjects / AAL90.
Nothing in this note was executed. No 116-ROI data, graph or cache was created.

## 1. The 26 additional AAL regions (indices 91-116)
Cerebellum (18): Cerebelum_Crus1_L/R, Cerebelum_Crus2_L/R, Cerebelum_3_L/R,
  Cerebelum_4_5_L/R, Cerebelum_6_L/R, Cerebelum_7b_L/R, Cerebelum_8_L/R,
  Cerebelum_9_L/R, Cerebelum_10_L/R
Vermis (8): Vermis_1_2, Vermis_3, Vermis_4_5, Vermis_6, Vermis_7, Vermis_8,
  Vermis_9, Vermis_10
In the C-PAC rois_aal header these are exactly the codes >= 9001 (9001..9170).

## 2. Which existing raw sources ALREADY contain them
  data/ALFF_need/rois_aal/*.1D          116 columns per subject, all 956 subjects.
                                        The 26 cerebellar/vermis columns ARE PRESENT and
                                        were discarded only by the < 9001 cutoff.
  aal_mask_pad.nii.gz (C-PAC atlas)     all 116 labels present.
  AAL_61x73x61_YCG.nii (DPABI atlas)    labels 1..116 present.
  func_preproc 4D NIfTI                 whole-brain, includes cerebellum coverage
                                        (subject to field-of-view, see risk 3).

## 3. What the FROZEN sources currently LACK
  M1 alff_roi_first.npz / M2 alff_voxel_first.npz   [954, 90, 3] — generated with
        N_ROIS=90 and roi_masks over range(1,91). The 26 regions were never computed.
  OLD alff_new.npz                                   [956, 90, 3] — codes >= 9001 dropped
        by scripts/compute_alff.py (CEREBELLUM_VERMIS_THRESHOLD = 9001).
  FC cropped_matrix                                  90x90 — built from the same < 9001 mask.
  S5 graph caches M1_B / M1_C / M1_D                 90 nodes, 8100 edges.

## 4. What would need regeneration (in dependency order)
  a. FC 116x116: cheap. Re-run the notebook loader path WITHOUT the 9001 cutoff on the
     same filt_noglobal .1D files. No new download. 954 x 116x116.
  b. OLD-route ALFF 116: cheap. Same .1D files, drop the cutoff in compute_alff.py's
     keep-mask. Formula already frozen and validated (S3A).
  c. M1/M2 ALFF 116: EXPENSIVE. Requires re-running dual_alff_recompute.py over all
     func_preproc NIfTIs with roi_masks over range(1,117), including its zero-valid-voxel
     guard. This is the step that dominates cost.
  d. Atlas validation for 91-116: confirm every cerebellar/vermis label is non-empty in
     the YCG atlas AND has valid functional coverage per subject.
  e. Cohort re-alignment: the zero-valid-voxel guard currently excludes 2 subjects on
     AAL90 (CMU_b_0050669 ROI 87, Leuven_1_0050706 ROI 28). Cerebellar coverage is
     systematically worse at the bottom of the FOV, so the exclusion set WILL change and
     may grow substantially. The 954 cohort would no longer be the cohort.
  f. Normalization audit: B/C/D must be recomputed on the 116 axis; per-band min-max and
     z-score statistics change when 26 new regions enter.
  g. Graph reconstruction: new caches with 116 nodes and 116*116 = 13,456 directed edges.
  h. Full re-audit of S1 ROI alignment, S4 FC validity and S5 construction on the new axis.

## 5. Estimated compute / storage (order-of-magnitude)
  FC 116 rebuild                ~10 CPU-min total (reads .1D, corrcoef)
  OLD ALFF 116 rebuild          ~5 CPU-min
  M1/M2 116 recompute           ~15-20 CPU-hours (S3A's full-cohort recompute was
                                 ~24 CPU-h for 90 ROIs across 32 tasks; 116 scales it up)
  graph caches (3 branches)     ~260 MB each at 116 nodes (vs 156 MB at 90) -> ~0.8 GB
  full re-audit S1/S4/S5        ~5-10 CPU-hours
  TOTAL                         roughly 30-40 CPU-hours plus ~1 GB storage.
  No GPU required.

## 6. Scientific reason it may be wanted
  The paper's headline AAL1 configuration uses 116 ROIs (cerebrum + cerebellum + vermis),
  and correction.md records that the paper's 80.65% number came from that configuration.
  Our frozen 90-ROI axis is a defensible but DIFFERENT atlas configuration and is not
  directly comparable to the paper's Table 2. A 116-ROI branch would exist solely as a
  PAPER-REPRODUCTION branch, run alongside — never replacing — the frozen 90-ROI primary.

## 7. Principal risks
  1. The exclusion set changes (item 4e), so results would not be on the frozen 954 cohort.
  2. Cerebellar FOV truncation is common in ABIDE; coverage must be audited per subject
     before any ALFF value is trusted.
  3. Every downstream frozen artefact (S1 alignment hashes, S4 FC validity, S5 caches,
     S3C splits) is 90-ROI specific and would need regeneration and re-freezing.
  This is a NEW SCIENTIFIC DATA BRANCH, not a one-line filter change.
