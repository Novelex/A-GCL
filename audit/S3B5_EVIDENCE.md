# S3B.5 — norm_matrix FEATURE PROVENANCE (evidence only)
2026-08-18 | HEAD 906a494b… | tree ?? ALFF_func_proc/ | nothing modified, nothing deleted/rebuilt
Frozen 954 subjects / 90 ROIs / 3 bands. No classification, no training, no ComBat, no
choice of best ALFF.

## 1. norm_matrix SOURCE — FOUND AND CONFIRMED EXACTLY
Generator: /users/3171356m/muhammad/GraSTIACL/notebooks/03_alff_node_features.ipynb
  "Notebook 3: ALFF Node Features ... Output: {FILE_ID}_nf.mat, key norm_matrix,
   shape (90, 3), z-scored per subject per band, UNHARMONIZED."
Writer, cell 13:
    norm_matrix = norm_matrix_all[si].astype(np.float64)   # (90, 3): slow5, slow4, classical
    savemat(out_dir / f"{file_id}_nf.mat", {"norm_matrix": norm_matrix})
Written to  data/GraSTIACL_ABIDE_979/raw/{ASD,NC}_NF/  (DX_GROUP==1 -> ASD_NF else NC_NF),
then copied into A-GCL/data/raw/{ASD,NC}_NF/. Sibling of the FC notebook 02 (same Jul 26 night).
Grep for writers of the key across all trees returns this notebook only; every other hit
(A-GCL/datasets/abideDataset.py, GraSTIACL/datasets/Dataset.py, ml/run_baseline.py,
nested_cv/data.py, scripts/compare_alff.py, scripts/compute_alff.py, several docs) READS it.

## 2. FULL GENERATION CHAIN (each step proven from the notebook source)
raw functional
  -> DPARSFA (DPABI/DPARSF) pipeline, work dir data/dparsf_work/full_{band}_chunk{01..10},
     output Results/ALFF_FunImgD/mALFFMap_{FILE_ID}.nii
     "ALFF_FunImgD" is the DPARSF naming for ALFF computed on FunImg + Detrend
     (no F = no filtering suffix, no S = no smoothing, no C = no covariate regression).
     Run SEPARATELY PER BAND: three independent DPARSFA runs (full_slow5, full_slow4,
     full_classical), 10 chunks each.
  -> ALFF computed VOXELWISE by DPARSFA, and stored as mALFF (ALFF / global-mean ALFF).
     "mALFF is the primary feature (raw ALFF is not saved -- decided, not deferred)."
  -> FLIRT resample of each 3D mALFF map into the atlas grid (cell 7):
       flirt -in <mALFFMap> -ref aal_mask_pad.nii.gz -out <tmp>
             -applyxfm -init ident.mat -interp trilinear
     ident.mat is the literal 4x4 identity.
  -> ROI operation (cell 7): region_labels = sorted atlas labels != 0 and < 9001 (=90);
       malff_region_matrix[si,ri,bi] = resampled_data[region_mask].mean()
     "mean over ALL region voxels unconditionally" — including the zeros DPARSFA's brain
     mask writes at region boundaries. Coverage is computed but explicitly "diagnostic
     only (not used to filter the mean below)".
  -> 3 bands: BANDS = ["slow5","slow4","classical"] -> column order of norm_matrix.
  -> normalization BEFORE saving (cell 11):
       subject_band_mean = np.nanmean(malff_region_matrix, axis=1, keepdims=True)
       subject_band_std  = np.nanstd (malff_region_matrix, axis=1, keepdims=True)
       norm_matrix_all   = (malff_region_matrix - subject_band_mean) / subject_band_std
     matrix is (n_subjects, 90, n_bands), so axis=1 IS the 90-ROI axis
     => PER-SUBJECT, PER-BAND z-score. Matches the S3B empirical finding (954/954 have
        per-band mean 0, sd 1) exactly.
  -> savemat key norm_matrix.

## 3. THE ANSWERS
ROI-first or voxel-first : VOXEL-FIRST. ALFF is computed per voxel by DPARSFA, then
                           averaged over ROI. (Same ordering as M2, opposite to M1/OLD.)
atlas                    : aal_mask_pad.nii.gz (official C-PAC AAL, 2001-style codes),
                           shape (65,77,63) @ 3 mm — a PADDED variant.
which 90 ROIs            : sorted labels != 0 and < 9001 = the AAL90 cerebrum set, the SAME
                           90 and the SAME order as the frozen S1 axis.
preprocessing strategy   : DPARSF FunImgD (detrend only). NOT C-PAC. Different pipeline
                           from OLD (cpac nofilt_noglobal rois_aal), from M1/M2 (cpac
                           nofilt_noglobal func_preproc) and from FC (cpac filt_noglobal).
exact 3 frequency bands  : slow5 / slow4 / classical, as configured in the three separate
                           DPARSFA runs. The notebook does not restate the Hz edges; the
                           band definitions live in the DPARSFA run configuration, which
                           I did not locate -> the numeric Hz edges are UNKNOWN from the
                           notebook itself (see unresolved).
ALFF formula             : DPARSFA's own y_alff_falff.m (the same function S3A audited and
                           matched: detrend, 2^nextpow2 zero-pad, 2*abs(fft)/T, mean over
                           band bins), then mALFF = ALFF / global mean.
normalization before save: YES — per-subject per-band z-score (formula above).
z-score axis/formula     : axis=1 of (n_subjects, 90, n_bands) = across the 90 ROIs, using
                           np.nanmean / np.nanstd (population sd, ddof=0), per band.

## 4-5. NUMERICAL COMPARISON AND RECONSTRUCTION TESTS (all 954 x 90 x 3)
No audited source reproduces norm_matrix under any candidate transform:
  candidate                                   max_abs_err   pearson  spearman
  per-subj per-band z of M1                        6.6912    0.4697    0.4483
  per-subj per-band z of M2                        5.9083    0.5759    0.5449
  per-subj per-band z of OLD alff                  4.4527    0.7381    0.7162
  per-subj per-band z of OLD malff                 4.4527    0.7381    0.7162
  JOINT z of OLD alff                              5.1133    0.7088    0.6934
  JOINT z of M2                                    6.6292    0.5560    0.5268
  per-band MIN-MAX of OLD alff                     4.8444    0.7047    0.6883
  per-band MIN-MAX of M2                           5.2681    0.5198    0.4968
  raw OLD malff (no transform)                     3.9183    0.7364    0.7168
=> norm_matrix is a GENUINELY FOURTH ALFF CONSTRUCTION.
Algebraic note (verified): max|D(OLD malff) - D(OLD alff)| = 4.44e-15. Since mALFF divides
by a per-subject-per-band scalar, the per-band z-score ERASES the mALFF step entirely — so
norm_matrix's "mALFF" choice is INERT once the z-score is applied.

## 6. SUBJECT-BY-SUBJECT AND ROI-BY-ROI ALIGNMENT (954 x 90 x 3, NaN = 0)
  vs D(M1)  subject-level r: min 0.0540 median 0.4699 max 0.8210, n<0.3 = 81
            ROI-level  r: min 0.2067 median 0.5441 max 0.7750, n<0.3 = 3
  vs D(M2)  subject-level r: min 0.1017 median 0.5849 max 0.9418, n<0.3 = 20
            ROI-level  r: min 0.5887 median 0.8035 max 0.9328, n<0.3 = 0
  vs D(OLD) subject-level r: min 0.3413 median 0.7474 max 0.9396, n<0.3 = 0
            ROI-level  r: min 0.4330 median 0.6683 max 0.8496, n<0.3 = 0
Per band vs D(OLD): slow5 0.7432, slow4 0.7307, classical 0.7404.
CLOSEST RELATIONSHIP: D(OLD) overall and per subject (median 0.747); but D(M2) is closest
at ROI level (median 0.804), consistent with norm_matrix being voxel-first like M2.

## END-TO-END RECONSTRUCTION — CHAIN CONFIRMED EXACTLY (positive control passed)
Unlike S3A's DPABI case, a positive control was available here: apptainer and
fsl_6.0.7.4.sif are both present, so I ran the REAL FLIRT command from the notebook.
  CONTROL 1  Olin_0050102, classical band:
    stored norm_matrix[:,2] vs z-score(region means of real-FLIRT output)
      pearson = 1.000000, max_abs = 0.0000
  CONTROL 2  Trinity_0050246, slow5 band (different subject AND different band):
      pearson = 1.000000, max_abs = 4.44e-16
=> the reconstructed chain is EXACT. norm_matrix provenance is PROVEN, not inferred.
My earlier nibabel-based substitute reached only r = 0.47-0.76 (median 0.617); that gap is
itself the finding below.

## NEW FINDING — THE FLIRT STEP MISALIGNS THE NODE FEATURES BY 6 mm
Geometry (constant across 72 inspected mALFFMap files spanning 3 bands x 3 chunks —
1 distinct geometry, so this applies uniformly to all 954 x 3):
  DPARSF mALFF grid      (61,73,61) @ origin ( 90, -126, -72)
  AAL_61x73x61_YCG       (61,73,61) @ origin ( 90, -126, -72)   <- IDENTICAL to mALFF grid
  aal_mask_pad           (65,77,63) @ origin ( 96, -132, -71)   <- padded, affine records it
  origin difference (atlas - mALFF) = (6, -6, 1) mm
FLIRT was invoked with -applyxfm -init <identity>. FSL applies that transform in its own
scaled-voxel space, which ignores the differing world-space origins. Measured consequence
(Olin_0050102, classical): FLIRT output vs world-space (sform-aware) resampling of the same
input onto the same grid:
    zero-shift agreement            r = 0.3243
    best integer-voxel realignment  r = 0.9668  at (dx,dy,dz) = (-2, -2, 0) voxels
                                              = (-6, -6, 0) mm
i.e. the FLIRT result is displaced by EXACTLY the pad recorded in the atlas affine.
Which alignment is anatomically correct: world-space. S1 independently validated
aal_mask_pad's affine by mapping its 2001-style codes bijectively onto YCG indices 1..116
via voxel overlap after world-space resampling, agreeing with aal_Labels.mat centroids to
<= 7.82 mm with 0 disagreements over all 116 regions.
=> the node features the ACTIVE pipeline consumes appear to be sampled ~2 voxels (6 mm)
   off in x and y, so each AAL ROI's value is drawn from displaced tissue.
Note the DPARSF grid is IDENTICAL to AAL_61x73x61_YCG, so no resampling was needed at all —
using YCG directly would have avoided both the FLIRT step and the shift.
CORROBORATION from the generator's OWN QC file (data/raw/alff_nf_qc.csv, 956 rows):
  any_flag True for 956/956 subjects. n_low_coverage > 0 for 956/956, 18291 region-band
  flags total, median 18 and max 69 of 270 per subject. n_nonfinite 0, n_exact_zero 3
  subjects, n_outlier 72 subjects.
  A universal low-coverage flag is exactly what a 2-voxel displacement predicts: shifted
  brain edges fall outside atlas regions, so region masks pick up out-of-brain zeros —
  and because the mean is taken over ALL region voxels unconditionally, those regions are
  DILUTED TOWARD ZERO rather than excluded. The flag fired for every subject and was
  dismissed as "diagnostic only".
  Worst-flagged: Yale_0050605 (69), SDSU_0050204 (66), Caltech_0051462 (54),
  Leuven_2_0050736 (51), Caltech_0051456 (51). Yale_0050605 and Caltech_0051462 are the
  SAME subjects S3A independently surfaced (min 2 valid voxels; lowest M1-vs-OLD r 0.416)
  — a fourth independent arrival at the same coverage problem.
SCOPE: this affects the NODE FEATURES only. The FC/ADJ and DW matrices are unaffected —
S1/S2 proved they were built from rois_aal .1D header columns with no atlas resampling.
CALIBRATION: the geometry, the measured shift, and the exact chain reproduction are PROVEN.
The inference that world-space is the correct target rests on the S1 affine validation; an
independent anatomical check (e.g. overlaying a labelled template) would settle it beyond
doubt and has NOT been performed here.

## 7. PROCESSED PyG CACHE IDENTITY (not deleted, not rebuilt)
datasets/abideDataset.py:33-38  processed_file_names returns 'data_dense_v3.pt'  <- ACTIVE
  /users/3171356m/A-GCL/data/processed/data_dense_v3.pt
    155,963,532 bytes, mtime 2026-08-15 21:59
    sha256 a63db36dec759f2ffe3c1ebbe0aaf13d44470a5d9639fb39ab3ddcb41ffc5969
  This file contains norm_matrix-derived node features with the loader's per-subject
  per-band min-max ALREADY BAKED IN (S3B: normalization happens inside process()).
Stale, unused caches (left untouched):
  data_dense_v2.pt  155,963,532 B  mtime 2026-08-13 21:33
                    sha256 9940ffb63030bb7669823a2c91cf8a88b8eb43eda2fee7c21aeb7a6f763c803d
  data.pt           155,961,294 B  mtime 2026-08-13 16:14
                    sha256 1f673a108e024b630743dc46cd2a2d7e6f65ea216a6bccf4dfae78a2ea8f01ca
  pre_filter.pt / pre_transform.pt  864 B each
                    sha256 bed90159f346b7ee0fa80e1d287e0ebe46264111c1e928ef341d926aae6a075c
All five re-verified against the S0 manifest: sha256sum -c -> 5/5 OK, unchanged since S0.

## 8. IMPLICATION FOR USING M1/M2/OLD LATER (recommendation only, nothing changed)
YES — the cache must be bypassed or rebuilt. Reasons:
 (a) node features in data_dense_v3.pt come from norm_matrix, which S3B.5 proves is a
     fourth construction that no transform of M1/M2/OLD reproduces (best r = 0.738);
 (b) the loader's per-band min-max is applied inside process(), so the transform is frozen
     into the cache — swapping the normalization also requires a rebuild;
 (c) PyG reloads the cache purely on the filename in processed_file_names, so simply
     changing the .mat inputs would be SILENTLY IGNORED.
Recommended (do NOT execute in this stage):
 - bump processed_file_names to a new version string per feature source, e.g.
   data_dense_v4_m1.pt / _m2.pt / _old.pt, so each source gets its own cache and the
   existing v3 cache is never silently reloaded — the same guard pattern the repo already
   uses for the v2->v3 change (lines 34-38);
 - keep data_dense_v3.pt on disk as the frozen baseline; do not delete any cache;
 - feed M1/M2/OLD as RAW per-band ALFF and let the chosen normalization be applied
   explicitly, so the S3B analysis maps onto what the model actually sees;
 - before any of that, resolve the FLIRT misalignment question, because if confirmed it
   means the current baseline results rest on displaced node features.

## UNRESOLVED PROVENANCE
1. The numeric Hz edges of the three DPARSFA bands are NOT stated in the notebook; they
   live in the DPARSFA run configuration for full_slow5 / full_slow4 / full_classical,
   which I did not locate. Whether they equal the frozen (0.010,0.027) / (0.027,0.073) /
   (0.010,0.080) is UNKNOWN — assumed by the column naming only.
2. The FLIRT misalignment needs an independent anatomical confirmation (see CALIBRATION).
3. The notebook asserts "validated on real data (r=1.0000 vs official rois_aal ...)" and a
   "single-subject 3.5 validation" that "grids/affines matched". The affines demonstrably
   do NOT match (origin differs by (6,-6,1) mm), so that validation claim is inconsistent
   with the files; no stored log of it was found.
4. norm_matrix covers the 956-subject phenotypic set; the frozen cohort is 954. The two
   S0-excluded subjects have norm_matrix rows that are simply never selected.
5. The adversarial cross-check workflow for this stage FAILED THREE TIMES (all agents,
   API 529 Overloaded, zero results each run). Everything above is my own direct
   verification, including two real-FLIRT positive controls; it has not been reviewed by
   an independent second reader.

S3B.5 STATUS: EVIDENCE COMPLETE
