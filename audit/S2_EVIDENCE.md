# S2 — ALFF INPUT-SIGNAL + PREPROCESSING PROVENANCE AUDIT (evidence only)
2026-08-18 | HEAD 906a494b… | tree ?? ALFF_func_proc/ | no data/code modified
Cohort frozen: 954 / 455 ASD / 499 NC / 90 AAL ROIs /
sha256 aca3d945f7d89ccb1cc6fb46ca01f6036ccf036d81d120d2b09fe4bd0d1d68c9
No ranking, no normalization, no classifiers, no significance tests, no ComBat, no training.

## M1 INPUT PROVENANCE
derivative : cpac / nofilt_noglobal / func_preproc  (4D NIfTI)
path       : /mnt/scratch/.../muhammad-GraSTIACL/data/raw/func_preproc/{ID}_func_preproc.nii.gz
evidence   : data/raw/download_manifest.csv — 1035 rows "cpac/nofilt_noglobal/func_preproc",
             S3 URL .../Outputs/cpac/nofilt_noglobal/func_preproc/{ID}_func_preproc.nii.gz
generator  : layer_testing/dual_alff_recompute.py  FUNC_DIR = raw/func_preproc
availability: 954/954 present
atlas      : DPABI Templates/AAL_61x73x61_YCG.nii, labels 1..90 (frozen S1 order)
TR source  : phenotypic_filtered_v2.csv TR_seconds, cross-checked vs NIfTI header
             (TR_MISMATCH_TOL_SEC = 0.01, hard failure on breach)

## M2 INPUT PROVENANCE
Identical to M1 in every respect — same file, same load, same atlas, same TR, same
detrended voxel matrix. M1 and M2 are two branches inside ONE process_subject() call.

## OLD INPUT PROVENANCE
derivative : cpac / nofilt_noglobal / rois_aal  (.1D, 116 columns)
path       : A-GCL data/ALFF_need/rois_aal/{ID}_rois_aal.1D
evidence   : A-GCL data/ALFF_need/download_manifest.csv — 956 rows
             "cpac/nofilt_noglobal/rois_aal"; recorded size_bytes matches the bytes on
             disk exactly for every subject checked (274711/166286/204534).
generator  : A-GCL scripts/compute_alff.py — header-parsed, keeps codes < 9001 -> 90 cols
availability: 954/954 present

## SEPARATE FINDING — FC/ADJ AND DW USE A DIFFERENT STRATEGY
FC (cropped_matrix) and DW were built by notebooks/02_local_global_pcc.ipynb from
GraSTIACL data/raw/rois_aal, which is **cpac / filt_noglobal / rois_aal**
(download_manifest_rois_aal_filt_noglobal.csv, 1035 rows). Both the nofilt and filt
manifests name the SAME local path; the filt download (17:07) overwrote the nofilt one
(15:47). Proof it is the filt copy: on-disk sizes equal the filt manifest's size_bytes
exactly (271516 / 164510 / 202188) and differ from the nofilt sizes; A-GCL's copies match
the nofilt sizes. SHA-256 of A-GCL vs GraSTIACL .1D differ for every subject tested.
=> FC edges come from band-pass-filtered signal; OLD/M1/M2 ALFF do not.

## M1 vs M2 SAME INPUT: **YES** (provable, not inferred)
One nib.load per subject; one voxel_valid mask (finite AND non-all-zero); one
detrended_by_roi matrix; both methods consume exactly that. Identical TR, identical valid
voxel set, identical bands. Confirmed empirically in S1: the two npz files have identical
file_ids and byte-identical `tr` arrays; only `alff` differs.

## OLD vs NEW PREPROCESSING EQUIVALENCE: **PARTIAL / UNPROVEN at voxel level**
SAME  : same ABIDE FILE_ID, same C-PAC pipeline, same strategy token nofilt_noglobal,
        same scan/session, same T (0/954 mismatches), same TR (0/954 mismatches).
SAME  : ALFF algorithm is identical — both use detrend -> nfft=2^ceil(log2 T) zero-pad ->
        amp = 2|rfft|/n -> mean amplitude per band, BANDS (0.010,0.027)/(0.027,0.073)/
        (0.010,0.080). (compute_alff.py lines 48-57 vs dual_alff_recompute.py
        alff_from_timeseries.)
DIFFER: derivative product — OLD consumes C-PAC's own ROI-averaged rois_aal.1D;
        M1/M2 re-derive ROI means from the 4D func_preproc with a different atlas
        (DPABI YCG) and no func_mask.
UNPROVEN: whether the two are voxel-level equivalent. I attempted to replicate
        validate_aal_averaging.py (official aal_mask_pad + func_mask, ROI-average
        func_preproc, correlate against rois_aal). The POSITIVE CONTROL FAILED: the
        author's own filt_noglobal 4D vs filt rois_aal gave r mean 0.854 (min 0.360),
        not the >=0.99 the script requires. My replication is therefore not trustworthy
        and I draw NO equivalence conclusion from it. No stored run log of the original
        validation exists anywhere in the searched trees.
CONSISTENT: a like-for-like spectral check (all series ROI-averaged over AAL90 first)
        shows func_preproc and nofilt rois_aal are near-identical in high-frequency
        content — 0.2809 vs 0.2793, 0.3560 vs 0.3533, 0.3552 vs 0.3718 (fraction of
        power > 0.1 Hz). Consistent with the same underlying signal, but consistency is
        not proof of voxel-level identity.

## TR MISMATCH COUNT: **0**
M1.tr vs M2.tr                        0/954
M1.tr vs A-GCL subject_tr.csv         0/954
M1.tr vs GraSTIACL subject_tr.csv     0/954
A-GCL subject_tr vs GraSTIACL         0/954
53 distinct TR values (1.5, ~1.65 family, 2.0, 2.2, 2.5, 3.0). The ~1.65 cluster is
per-subject header-precision values, not one nominal TR — carried verbatim, not rounded.
Generator additionally enforces CSV-vs-NIfTI-header TR agreement within 0.01 s per subject.

## TIME POINTS
OLD .1D rows vs A-GCL N_VOLUMES            0/954 mismatches
A-GCL N_VOLUMES vs GraSTIACL N_VOLUMES     0/954 mismatches
nofilt .1D T vs filt .1D T                 0/954 mismatches
range 116..316 volumes. Usable-frame count after censoring: UNKNOWN (see gaps).

## SCAN/SESSION MISMATCH COUNT: **0**
All sources keyed on the same ABIDE FILE_ID, same S3 project path, same T, same TR.
No subject has two candidate scans/sessions in any manifest.

## SOURCE AVAILABILITY MISMATCH: **0**
nofilt .1D 954/954 · filt .1D 954/954 · func_preproc 954/954 · qc_passed 954/954

## PREPROCESSING PROVENANCE (only what files prove)
pipeline / software  : C-PAC (ABIDE PCP), token `cpac` in every manifest + S3 URL. PROVEN.
strategy token       : nofilt_noglobal (M1/M2 in, OLD in) ; filt_noglobal (FC in). PROVEN
                       as a token; the SEMANTICS of the token are ABIDE-PCP convention
                       (external documentation), not proven by any file here.
band-pass filtering  : PROVEN empirically, not assumed —
                       nofilt rois_aal: 20.6-31.3 % of power above 0.1 Hz
                       filt   rois_aal: 0.04-0.54 % above 0.1 Hz
                       func_preproc (ROI-averaged): 28.1-35.6 %, matching nofilt
                       => OLD and M1/M2 inputs are NOT band-pass filtered; FC input IS.
global signal regr.  : token `noglobal` on all three. No GSR. Token-level evidence only.
motion regression    : UNKNOWN — no C-PAC config, log or confounds file in any searched tree.
tissue regressors    : UNKNOWN — same reason.
nuisance regression  : UNKNOWN whether func_preproc has any. ABIDE-PCP convention says
                       func_preproc precedes nuisance regression while rois_aal follows it,
                       but no file here proves it and my replication's control failed.
detrending           : PROVEN in both generators, and it is OUR code, not C-PAC's —
                       OLD: scipy.signal.detrend(ts, axis=0) on the [T,90] ROI-mean matrix.
                       M1/M2: scipy.signal.detrend on each ROI's [T, n_valid_voxels] matrix
                       once, before the branch; generator also records
                       commute_max_abs_diff for detrend(mean) vs mean(detrend).
smoothing            : UNKNOWN — no file states a FWHM.
censoring / scrubbing: UNKNOWN — no motion/FD file, no scrub list, no frame-censoring
                       record anywhere. All T frames are used as-is by both generators.
voxel validity       : PROVEN (M1/M2 only) — voxel kept iff finite across time AND not
                       all-zero; ROI voxel set = AAL90 mask INTERSECT valid; non-finite
                       inside ROI territory is a hard error; zero-valid-voxel ROI is a
                       hard error (this is what excluded the 2 subjects).

## ALREADY-FILTERED STATUS
M1 input : NOT band-pass filtered (nofilt token + 28-36 % power > 0.1 Hz)
M2 input : identical to M1 — NOT filtered
OLD input: NOT band-pass filtered (nofilt token + 21-31 % power > 0.1 Hz)
FC input : IS band-pass filtered (filt token + 0.04-0.54 % power > 0.1 Hz)
Neither ALFF generator applies its own band-pass; both select FFT bins inside the band
instead, which is the correct construction for unfiltered input.

## EXACT ROI-FIRST vs VOXEL-FIRST DISTINCTION (from generator code)
Shared per ROI r: detrended voxel matrix D_r ∈ R^{T x V_r}; n = T;
nfft = 2^ceil(log2 T); A(x) = 2|rfft(x, nfft)| / n; band B_b = {k : lo_b <= f_k <= hi_b}.
  M1 (ROI-first) :  x̄(t) = (1/V_r) Σ_v D_r[t,v]        then
                    ALFF_M1[r,b] = mean_{k in B_b} A(x̄)[k]
  M2 (voxel-first):  ALFF_M2[r,b] = (1/V_r) Σ_v  mean_{k in B_b} A(D_r[:,v])[k]
i.e. M1 = f(mean over voxels), M2 = mean over voxels of f, with
f = band-mean of the FFT amplitude spectrum.
The ONLY intended difference is the ORDER of the voxel-average and the amplitude
operator. They differ because A(·) takes a complex modulus — a non-linear operation — so
|Σ z_v| != Σ |z_v| unless all voxel phases coincide. Everything else is byte-identical:
same voxel set, same single detrend, same TR, same nfft/zero-padding, same three bands,
same 90-ROI axis. Jensen-type inequality gives ALFF_M1 <= ALFF_M2 pointwise; consistent
with S1's observation that 257577/257580 elements differ while `tr` is byte-identical.
(Stated as the mathematical relation implied by the code — no ranking or evaluation.)

## ATLAS / ROI ORDER — FROZEN S1 ORDER CONFIRMED
M1/M2 : AAL_61x73x61_YCG.nii, roi_masks = {l: atlas==l for l in range(1,91)} — unchanged.
OLD   : rois_aal .1D codes < 9001, header order — unchanged, header byte-identical across
        all 956 files.
FC    : same .1D loader + same 9001 cutoff (different strategy, SAME ROI axis).
No re-derivation performed in S2; S1's 90/90 alignment stands untouched.

## SUBJECT-LEVEL MISMATCH COUNTS
IDs                : 0 / 954
TR                 : 0 / 954
source availability: 0 / 954
scan/session       : 0 / 954
time points        : 0 / 954

## UNRESOLVED PROVENANCE GAPS
1. My replication of validate_aal_averaging.py fails its own positive control
   (r mean 0.854 where >=0.99 is required). Voxel-level equivalence of func_preproc and
   rois_aal is therefore UNPROVEN in either direction. No original run log exists.
2. No C-PAC configuration file, nuisance-regressor list, motion/FD file or scrub record
   exists in any searched tree -> motion regression, tissue regressors, smoothing and
   censoring are all UNKNOWN, not "none".
3. filt/nofilt/global/noglobal semantics rest on ABIDE-PCP convention. The band-pass part
   is independently confirmed spectrally here; the GSR part is NOT independently confirmed.
4. FC/DW are built from filt_noglobal while OLD/M1/M2 ALFF are nofilt_noglobal. Node
   features and graph edges therefore derive from differently-filtered signal. Recorded as
   fact; consequences are out of S2 scope.
5. M1/M2 apply no func_mask (they use finite AND non-zero voxel validity instead), whereas
   C-PAC's rois_aal used the official func_mask. Different masking, not quantified here.
6. Usable frames after censoring unknown; both generators consume all T frames.
7. GraSTIACL manifests cover 1035 subjects vs the 956/954 used — the reduction step to 956
   is recorded in phenotypic_filtered_v2.csv but its filter criteria were not audited here.

## TOOLING NOTE
nibabel absent from A-GCL .venv; NIfTI reads used /users/3171356m/miniconda3/bin/python
(nibabel 5.4.0), read-only. Nothing installed. No file written in the repo or data/.

S2 STATUS: EVIDENCE COMPLETE
