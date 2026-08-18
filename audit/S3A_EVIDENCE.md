# S3A — MATHEMATICAL + NUMERICAL VALIDATION OF THE THREE ALFF SOURCES
2026-08-18 | HEAD 906a494b… | tree ?? ALFF_func_proc/ | no data/code modified
Cohort frozen: 954 / 455 ASD / 499 NC / 90 ROIs /
sha256 aca3d945f7d89ccb1cc6fb46ca01f6036ccf036d81d120d2b09fe4bd0d1d68c9 — re-verified True
No ranking by accuracy, no normalization, no ComBat, no classifiers, no training.

## 1+4. RECOMPUTATION ACCURACY
Independent implementation written from the documented spec (does NOT import either
generator): s3a_recompute.py. Sample n=19 spanning 8 TR families (1.5, 1.651, 1.665,
1.666, 2.0, 2.2, 2.5, 3.0 s), T = 116..316, 12 sites, both classes.
Inputs: OLD from A-GCL nofilt rois_aal .1D; M1/M2 from GraSTIACL func_preproc +
AAL_61x73x61_YCG, validity = finite AND non-all-zero, per-ROI voxel detrend.

  source   max_abs_error   mean_abs_error   max_rel_error   mean_rel_error
  OLD      5.33e-15        3.46e-16         7.39e-16        7.55e-17
  M1       7.11e-15        3.83e-16         5.92e-16        6.68e-17
  M2       1.42e-14        8.40e-16         5.81e-16        6.30e-17
All 19 subjects reproduce at float64 machine precision (~1e-16 relative). No subject
exceeds 1.5e-14 absolute. T from the .1D equals T from the NIfTI for all 19.

## 2+3. FREQUENCY / BIN CORRECTNESS
Convention recovered from both generators and reproduced exactly:
  nfft = 2^ceil(log2 T)   (zero-padded; DPABI-style)
  amp  = 2 * |rfft(x, nfft)| / T        <-- divisor is T, NOT nfft
  freqs = rfftfreq(nfft, d=TR)
  band mask = (freqs >= lo) AND (freqs <= hi)   -- INCLUSIVE at BOTH edges
  ALFF[band] = mean of amp over the masked bins
Bands: slow5 0.010-0.027, slow4 0.027-0.073, classical 0.010-0.080 Hz.

  subject           T    TR      fs      nyq   nfft   df(Hz)    n5  n4  ncl
  Pitt_0050058     196  1.5000  0.6667  0.3333  256  0.002604    7  18   27
  CMU_b_0050652    316  1.5000  0.6667  0.3333  512  0.001302   13  36   54
  Leuven_2_0050722 246  1.6519  0.6054  0.3027  256  0.002365    7  19   29
  Leuven_1_0050682 246  1.6667  0.6000  0.3000  256  0.002344    7  20   30
  Caltech_0051473  146  2.0000  0.5000  0.2500  256  0.001953    8  24   35
  NYU_0050952      176  2.0000  0.5000  0.2500  256  0.001953    8  24   35
  USM_0050433      236  2.0000  0.5000  0.2500  256  0.001953    8  24   35
  UM_1_0050272     296  2.0000  0.5000  0.2500  512  0.000977   17  47   71
  SBL_0051556      196  2.2000  0.4545  0.2273  256  0.001776   10  26   40
  KKI_0050792      124  2.5000  0.4000  0.2000  128  0.003125    5  15   22
  KKI_0050825      152  2.5000  0.4000  0.2000  256  0.001563   11  29   45
  UCLA_1_0051261   116  3.0000  0.3333  0.1667  128  0.002604    7  18   27
  MaxMun_d_0051361 196  3.0000  0.3333  0.1667  256  0.001302   13  36   54
  (full 19 rows in s3a_recompute.csv / s3a_band_algebra.txt)
Nyquist > 0.080 Hz for every subject (min 0.1667 Hz), so all three bands are always
representable — the generator's Nyquist guard never fires.
Smallest band population is slow5 with 5 bins (KKI_0050792, T=124, nfft=128).
BOUNDARY-CONVENTION SENSITIVITY: no FFT bin lands exactly on 0.010, 0.027, 0.073 or
0.080 Hz in any of the 19 subjects, so switching the upper edge from <= to < changes bin
counts by 0 and ALFF by exactly 0.0. The inclusive convention is therefore documented but
untested by the data — a latent, not an active, difference.

## 5. INVALID / EXTREME VALUES — all 954x90x3 = 257,580 per source
  source   NaN   Inf   exact_zero   negative
  M1        0     0        0            0
  M2        0     0        0            0
  OLD       0     0        0            0
Robust extremes (>3*IQR beyond quartiles):
  M1  slow5 483 (0.56%)  slow4 455 (0.53%)  classical 446 (0.52%)
  M2  slow5 528 (0.61%)  slow4 251 (0.29%)  classical 315 (0.37%)
  OLD slow5 557 (0.65%)  slow4 463 (0.54%)  classical 435 (0.51%)
max/median ratio 7.9-11.8 across sources/bands. No pathological value anywhere.

## 6. M1 <= M2 — ZERO VIOLATIONS
Elements with M1 > M2 + 1e-12: **0 / 257,580**. Max violation (M1-M2) = 0.000000e+00
(exactly zero, not merely within tolerance). Per band: slow5 0, slow4 0, classical 0.
Max margin (M2-M1) = 6.410e+01, mean margin = 7.806e+00.
This confirms the S2-derived Jensen relation empirically over the entire array.

## 7. CLASSICAL-BAND RELATIONSHIP (actual FFT bin sets, not assumed)
Per subject the three bin sets satisfy, for all 19:
  slow5 ∩ slow4 = EMPTY (0 overlap bins — no bin sits exactly at 0.027 Hz)
  classical = slow5 ∪ slow4 ∪ gap,  gap = bins in (0.073, 0.080], |gap| = 2..7
  |classical| = |slow5| + |slow4| + |gap|  verified True for all 19
Naive assumption classical == (slow5 + slow4)/2 is FALSE: max error 3.27e+00
(range 0.77-3.27 across the sample) — a large, systematic error.
Exact bin-set algebra
  classical = ( slow5*|slow5| + slow4*|slow4| - Σ_overlap amp + Σ_gap amp ) / |classical|
holds to 7.11e-15 (machine precision) for every sampled subject.
=> Classical is a genuinely independent band average, NOT derivable from slow5/slow4
   without the (0.073, 0.080] bins.

## 8. RAW DISTRIBUTIONS (no normalization applied)
  src band        min        max      mean    median      std      IQR
  M1  slow5     0.46091    62.083    6.229   5.6054    3.073    3.5555
  M1  slow4     0.48303    48.912    4.9898  4.5498    2.1822   2.531
  M1  classical 0.48468    49.654    5.2173  4.7734    2.2524   2.6279
  M2  slow5     0.53964   126.18    13.916  13.277     4.9018   5.5646
  M2  slow4     0.57704    98.32    12.884  12.466     4.1473   5.2163
  M2  classical 0.57267   104.47    13.055  12.616     4.2386   5.2047
  OLD slow5     0.0018327  51.938    5.0541  4.3918    2.9444   3.3251
  OLD slow4     0.0027055  33.372    4.06    3.5977    2.145    2.4807
  OLD classical 0.0025317  38.155    4.2421  3.7608    2.2238   2.5671
Structural observation (not a ranking): M2 sits ~2.5x above M1 in level, as the
Jensen relation requires. OLD's minimum is ~250x smaller than M1's — see suspicious items.

## 9. AGREEMENT (raw values; agreement analysis, NOT ranking)
                overall           slow5             slow4             classical
  M1 vs M2   r=0.7366 ρ=0.6980  r=0.7576 ρ=0.7342  r=0.7132 ρ=0.6677  r=0.7305 ρ=0.6885
  M1 vs OLD  r=0.8116 ρ=0.7953  r=0.8013 ρ=0.7882  r=0.8113 ρ=0.7928  r=0.8067 ρ=0.7891
  M2 vs OLD  r=0.6127 ρ=0.5773  r=0.6252 ρ=0.6053  r=0.5914 ρ=0.5495  r=0.6007 ρ=0.5609
Subject-level Pearson (270 values per subject):
  M1-M2 : min 0.3165  p05 0.5771  median 0.7724  max 0.9544  n<0.5 = 24
  M1-OLD: min 0.4157  p05 0.6014  median 0.7279  max 0.9143  n<0.5 = 3
  M2-OLD: min 0.0618  p05 0.3710  median 0.5834  max 0.8528  n<0.5 = 234
Per-file: s3a_subject_agreement.csv, s3a_subject_corrs.npz

## SUSPICIOUS SUBJECTS / ROIs / SITES
ROI-level, and strongly coherent with S0/S1:
  OLD < 0.05 occurs in 48 cells / 15 subjects / 6 ROIs, concentrated in
  Temporal_Pole_Mid_L (22) and Rectus_R (14), then Rectus_L, Frontal_Mid_Orb_R,
  Frontal_Med_Orb_L, Amygdala_R (3 each).
  These are EXACTLY the two ROIs that caused the S0 exclusions — CMU_b_0050669 failed on
  AAL 87 = Temporal_Pole_Mid_L and Leuven_1_0050706 on AAL 28 = Rectus_R. Independent
  arrival at the same two regions from a different route (OLD value tail vs M1/M2 zero-
  coverage) is mutually corroborating: these are low-coverage / susceptibility-dropout
  regions, not a code defect.
Weakest cross-source ROI agreement (classical band): Cingulum_Post_L r(M1,OLD)=0.632,
  Frontal_Sup_Medial_R 0.799, Temporal_Pole_Mid_L 0.804, Amygdala_L 0.809.
  Strongest: Calcarine_L 0.979, Putamen_R 0.976, Calcarine_R 0.975.
Subjects with lowest M1-vs-OLD agreement: Caltech_0051462 (0.416), Leuven_1_0050685
  (0.467), UCLA_1_0051249 (0.498), NYU_0051002 (0.501), Olin_0050112 (0.511).
Site-level median M1-vs-OLD spans 0.690 (UCLA_2) to 0.785 (YALE); M2-vs-OLD spans
  0.464 (USM) to 0.712 (LEUVEN_1). No site is an outlier; the spread is gradual.
Fragile geometry: the smallest per-ROI valid-voxel count in the sample is 6 voxels
  (max 1510). A 6-voxel ROI mean is a weak estimator; M2's per-voxel averaging over 6
  voxels is weaker still. Not an error, but a robustness concern to carry forward.

## 10. FROZEN ORDER RE-VERIFIED
cohort sha256 matches S1 exactly. M1 file_ids == cohort order (True); M2 == cohort order
(True). OLD subset taken BY ID per the S1 rule — positional indexing would have been
wrong for 934 of 954 rows. 90-ROI axis untouched (S1 90/90 alignment stands).

## UNRESOLVED MATHEMATICAL ISSUES
1. This is a REIMPLEMENTATION check, not an external-reference validation. My code
   reproduces the generators' own convention to 1e-16; it does not prove that convention
   matches DPABI/AFNI/REST canonical ALFF. No independent reference implementation was
   run. The amp = 2|rfft|/T normalization with nfft-point zero padding is self-consistent
   but its equivalence to the published ALFF definition is UNVERIFIED here.
2. The >= / <= inclusive boundary convention is untested by the data (no bin lands on an
   edge in the 19-subject sample). A future cohort with different T/TR could hit an edge
   and silently double-count the 0.027 Hz bin in slow5 and slow4.
3. Zero-padding to the next power of two changes the frequency grid per subject, so band
   bin counts vary across subjects (slow5: 5 to 17 bins). Whether averaging over a
   subject-varying number of bins is desirable is a design question, not validated here.
4. Recomputation covers 19 of 954 subjects (2.0%). The full-array checks (5,6,7,8,9) cover
   all 954, but element-level recomputation does not.
5. OLD's ~250x smaller minimum than M1/M2 is explained by ROI coverage, not proven so;
   OLD inherits C-PAC's masking which S2 could not fully characterize.

## ARTEFACTS (all outside the repo, /users/3171356m/agcl_audit_s0/)
s3a_recompute.py, s3a_recompute.csv, s3a_recompute.log, s3a_recompute_summary.txt,
s3a_sample.json/.txt, s3a_band_algebra.txt, s3a_fullarray.txt, s3a_suspicious.txt,
s3a_subject_agreement.csv, s3a_subject_corrs.npz

S3A STATUS: EVIDENCE COMPLETE

================================================================================
# S3A CLOSE-OUT — ALFF FORMULA vs INTENDED SCIENTIFIC DEFINITION
2026-08-18 | HEAD 906a494b… | tree ?? ALFF_func_proc/ | stored ALFF NOT changed

## A. PAPER DEFINITION EVIDENCE  (Zhang et al. 2023, Med Image Anal 90:102932, §2.1)
Verbatim: "The node features are derived from 3 frequency bands of the ALFFs (Slow-5:
0.01-0.027 Hz, Slow-4: 0.027-0.073 Hz, classical: 0.01-0.08 Hz) in BOLD signals, which
are defined as the total power within the low-frequency range and are calculated from the
Fourier transform of the mean time series (Guo et al., 2017)."
Also: "In each ROI, the mean time series is calculated by averaging all the BOLD signals
in the region." And node features are later min-max normalised to [0,1] across the 3
channels (normalisation NOT exercised here).
  bands                     : PROVEN identical to ours (all three edges match exactly)
  "Fourier transform of the MEAN time series" : ROI-first. M1 CONSISTENT; M2 DIFFERENT
                              (M2's voxel-first order is not what the paper describes)
  "defined as the TOTAL POWER"                : DIFFERENT from our mean-amplitude, on two
                              axes at once (power vs amplitude, total/sum vs mean).
NOTE: this sentence also contradicts the same authors' own released method (source B) and
the canonical ALFF definition, so it reads as loose prose rather than an implementable
spec. Treated as wording evidence, not as the operational definition.

## B. ORIGINAL-CODE DEFINITION EVIDENCE  (qbmizsj/A-GCL, recovered from git history)
Original *code*: contains NO ALFF computation. The earliest tree (commit bed5441) has only
datasets/abideDataset.py, which LOADS precomputed features: x = nf['alff_value_cache']
from ASD_NF / HC_NF. The generating code was never released -> UNKNOWN from code alone.
Original *documentation*: README.md (commits 37c1082, 1f7e6c4, 2421850) is explicit —
  "We used the DPABI_V7.0 to calculate the ALFF on MATLAB. The function in DPABI is
   y_alff_falff.m"
and gives the literal call:
  [ALFF, fALFF, hdr] = y_alff_falff(sub_0050002, hdr.TR, 0.08, 0.01, mask, out, [],
                                    ScrubbingMethod='cut', hdr, 1)
  => HighCutoff 0.08, LowCutoff 0.01; TemporalMask = [] which the function's own docstring
     defines as "Empty ... means do not need scrube" -> NO SCRUBBING. PROVEN.
Same README also documents their preprocessing: fMRIPrep (not C-PAC), and AAL3 registered
to each subject's native space via ANTs; templates AAL1(116), AAL3(166), Shen268(268).
  ALFF FORMULA per original authors : PROVEN = DPABI v7 y_alff_falff.m
  preprocessing pipeline            : DIFFERENT from ours (fMRIPrep native-space vs
                                      C-PAC ABIDE-PCP MNI derivatives - see S2)
  atlas                             : DIFFERENT (AAL1-116 / AAL3-166 vs our AAL1-90)
  scrubbing                         : PROVEN none, and ours is also none - MATCH

## C. REFERENCE-TOOL DEFINITION EVIDENCE  (DPABI y_alff_falff.m, on disk)
File: DPABI/DPARSF/Subfunctions/y_alff_falff.m (the exact function the README names).
Core lines: paddedLength = 2^nextpow2(sampleLength); idx_LowCutoff = ceil(LowCutoff*
paddedLength*ASamplePeriod + 1); idx_HighCutoff = fix(HighCutoff*paddedLength*
ASamplePeriod + 1); detrend(AllVolume); zero-pad to paddedLength;
AllVolume = 2*abs(fft(AllVolume))/sampleLength; ALFF_2D = mean(AllVolume(idx_Low:idx_High,:)).
Its own header note: "the ALFF generated by the new version is sqrt(2/N) times of the
original version. (new version used: 2*abs(fft(x))/N; original version used:
sqrt(2*abs(fft(x))^2/N))".

  axis                        DPABI v7                       ours                verdict
  amplitude vs power          2*abs(fft)/N  = AMPLITUDE       same                PROVEN
  sqrt(power) convention      new version, no sqrt(power)     same                PROVEN
  one-sided FFT scaling       factor 2                        factor 2            PROVEN
  divisor T vs nfft           sampleLength (T), NOT padded    T                   PROVEN
  zero-padding                2^nextpow2(T), zeros appended   rfft(n=2^ceil(log2 T)) PROVEN
  mean vs sum across bins     mean(...)  (sum only for fALFF) mean                PROVEN
  detrending                  detrend before FFT              scipy detrend before PROVEN
  frequency boundaries        ceil(lo..)/fix(hi..) = incl/incl (f>=lo)&(f<=hi)     PROVEN

NUMERICAL VERIFICATION (MATLAB/Octave unavailable, so y_alff_falff.m was PORTED
line-for-line to Python including its 1-based ceil/fix index arithmetic):
  synthetic signals, 72 (T,TR) combinations (T in {116..512}, TR in {1.5,1.65,2.0,2.2,
    2.5,3.0}): bin sets IDENTICAL in all 72; max_abs_diff 5.55e-17, max_rel 9.69e-16
  real ROI time series, 19 subjects: max_abs(ours - DPABI_port) = 7.11e-15,
    max_rel = 5.54e-16; DPABI_port vs STORED values agree to 1.78e-15
=> our formula is the DPABI canonical ALFF, to machine precision.

## EXACT DIFFERENCES
1. Paper prose says "total power"; DPABI (which the authors state they used) computes
   MEAN AMPLITUDE. Our code follows DPABI. This is a paper-wording vs released-method
   discrepancy in the SOURCE MATERIAL, not a defect in our implementation.
2. Paper describes ROI-first only. M2 (voxel-first) is an ADDITIONAL variant with no basis
   in the paper text; M1 is the paper-aligned ordering.
3. Preprocessing and atlas differ from the original authors (fMRIPrep native-space AAL3 vs
   C-PAC MNI AAL1-90). Formula unaffected; already recorded in S1/S2.
4. No difference found on any of the eight audited formula axes.

## DOES THE PAPER-WORDING DIFFERENCE AFFECT OUR 954-SUBJECT DATA?
Quantified on the OLD route for all 954 subjects (variants computed, nothing normalised,
nothing overwritten). Bins per band across the cohort: slow5 5-17, slow4 15-47,
classical 22-71 -> a 3.23x spread in classical bin count BETWEEN subjects.
  variant vs ours (all 954x90x3)            pearson    spearman
  V2 sum|amp|      ("total amplitude")      0.5252     0.5381
  V3 mean amp^2    (mean power)             0.9098     0.9970
  V4 sum amp^2     (paper's literal wording)0.6466     0.7992
  within-subject-band ROI rank preservation: V2 exactly 1.000000 (positive constant scale);
    V3/V4 min 0.9848, mean 0.9953
  cross-subject artefact: corr(subject mean level, classical bin count)
    ours (mean) = +0.1441   vs   sum = +0.7865
YES, materially, if the literal wording were adopted: switching mean -> sum would inject a
strong scan-length/TR-dependent scale across ABIDE sites (+0.14 -> +0.79), and squaring
changes value distributions (though it nearly preserves within-subject ROI ordering).
Because sources B and C both authoritatively specify mean-amplitude and our code matches
them to 1e-16, NO change to the stored ALFF is warranted on this evidence. Stored ALFF
was not modified.

## FULL-COHORT RECOMPUTATION RESULT (all 954, performed)
SLURM array 1868818, 32 chunks x 4 CPU, all COMPLETED; 954/954 rows, ids match the frozen
cohort exactly.
  source   max_abs_error   mean(max_abs)   max_rel_error   mean(mean_abs)
  OLD      1.4211e-14      3.0472e-15      8.0807e-16      3.3210e-16
  M1       1.4211e-14      3.1184e-15      7.2822e-16      3.4487e-16
  M2       2.8422e-14      6.6874e-15      8.3025e-16      7.9530e-16
Subjects with max_rel > 1e-10: 0 / 954. Worst subject for all three sources is the same
one, UM_1_0050296 (T=296, the longest-nfft class) — consistent with float accumulation,
not a data fault. T range 116..316.
=> All three stored ALFF arrays are reproduced element-wise at float64 machine precision
   for the ENTIRE cohort, not just a sample.
NEW ROBUSTNESS DATUM: minimum per-ROI valid-voxel count across all 954 is 2 voxels
(Yale_0050605); 19 subjects have an ROI with <20 valid voxels. Yale_0050605 was already
the top subject in the OLD<0.05 low-value tail — third independent arrival at the same
coverage problem.

## FINAL UNRESOLVED ISSUE
DPABI could not be EXECUTED — no MATLAB or Octave on this system. Equivalence rests on a
line-for-line Python port of y_alff_falff.m that I wrote and cross-checked on 72 synthetic
(T,TR) combinations and 19 real subjects. A port validated against its own source text is
strong evidence but is not the same as running DPABI v7 itself; a residual risk remains
that some behaviour outside the ported lines (e.g. its segmented detrend over CUTNUMBER
blocks, or scrubbing paths we do not use) differs. Additionally, the paper-vs-README
contradiction over "total power" is a defect in the published source material that this
audit can document but not resolve; only the original authors can say which they meant,
and their released README plus DPABI both point to mean amplitude.

S3A STATUS: EVIDENCE COMPLETE (formula question closed on B+C; paper wording flagged)
