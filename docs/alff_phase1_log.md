# ALFF Phase 1 — Detailed Log

Everything done from the ALFF-download-and-recompute investigation through the Step 1.6
decision point, in order. This covers `docs/COMPLETE_PLAN.md.pdf`'s Phase 1 ("REBUILD ALFF")
in full, plus the reasoning that led to starting this investigation in the first place. See
`docs/plan.md` and `docs/COMPLETE_PLAN.md.pdf` for the original plan this log follows.

## Why this investigation started

A-GCL's self-supervised encoder, trained on this project's real 956-subject ABIDE-I dataset,
never learned to separate ASD from non-ASD across 200 real epochs (`corrected`/budget-mode
run, job 1842676) — validation accuracy never beat the untrained epoch-0 baseline. A raw-FC
baseline (flatten `cropped_matrix`, linear SVM, no GNN at all) reached 64.0% accuracy / 67.8%
AUC on the same subjects — proving the connectivity data carries real signal the encoder
wasn't extracting.

Investigating why led to the node features: `norm_matrix` (the existing `_nf.mat` files) turned
out to already be **z-scored per subject per band** (mean≈0, std=1.0 exactly, verified directly
on real files) — not raw ALFF as the A-GCL paper's Section 2.1 describes (raw ALFF → per-subject
min-max to `[0,1]`). Traced to source: DPARSFA's standard ALFF pipeline automatically writes a
`zALFFMap` alongside the raw `ALFFMap`, and that z-scored output is what ended up in
`norm_matrix`. Separately, `docs/COMPLETE_PLAN.md.pdf` identifies a second, independent issue
with the existing ALFF: it was computed **voxelwise, then averaged into ROIs** — but ALFF
involves `|FFT|`, a nonlinear operation, so `average(ALFF(voxels)) ≠ ALFF(average(voxels))`.
A-GCL's own method computes ALFF from the ROI-averaged time series directly. This phase
rebuilds ALFF the correct way and tests whether it actually matters.

## Step 1.1 — Download `rois_aal.1D`

**Why `nofilt_noglobal` specifically:** ALFF needs the unfiltered signal's full spectrum —
`filt_noglobal`'s bandpass filter would remove the very low-frequency components ALFF
measures. `rois_aal.1D` is already the ROI-averaged time series (AAL atlas) — exactly what
Step 1.2 needs as input, and exactly A-GCL's own input per the paper.

**Files:**
- `scripts/download_alff_rois.py` — downloader, adapted from the sibling GraSTI-ACL project's
  working `preprocessing/download_abide_pcp.py` (same streaming-download-with-resume/retry
  logic), simplified to just the one derivative needed.
- `scripts/download_alff_rois.slurm` — SLURM wrapper (2 CPUs, 4G, 2h budget — network I/O, not
  compute).

**Subject list — a deliberate design choice:** derived from `data/raw/ASD_ADJ` and
`data/raw/NC_ADJ`'s existing filenames (strip the `_adj.mat` suffix to get the ABIDE `FILE_ID`),
**not** a fresh phenotypic-CSV site filter. This guarantees every downloaded `rois_aal.1D`
lines up 1:1 with a subject we already have a PCC edge matrix and ASD/NC label for, rather
than risking a different subject set than what's already on disk (the sibling project's own
cohort, for comparison, excludes OHSU/CALTECH/TRINITY and uses a different subject count —
our data includes Caltech, so the cohorts are genuinely different).

**Result — job 1867348:** `Pipeline: cpac | Strategy: nofilt_noglobal | Derivative: rois_aal.1D`.
**956/956 downloaded successfully, 0 failed.** Output: `data/ALFF_need/rois_aal/*.1D` (956
files), manifest at `data/ALFF_need/download_manifest.csv`.

Verified on a sample file (`CMU_a_0050642_rois_aal.1D`): shape `(236, 116)` (236 timepoints ×
116 AAL regions), all finite, no dead (all-zero) columns.

## `data/subject_tr.csv` — TR per subject

TR (repetition time, needed for `compute_alff`'s FFT frequency-band limiting) isn't in the
downloaded `.1D` files or derivable without the raw NIfTI headers, which this plan
deliberately avoids downloading. The user supplied `data/subject_tr.csv` directly (1035 rows —
matching the plan's own exclusion-cascade starting point, `1035 → 1009 → QC → FD → 979`).

**Verification and filtering performed:**
- Confirmed all 956 of our subjects are present in the 1035-row file (0 missing) — no gaps
  that would have blocked any subject from getting ALFF computed.
- 79 records in the file belonged to subjects outside our 956-subject cohort. Filtered these
  out in place, so `data/subject_tr.csv` now contains **exactly** our 956 subjects (957 lines
  including header), sorted by `FILE_ID`.
- Cross-checked one subject end-to-end: `subject_tr.csv`'s `N_VOLUMES` for `CMU_a_0050642` is
  236 — exactly matching the downloaded `.1D` file's actual timepoint count (`T=236`).

## Step 1.2 — The ALFF computation script

**File:** `scripts/compute_alff.py`. Implements the plan's Step 1.2 code verbatim:

```python
BANDS = [(0.010, 0.027),   # slow-5
         (0.027, 0.073),   # slow-4
         (0.010, 0.080)]   # classical

def compute_alff(ts, tr):
    ts = detrend(ts, axis=0)
    n = ts.shape[0]
    nfft = 2 ** int(np.ceil(np.log2(n)))          # zero-pad, matches DPABI
    amp = 2 * np.abs(np.fft.rfft(ts, n=nfft, axis=0)) / n
    freqs = np.fft.rfftfreq(nfft, d=tr)
    alff = np.zeros((ts.shape[1], 3))
    for b, (lo, hi) in enumerate(BANDS):
        m = (freqs >= lo) & (freqs <= hi)
        alff[:, b] = amp[m].mean(axis=0)
    return alff

def to_malff(alff):
    return alff / alff.mean(axis=0, keepdims=True)
```

**ROI selection:** AAL labels `< 9001` (90 cortical/subcortical regions) are kept; labels
`>= 9001` (26 regions: cerebellum 9001–9082, vermis 9100–9170) are dropped. Implemented by
parsing each `.1D` file's own `#`-prefixed header line (one AAL label per column) and filtering
on the actual label values — not a hardcoded column position. Directly verified against a real
file: 116 total labels, 90 kept (`2001`–`8302`), 26 dropped (`9001`–`9170`), confirming this
matches the file's real structure, not an assumption.

No nuisance regression applied — C-PAC's `nofilt_noglobal` output already has it applied
upstream, per the plan's Notes.

## Step 1.3 — Verify on ONE subject first

`compute_alff.py --verify-subject FILE_ID` runs the plan's 4 required checks and refuses to
imply Step 1.4 is safe unless all pass:

| Check | Expected |
|---|---|
| Shape | `(90, 3)` |
| All finite | no NaN, no Inf |
| `malff.mean(axis=0)` | ≈ 1.0 for each band |
| No zero rows | every ROI has a value |

Tested on 4 subjects spanning different sites/TRs/scan lengths (`CMU_a_0050642`, `Yale_0050628`,
`CMU_a_0050646`, `CMU_a_0050647`) — **all 4 checks passed on every one.**

## Step 1.4 — Run all subjects

**Files:** `scripts/compute_alff.py` (default mode, or `--max-subjects N` for a quick partial
test), `scripts/compute_alff.slurm`.

A `--max-subjects 1` dry run was done first to exercise the full save/aggregate pathway
end-to-end before committing to all 956 (temp output file deleted after verifying its
contents).

**Result — job 1867569 (~13 seconds runtime):** **956/956 succeeded, 0 failed, 0 NaN, 0
all-zero ROIs.** Saved to `data/ALFF_need/alff_new.npz` — arrays `alff` and `malff`, both
`(956, 90, 3)`, plus `file_ids`, `dx_group` (455 ASD / 501 NC, matching known counts exactly),
`ok` (all `True`).

Verified: `malff.mean(axis=1)` = exactly `[1, 1, 1]` per band, for every subject sampled — the
normalization invariant holds throughout, not just for one test case. Verified: `alff_new.npz`'s
956 `file_ids` are an **exact** match (0 missing, 0 extra) to the 956 subjects in
`data/raw/{ASD,NC}_ADJ`.

## Step 1.5 — Compare old ALFF vs new ALFF

**File:** `scripts/compare_alff.py` (run directly — single-threaded, lightweight, no SLURM
needed).

**Fairness correction applied:** the existing old ALFF (`norm_matrix`) is already z-scored per
subject per band. Correlating it directly against a *raw* new ALFF would conflate "the two
methods disagree" with "one side is normalized and the other isn't." To make a genuine
like-for-like comparison, the same per-subject-per-band z-score was applied to new ALFF before
correlating. Sanity-checked: old ALFF's z-score really is exact (`mean(|per-subject-band
mean|)=0.000000`, `mean(per-subject-band std)=1.0000`).

**Method:** for each of the 90 ROIs × 3 bands (270 combinations), Pearson correlate old vs.
new across all 956 subjects.

**Result:**

| Metric | Value |
|---|---|
| Overall r (mean / median) | 0.6576 / 0.6571 |
| Overall r (min / max) | 0.3818 / 0.8499 |
| r > 0.95 ("agree") | 0 / 270 (0.0%) |
| r in [0.6, 0.9] ("meaningfully different") | 193 / 270 (71.5%) |
| r < 0.5 ("very different") | 24 / 270 (8.9%) |

Per-band breakdown was uniform (slow-5 mean 0.656, slow-4 mean 0.652, classical mean 0.665) —
no single band drove the result. Per the plan's own decision table (r > 0.95 → old ALFF was
fine; 0.6–0.9 → continue to 1.6; < 0.5 → continue to 1.6 and double-check both):
**verdict = meaningfully different, continue to Step 1.6.** Saved:
`data/ALFF_need/alff_correlation.npz` (full 90×3 r-matrix).

## Step 1.6 — THE DECISION POINT: classical ML on both

**Files:** `scripts/compute_alff_baseline.py` (supports a single `--feature-set`/`--classifier`
combination for array-job use, or `--summarize-only` to print the combined table once all
result files exist), `scripts/compute_alff_baseline_array.slurm` (4-task array),
`scripts/compute_alff_baseline_elasticnet.slurm` (2-task follow-up, see below).

**Setup, per the plan:** `X_old`/`X_new` both `(956, 270)`, same z-score normalization applied
to both (reusing Step 1.5's `zscore_per_subject_per_band`), `StratifiedKFold(5, shuffle=True,
random_state=123)`, `StandardScaler` inside a `Pipeline`, `GridSearchCV(cv=5)`. Labels remapped
from ABIDE's own `DX_GROUP` (1=ASD, 2=control) to this project's convention (ASD=1, NC=0).

**Two classifiers**, both run on both feature sets (a 2×2 grid), by explicit request — not just
the plan's own `LinearSVC` spec:
- **LinearSVC** — `C ∈ {0.001, 0.01, 0.1, 1, 10, 100, 1000}`. The plan's own spec; matches the
  raw-FC baseline and `embedding_evaluation.py`'s classifier everywhere else in this project.
- **Elastic Net logistic regression** — `C` × `l1_ratio ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` (35
  combos), `solver='saga'`. Added to test whether a sparsity-inducing model (useful with 270
  features and only 956 samples) tells a different story than LinearSVC. If both agree, that's
  much stronger evidence than either alone.

**Execution history:** first submitted as a single 4-task array (job 1867572, 30 min budget,
4 CPUs/`n_jobs=4`). The two LinearSVC tasks finished in well under a minute each. The two
ElasticNet-LR tasks were markedly slower (bigger grid + `saga`'s slower convergence) — after
~6 minutes with not even one of 5 folds complete, and no partial-save mechanism (a timeout would
have lost the run entirely), those two tasks were cancelled and resubmitted separately as job
1867576 (2-task array, 8 CPUs/`n_jobs=8`, 2h budget). Both completed in this second run
(~8 minutes each).

**Final result — full 2×2 table:**

| Features | Classifier | Accuracy | AUC |
|---|---|---|---|
| Old ALFF | LinearSVC | 0.5722 ± 0.0306 | 0.5916 ± 0.0219 |
| Old ALFF | ElasticNet-LR | 0.5544 ± 0.0252 | 0.5933 ± 0.0152 |
| **New ALFF** | LinearSVC | **0.5742 ± 0.0439** | 0.5952 ± 0.0326 |
| **New ALFF** | ElasticNet-LR | **0.5596 ± 0.0370** | 0.5909 ± 0.0317 |

Plan's own reference point (a different, 979-subject cohort, LinearSVC only): Old ALFF
acc=0.59, AUC=0.616 — our own old-ALFF numbers (0.572 acc / 0.592 AUC) are in the same rough
range, on a different (956-subject) cohort.

**Exact differences (New − Old):**

| Classifier | Accuracy diff | AUC diff |
|---|---|---|
| LinearSVC | +0.21 pp | +0.36 pp |
| ElasticNet-LR | +0.52 pp | −0.24 pp |

**New ALFF wins on accuracy under both classifiers** (the consistent, cross-algorithm-agreement
part). It does **not** win cleanly on AUC — ElasticNet-LR actually favors Old ALFF slightly
there (−0.24pp). All four differences are far smaller than each result's own fold-to-fold
standard deviation (2.5–4.4pp) — effectively noise-level, not a meaningful gap by any normal
statistical standard.

Output: `data/ALFF_need/alff_baseline_{old,new}_{LinearSVC,ElasticNet-LR}.npz` (4 files, one per
combination).

## Verdict

Per the plan's own decision rule ("Whichever wins is your ALFF from here on"): **New ALFF is
the technical winner** (accuracy, both classifiers agree) — this closes out Phase 1.

But the practical takeaway is more important than the technical tie-break: **old and new ALFF
perform equivalently for classification.** Neither is a strong standalone predictor (~55–57%
accuracy either way, barely above chance-adjacent). This is consistent with, and reinforces,
the earlier finding that motivated this whole investigation: the real diagnostic signal in this
dataset appears to live predominantly in the **connectivity structure** (the raw-FC baseline
reached 64.0% accuracy / 67.8% AUC — meaningfully better than either ALFF variant), not in
node-level ALFF amplitude features, regardless of which ALFF computation method is used to
produce them.

## File inventory

**Scripts (all in `scripts/`):**
- `download_alff_rois.py` / `.slurm` — Step 1.1
- `compute_alff.py` / `.slurm` — Steps 1.2–1.4
- `compare_alff.py` — Step 1.5
- `compute_alff_baseline.py`, `compute_alff_baseline_array.slurm`,
  `compute_alff_baseline_elasticnet.slurm` — Step 1.6

**Data (all in `data/` or `data/ALFF_need/`):**
- `data/subject_tr.csv` — 956 subjects, filtered
- `data/ALFF_need/rois_aal/*.1D` — 956 downloaded time-series files
- `data/ALFF_need/download_manifest.csv`
- `data/ALFF_need/alff_new.npz` — the computed new ALFF/mALFF for all 956 subjects
- `data/ALFF_need/alff_correlation.npz` — Step 1.5's per-(ROI,band) r-matrix
- `data/ALFF_need/alff_baseline_{old,new}_{LinearSVC,ElasticNet-LR}.npz` — Step 1.6's 4 results

**SLURM jobs run this phase:** 1867348 (download), 1867569 (compute ALFF), 1867572 (baseline
array, LinearSVC tasks completed / ElasticNet-LR tasks cancelled), 1867576 (ElasticNet-LR
resubmission, both completed).

## Next steps (per the plan, not yet started)

Phase 2 — "THE GNN LADDER": Step 2.0 (fix the `Beta(nan, nan)` crash in `TA_encoder.py` that
killed 4 of 12 prior Stage B jobs, plus add a clean divergence check for NaN `model_loss`),
Step 2.1 (write the winning — New — ALFF into `norm_matrix` for every subject, delete
`processed/data.pt` so the loader rebuilds, verify 3 known subjects' tensors match), Step 2.2
(the anchor run), then the 5-round ladder (~12 jobs) and Phase 3's final numbers.
