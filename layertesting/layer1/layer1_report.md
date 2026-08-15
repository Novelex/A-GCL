# Layer 1 — Data: Format, Normalization, Status

Part of the full pipeline audit (see `/users/3171356m/.claude/plans/eager-drifting-teapot.md`).
This layer checks every raw input source **as it exists on disk**, before anything in the
pipeline touches it — format, whether normalization has already happened upstream, and overall
cross-source health. Produced by `layertesting/layer1/test_layer1_data.py`.

## Old ALFF (`norm_matrix`)

| | |
|---|---|
| File format | `.mat`, key `norm_matrix` |
| Shape (per subject) | `(90, 3)` — 90 AAL ROIs × 3 frequency bands |
| dtype | `float64` |
| All finite (956 subjects) | ✅ True, no NaN/Inf |
| **Normalization status** | **Already z-scored per subject per band** |

**Verification:** across all 956 subjects, `mean(|per-subject-band mean|) = 0.00000000`,
`mean(per-subject-band std) = 1.000000` — exact, not approximate. This confirms (again, now
formally as part of the layer audit) what was traced earlier this session: `norm_matrix` is
DPARSFA's automatic `zALFFMap` output, not raw ALFF. See `docs/alff_phase1_log.md` for the
provenance trace to the MATLAB pipeline stage that produces this.

## New ALFF (`alff_new.npz`)

| | |
|---|---|
| File format | `.npz`, keys `file_ids`, `alff`, `malff`, `dx_group`, `ok` |
| `alff` shape | `(956, 90, 3)`, `float64` |
| `malff` shape | `(956, 90, 3)`, `float64` |
| All finite | ✅ True (both `alff` and `malff`) |
| **Normalization status** | **`alff` is raw, un-normalized** (as computed by `compute_alff.py`'s Step 1.2 formula) |

**Verification:** `mean(|per-subject-band mean|) = 4.45`, `mean(per-subject-band std) = 1.98` —
clearly not z-scored, real magnitude variation present. This is the **expected, correct** state
at this layer — `alff` is deliberately left raw so downstream normalization choices (z-score,
ComBat, etc.) are applied explicitly and separately, not baked in unrecoverably the way old
ALFF's z-scoring was. `malff` (mean-normalized per band) does carry the expected invariant:
`malff.mean(axis=1)` averages to exactly `[1, 1, 1]` across all subjects.

## Raw PCC / FC (`cropped_matrix`)

| | |
|---|---|
| File format | `.mat`, key `cropped_matrix` |
| Shape (per subject) | `(90, 90)` |
| dtype | `float64` |
| On-disk range (sample) | `[-0.203, 1.000]` |
| Diagonal (self-correlation) | exactly `1.0` |
| NaN present (sample) | ❌ None |
| **Normalization status** | **Raw Pearson correlations, not yet max-abs normalized** |

Pearson correlation is bounded to `[-1, 1]` by construction, so these values are already
"normalized" in the trivial sense of being bounded — but the pipeline's own max-abs
normalization step (dividing by each subject's own max absolute value, per the paper's Section
2.1) has **not** been applied yet at this layer; that happens during Layer 2 (graph
construction, `abideDataset.py`), not here. Diagonal = 1.0 exactly (self-correlation, confirmed
previously this session across 10 subjects — see `correction.md` Section 4) means the max-abs
normalization is a documented no-op for real data (dividing by exactly 1.0 changes nothing).

## Labels

- ABIDE's own convention: `DX_GROUP ∈ {1, 2}` (1 = ASD, 2 = control), stored in `dx_group`.
- Project convention used everywhere downstream: **ASD = 1, NC = 0** (explicitly remapped, not
  ABIDE's raw encoding — see `correction.md` Section 1).
- Counts: **N = 956, ASD = 455, NC = 501** — matches every other count taken this session.

## Cross-source status

- **Subject-set consistency:** the 956 subjects in `data/raw/{ASD,NC}_ADJ` and the 956
  `file_ids` in `alff_new.npz` are an **exact match** — 0 missing, 0 extra, in either direction.
- **Old vs. new ALFF relationship (Step 1.5, already established):** mean Pearson r = 0.6576
  across all 270 (ROI, band) pairs — meaningfully different, not interchangeable, 0 pairs reach
  r > 0.95.
- **Known classifier baselines, for reference against later layers:**

  | Source | Accuracy | AUC |
  |---|---|---|
  | Raw PCC (no GNN) | 0.6401 | 0.6777 |
  | Old ALFF (LinearSVC) | 0.5722 | 0.5916 |
  | New ALFF (LinearSVC) | 0.5742 | 0.5952 |

## Verdict

**Layer 1 is clean.** Every source is internally consistent (correct shapes, no NaN/Inf,
correct subject counts, correct cross-source alignment), and the normalization status of each
is now explicitly documented rather than assumed: old ALFF is pre-normalized (z-scored)
upstream and cannot be un-normalized; new ALFF is deliberately raw; raw PCC is
correlation-bounded but not yet max-abs normalized. Nothing at this layer explains the signal
loss seen later in the pipeline — the raw-PCC and ALFF-alone baselines above prove real signal
is present here. Layer 2 (graph construction) is next.
