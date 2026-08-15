# Layer 2 Follow-up — Node Feature Normalization: Joint vs. Per-Band

Follow-up to `layer2_report.md`. Triggered by the user's own hypothesis: `datasets/abideDataset.py`
normalized node features with a single JOINT min-max (one min/max over the full 90×3 tensor per
subject) rather than a PER-BAND min-max (each of the 3 ALFF bands scaled to its own [0,1] range).
Joint normalization lets whichever band has the largest raw magnitude claim the full [0,1] range
while the other bands get compressed — verified on real data that this happens consistently
(slow-5 dominates every subject checked).

**Fix applied**: `datasets/abideDataset.py` now does per-band min-max (`x.min(dim=0)` /
`x.max(dim=0)`, per band, not global). Cache bumped `data_dense_v2.pt` → `data_dense_v3.pt` so a
stale joint-normalized cache is never silently reloaded.

## Test 1 — per-band fix vs. the original Step 1.6 baseline (z-scored)

`test_layer2_normalization_fix.py` (job 1867856). This was NOT a clean single-variable test —
Step 1.6's baseline used **z-scoring**, not joint min-max, so this changed two things at once
(normalization *type* AND *scope*).

| Features | Accuracy | AUC | Δ vs. z-scored baseline |
|---|---|---|---|
| Old ALFF (per-band min-max) | 0.5837 ± 0.0102 | 0.6085 ± 0.0250 | acc +1.15pp, auc +1.69pp |
| New ALFF (per-band min-max) | 0.5795 ± 0.0264 | 0.6031 ± 0.0291 | acc +0.53pp, auc +0.79pp |

Both improved — but old ALFF (theoretically unaffected by joint-vs-per-band, since it's already
z-scored per band upstream) improved *more* than new ALFF on accuracy. That's the tell that this
comparison wasn't isolating the hypothesis cleanly.

## Test 2 — the isolated control: joint vs. per-band, holding min-max fixed

`test_layer2_normalization_joint_control.py` (job 1867858). Same min-max method both times;
only joint vs. per-band *scope* varies. This is the real test of the hypothesis.

| Features | Joint min-max | Per-band min-max | Per-band gain |
|---|---|---|---|
| Old ALFF | acc 0.5680 ± 0.0177 / auc 0.6019 ± 0.0235 | acc 0.5837 ± 0.0102 / auc 0.6085 ± 0.0250 | acc +1.57pp, auc +0.66pp |
| New ALFF | acc 0.5774 ± 0.0277 / auc 0.5946 ± 0.0400 | acc 0.5795 ± 0.0264 / auc 0.6031 ± 0.0291 | acc +0.21pp, auc +0.85pp |

## Verdict

**The hypothesis does not hold up under the isolated test.** If band-dominance from new ALFF's
raw, unequal-magnitude bands were the real problem, new ALFF should have shown the larger gain
from per-band normalization and old ALFF should have barely moved. The opposite happened on
accuracy (old ALFF gained 1.57pp vs. new ALFF's 0.21pp); AUC gains were roughly comparable
between the two (+0.66pp old vs. +0.85pp new). Per-band min-max looks like a mild, general
improvement in input conditioning for the classifier — not a fix that specifically rescues new
ALFF from a joint-normalization-induced band-dominance problem.

**Decision: keep the fix anyway.** It is still the more defensible normalization choice on its
own merits (each band gets a fair [0,1] range regardless of its raw units — that's correct
regardless of whether it moves the needle much), it does not hurt either ALFF source, and it
modestly helps both (+0.2 to +1.6pp depending on metric/source). There is no reason to revert a
correct fix because it wasn't the root cause of the pipeline's chance-level accuracy — the
search for that root cause continues at Layer 3 (main model, part by part).

**Practical implication**: the fix has not yet been exercised by an actual GNN run — no
`data_dense_v3.pt` cache exists yet (would be built automatically on the next `ABIDEDataset`
instantiation). All results above are classical-ML (LinearSVC) only, per Layer 2's established
in-memory-swap methodology, not the graph pipeline itself.
