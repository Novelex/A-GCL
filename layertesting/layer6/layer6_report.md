# Layer 6 — Loss Functions, Decomposed: What Is The Model Actually Learning?

Part of the full pipeline audit (`layertesting/layer1-4`). Produced by
`layertesting/layer6/test_layer6_loss_functions.py` (job 1867926). Tests whether the contrastive
loss succeeds at its own task, decomposes the loss terms over training, and — the key test —
checks whether the trained embeddings encode `SITE_ID` or node-strength instead of diagnosis.

## 6c — the headline finding: the model learns node-strength, not diagnosis, not site

Quick-probe results on the final (`z`) embeddings, trained (30 epochs) vs. untrained, both ALFF
sources:

| Target | Old, untrained | Old, trained | New, untrained | New, trained | Chance |
|---|---|---|---|---|---|
| DX (accuracy) | 0.5105 | 0.4905 | 0.5345 | 0.5167 | 0.500 |
| SITE (accuracy, 19 classes) | 0.1778 | 0.1768 | 0.1789 | 0.1653 | 0.0526 |
| node-strength (R²) | 0.0924 | **0.6372** | 0.1411 | **0.6691** | 0.0 |

DX stays at chance, exactly as Layers 3-4 already established. **SITE does not increase with
training** — 0.1778→0.1768 (old), 0.1789→0.1653 (new), if anything slightly down — ruling out
the site-confound hypothesis that motivated this layer. But **node-strength (sum of `|edge_weight|`
per subject, excluding self-loops) explodes from R²≈0.09-0.14 (untrained) to R²≈0.64-0.67
(trained)**. The model learns, almost to the exclusion of everything else, to encode each
subject's overall connectivity magnitude.

This is mechanistically exactly what the architecture would produce: `WGINConv.message()`
scales every message by `edge_weight` directly, `global_add_pool` **sums** (not averages) 90
node embeddings into the graph vector, and nothing in the architecture normalizes away
graph-level magnitude after that sum. A subject whose FC values run systematically stronger
produces systematically larger-magnitude messages, and summing (rather than averaging) 90 of
them preserves and amplifies that scale difference into the final embedding — consistent with
the huge, highly variable embedding norms already seen in Layer 3 (`z_raw` norm
488.88 ± 423.05 — a coefficient of variation over 0.85). This gives the contrastive objective an
"easy" axis of individual difference to satisfy itself with (subjects differ a lot and
consistently in overall connectivity magnitude), crowding out the harder, subtler,
diagnostically-relevant FC-pattern differences.

Since DX-probe accuracy stays at chance even as node-strength is encoded almost perfectly,
node-strength itself is not a sufficient proxy for diagnosis in this cohort — this is a genuine
nuisance covariate, not an accidental backdoor to the real signal.

## 6a — a second, deeper anomaly: the model doesn't even solve its own task well

| | Old, untrained | Old, trained (ep30) | New, untrained | New, trained (ep30) |
|---|---|---|---|---|
| Batch retrieval top-1 acc (chance 0.03125, B=32) | 0.9332 | **0.0485** | 0.9634 | **0.0603** |
| Mean pos-pair sim | 1.0000 | 0.0470 | 1.0000 | 0.1976 |
| Mean neg-pair sim | 0.9984 | 0.0399 | 0.9958 | 0.1921 |
| pos − neg gap | 0.0016 | 0.0071 | 0.0042 | 0.0055 |

The untrained retrieval accuracy (93-96%) is not real signal — it's a floor effect of total
collapse: every embedding is within 0.998-1.000 cosine similarity of every other, so the tiny,
essentially-arbitrary residual left over after normalization happens to favor the true
diagonal slightly, purely because a subject's original and lightly-masked-augmented graphs
share almost all their input in common. The meaningful number is the **gap** between positive-
and negative-pair similarity, and it stays tiny (0.002-0.007) at every checkpoint, untrained or
trained.

After 30 epochs of training explicitly minimizing `calc_loss` (which directly rewards `pos_sim`
being large relative to `neg_sim`), retrieval accuracy **falls to barely above chance** and the
pos-vs-neg gap never meaningfully grows. This is despite the logged scalar `model calc_loss`
dropping steadily across the same 30 epochs (old: 1.29→0.64, new: 1.18→0.74) — a reminder that a
decreasing InfoNCE loss value under strong temperature scaling (`exp(cos_sim/0.2)`) does not by
itself certify that the model is getting better at the literal same-subject-recognition task the
loss is supposed to encode. Combined with 6c, the more complete picture is: training reduces the
scalar loss by learning to spread embeddings out along the node-strength axis (which contributes
some separability to the softmax-style ratio inside `calc_loss` even without genuine positive/negative
pair discrimination), rather than by learning genuine augmentation-invariant, subject-identifying
structure.

## 6b — loss decomposition over training

Both ALFF sources show the same pattern across all 30 epochs: `reg` (mean keep-probability,
`mu.mean()`) collapses monotonically from ~0.47 (epoch 1) down to ~0.003-0.004 (epoch 30) — the
view learner rapidly learns to mask out nearly everything, consistent with Layer 4's finding
that keep-probability compresses toward 0 broadly. `cr_loss` (memory-bank term) is
**consistently 2-3x larger in magnitude** than `calc_loss` throughout training (e.g. old epoch
30: `cr_loss=9.87` vs `calc_loss=4.23` on the view side; `cr_loss=2.35` vs `calc_loss=0.64` on
the model side) — the memory-bank regularization term dominates the raw magnitude of the
combined loss for the entire run, not just briefly. Full per-epoch trace in
`layertesting/layer6/../logs/agcl-layer6-loss_1867926.err` (also captured verbatim, all 30
epochs, both sources).

## Verdict

This layer finds the clearest concrete lead of the whole audit so far: **the model's trained
representation is dominated by connectivity magnitude (node-strength), not diagnosis and not
site**, and mechanistically this traces directly to `global_add_pool` (sum, not mean) combined
with `WGINConv`'s edge-weight-scaled messages — architecture, not the loss functions or view
learner per se, though the contrastive loss is what rewards learning *any* separable axis and
node-strength is simply the easiest one available. Additionally, the model does not appear to
be solving its own literal InfoNCE retrieval task well even after 30 epochs, despite a
steadily-decreasing scalar loss — worth extending to the real 200-epoch schedule to see if this
resolves with more training or is a persistent property. The natural next step is no longer
deeper into the loss functions or view learner (both now checked) but to test the pooling
hypothesis directly: does replacing sum-pooling with mean-pooling, or adding explicit
graph-level normalization, reduce the node-strength R² and let DX-probe accuracy move off
chance?
