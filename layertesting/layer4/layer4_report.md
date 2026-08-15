# Layer 4 — View Learner, Part by Part: Is It Masking Out Diagnostic Edges?

Part of the full pipeline audit (`layertesting/layer1-3`). Produced by
`layertesting/layer4/test_layer4_view_learner.py` (job 1867868). Tests whether the adversarial
edge-masking view learner is complicit in Layer 3's finding — training decollapses subject
embeddings but never along the ASD/NC axis — by checking whether it systematically masks out
edges that are more diagnostically discriminative between ASD and NC.

## 4a — the view learner's own encoder shows the same pattern as Layer 3

| | Old ALFF | New ALFF |
|---|---|---|
| Pairwise cos, untrained | 0.9991 | 0.9978 |
| Pairwise cos, trained (ep30) | 0.2446 | 0.3972 |
| Class-mean cos, untrained | 1.0000 | 1.0000 |
| Class-mean cos, trained (ep30) | 0.9959 | 0.9972 |
| Probe acc, trained | 0.4968 | 0.5011 |

Same story as the main encoder: severe untrained collapse, substantial decollapse with training,
but class-mean cosine similarity barely moves and probe accuracy stays at chance. Confirms this
"decollapse without class separation" pattern isn't specific to one encoder instance — it's a
property shared by any `TUEncoder` in this setup.

## 4b — the key test: is the view learner masking out diagnostic edges? **No.**

Correlation between each edge's diagnostic relevance (`abs` Welch t-statistic, ASD vs NC, on the
actual `edge_weight` tensor the view learner operates on) and its mean keep-probability `mu`
across all 956 subjects:

| | Old, untrained | Old, trained (ep30) | New, untrained | New, trained (ep30) |
|---|---|---|---|---|
| Pearson | +0.0223 | +0.0461 | +0.0209 | +0.0665 |
| Spearman | +0.0510 | +0.1157 | +0.0453 | +0.0989 |

**No negative correlation anywhere** — the hypothesis motivating this layer (that the view
learner adversarially discards the edges that matter most for classification) is **refuted**.
If anything, the correlation is weakly **positive** and gets slightly stronger with training —
the view learner mildly *favors* keeping more diagnostic edges, not discarding them. The effect
is weak (Spearman ~0.10-0.12, far from a strong systematic relationship) but the direction rules
out "targeted signal destruction" as the mechanism.

**A secondary, real observation from the same data**: after training, `mu` compresses toward 0
across almost *all* edges, not selectively — old ALFF's mean keep-probability tops out at 0.46
(down from ~0.52 untrained), new ALFF's tops out at just 0.20 (down from ~0.53 untrained). The
view learner is learning to mask out most edges fairly indiscriminately, not selecting which
ones to target. That's a property of the loss/regularizer balance driving the view learner
(`view_loss = batch_loss + 2*reg + 0.4*cr_loss`, `reg = mean(mu)`), not of edge-level
discrimination — worth Layer 6 (loss functions, decomposed) picking up, not resolved here.

## 4c — mask-sampling correctness

- `mu_corr`, `mu_paper`, `edge_mask_corr`: all finite, all within `[0, 1]` (up to a `[0.0000,
  0.9999]`-style float rounding at the extremes) — no NaN/inf, no out-of-range values, at any
  checkpoint, either ALFF source.
- Corrected-path `mu` is exactly symmetric ((i,j) == (j,i)) at every checkpoint, both sources —
  correct by construction (`symmetrize_edge_logits`).
- Paper-exact `mu` is **not** exactly symmetric, as documented — but the *degree* of asymmetry
  differs sharply by source: essentially zero measurable asymmetry for old ALFF (0.00% of edges
  differ by more than 1e-4, at both checkpoints) vs. real, growing asymmetry for new ALFF (3.4%
  untrained → 8.8% trained). This is explained by 4a: old ALFF's node embeddings stay more
  tightly collapsed than new ALFF's even after training (pairwise cos 0.24 vs 0.40) — since the
  paper-exact path's asymmetry comes entirely from `node_emb[i] != node_emb[j]` feeding the
  edge MLP in a different concatenation order, near-identical node embeddings mechanically
  produce near-identical (i,j)/(j,i) logits even without explicit symmetrization. Not a bug —
  a downstream consequence of how collapsed each source's embeddings happen to be.

## Verdict

The view learner is not the smoking gun either — at least not through the specific mechanism
under test (selectively discarding diagnostically useful edges). It is finite, well-formed,
correctly differentiated between the corrected/paper_exact profiles, and if anything mildly
biased toward *preserving* diagnostic edges. What it does do is push overall keep-probability
down broadly during training, which is a property of the loss balance, not of adversarial
targeting. Two encoders now show the identical "decollapse without class separation" pattern
with no clear culprit at the architecture or masking level — the remaining candidates are the
loss functions themselves (Layer 6: does the InfoNCE formulation or the memory-bank regularizer
actively reward class-irrelevant diversification?) and the augmented view construction / gradient
flow (Layers 7-8). Layer 5 (optimizers — confirming gradients reach every parameter, not just
that loss values move) is next in sequence, though Layer 6 may be the more informative jump
given this layer's result.
