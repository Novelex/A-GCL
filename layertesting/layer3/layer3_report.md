# Layer 3 — Main Model, Part by Part: Where Does Collapse Actually Originate?

Part of the full pipeline audit (`layertesting/layer1`, `layer2`). Produced by
`layertesting/layer3/test_layer3_model_architecture.py` (job 1867863). Localizes the near-total
embedding collapse first observed in `scripts/diagnose_collapse.py` (cos-sim 0.9997, untrained)
to a specific stage inside `TUEncoder`/`GInfoMinMax`, and tests the pre-pool `F.normalize()`
hypothesis directly via an ablation.

## Method

Replicated `TUEncoder.forward()` step by step (not monkey-patched) to capture 7 named stages:
`gin1` (after GIN layer 1) → `gin2` (after GIN layer 2) → `normalized` (after the pre-pool
`F.normalize(x, dim=1)`, a documented deviation from the paper) → `h_raw` / `h_normalized`
(graph embedding, pooled two ways as a direct ablation of that one line) → `z_raw` / `z_normalized`
(after `proj_head`). Node-level stages mean-pooled per subject for measurement only. Each stage
measured untrained (fresh init, eval mode) and after 30 real training epochs (same loop/hyperparams
as `diagnose_collapse.py`), for both old ALFF (real path) and new ALFF (Layer 2's per-band-min-max
swap, current pipeline normalization).

## Result 1 — the `F.normalize()`-before-pooling hypothesis is refuted

Collapse is already present (cos-sim 0.996–0.999) at **`gin1_subj`**, i.e. after just the first
GIN layer, before layer 2, before pooling, before `F.normalize()` ever runs. And the direct
ablation — `h_raw` (no pre-pool normalize) vs. `h_normalized` (real code path) — shows almost no
difference: untrained, old ALFF, `h_raw` cos=0.9988 vs. `h_normalized` cos=0.9986. Removing the
line does not fix collapse, because collapse is already fully baked in one layer upstream of it.
**This specific line is not the cause.**

## Result 2 — untrained collapse is near-total everywhere, consistent with the prior finding

Every stage, both ALFF sources, untrained: cos-sim 0.996–0.999, class-mean (ASD vs NC) cos-sim
exactly 1.0000. Confirms `diagnose_collapse.py`'s finding with finer granularity: this isn't a
late-stage artifact, it's present from the very first message-passing layer at random
initialization — consistent with oversmoothing on the dense, fully-connected 90-node graph (every
node aggregates from all 89 others in a single hop, so two layers reach the whole graph twice).

## Result 3 — training substantially reduces raw collapse

| Stage | Old, untrained | Old, trained (ep30) | New, untrained | New, trained (ep30) |
|---|---|---|---|---|
| gin1_subj | 0.9994 | 0.7131 | 0.9982 | 0.6858 |
| gin2_subj | 0.9988 | 0.3271 | 0.9964 | 0.1684 |
| h_normalized | 0.9986 | 0.3216 | 0.9961 | 0.1800 |
| z_normalized | 0.9983 | 0.0626 | 0.9957 | 0.0910 |

Pairwise cosine similarity drops from ~0.999 to 0.06–0.33 by the final projected embedding.
Gradients are flowing and the model is not stuck — training does actively pull subject
embeddings apart. The architecture is not permanently broken.

## Result 4 — the real finding: de-collapsing does not recover any diagnostic signal

Two numbers tell the story, both ALFF sources, both stay essentially flat across training:

- **Quick-probe LinearSVC accuracy**, z_normalized: old untrained 0.5105 → old trained 0.4717;
  new untrained 0.5345 → new trained 0.5126. **Chance, before and after training, at every stage.**
- **Class-mean (ASD vs NC) cosine similarity**, z_normalized: old untrained 1.0000 → old trained
  0.9876; new untrained 1.0000 → new trained 0.9806. **Still nearly 1.0 after training**, even
  though *individual*-subject pairwise cos-sim dropped to ~0.06–0.09 over the same training run.

That gap is the finding: training spreads subject embeddings apart from *each other*
(pairwise cos-sim collapses toward 0), but the ASD-mean and NC-mean directions stay almost
identical (class-mean cos-sim stays ~0.98–0.99). The model is not stuck in literal collapse by
epoch 30 — it successfully diversifies embeddings to satisfy the InfoNCE objective — but it
diversifies them along directions that carry no information about the label. The contrastive +
adversarial-augmentation objective has no supervision signal telling it which variation matters;
it can be (and apparently is) satisfied by learning invariance/diversity that has nothing to do
with ASD vs. NC.

## Old ALFF vs. new ALFF

Nearly identical pattern on every metric, both sources. This is not an ALFF-source-specific
problem — swapping the node-feature source doesn't change where or how signal disappears.

## Verdict

Not a normalization bug, not a dead/frozen network, not literal permanent collapse. The
untrained network starts oversmoothed (present from GIN layer 1, likely intrinsic to two-layer
message passing on a dense fully-connected 90-node graph). Training fixes the raw collapse
number but does not fix the actual problem: **the contrastive/adversarial training objective is
being satisfied in a way that is uncorrelated with the diagnostic label**, evidenced by
probe accuracy staying at chance throughout while class-mean cosine similarity stays high even
as pairwise similarity drops sharply. The next place to look (Layer 4 — view learner) is whether
the adversarial edge-masking view learner is masking edges in a way that specifically destroys
FC-derived class signal, since InfoNCE-style invariance training pushed toward "be invariant to
whatever the view learner masks" is exactly the kind of objective that could learn to discard the
very edges that carry diagnostic information, if the view learner isn't masking in a
label-relevant way.
