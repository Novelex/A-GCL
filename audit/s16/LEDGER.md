# S16 CHECKPOINT C0 — THE LEDGER
Reading only. No code written, no job submitted. Every cell cites file:line.
"ORIGINAL" = commit 1fb91bd (`Add files via upload`), the released A-GCL code.
"OURS" = repo HEAD. "AUDIT" = the frozen data actually used by S11-S15.
Paper = `docs/Zhang et al_2023_A-GCL.pdf` (Zhang et al., Med Image Anal 90 (2023)
102932); line numbers are of the extracted text, quoted verbatim.

## SOURCES READ IN FULL
1 paper PDF · 2 `1fb91bd:datasets/abideDataset.py` (128 ln) · 3 `1fb91bd:agcl_ABIDE.py`
(324 ln) · 4 `1fb91bd:unsupervised/learning/ginfominmax.py` (59 ln) ·
5 `1fb91bd:unsupervised/embedding_evaluation.py` · 6 `1fb91bd:unsupervised/convs/wgin_conv.py` ·
7 `datasets/abideDataset.py` (HEAD) · 8 `agcl_ABIDE.py` (HEAD) ·
9 `audit/s11/s11_core.py` · 10 `audit/s12a1/scripts/s12a1_core.py` ·
11 `audit/s12a5/scripts/s12a5_core.py` · 12 `audit/s12a5/scripts/w_wave1.py`

## MASTER TABLE
| # | CHOICE | PAPER SAYS | ORIGINAL CODE DOES | WE DO | VERDICT |
|---|---|---|---|---|---|
| L1 | node features | 3 ALFF bands, `X = {x_v ∈ R^3}` (paper:260-262,268) | `x = nf['alff_value_cache']` only (1fb91bd:abideDataset.py:61,100); `num_dataset_features=3` (1fb91bd:agcl_ABIDE.py:51,57) | same: `num_dataset_features=3` (agcl_ABIDE.py:99) | **C** |
| L2 | evaluated repr | not specified | `model.encoder` → h (1fb91bd:agcl_ABIDE.py:72,192) | default `--eval_representation z` (agcl_ABIDE.py:437) | **DISPUTED** |
| L3 | augmented view | Bernoulli edge-drop mask (paper:281-283) | mask passed AS edge_weight (1fb91bd:agcl_ABIDE.py:129,172) | fork multiplies `FC * mask` (audit S8 'C') | **DISPUTED** |
| L4 | regulariser sign | not specified | `reg = mean(1 - mask)` = mean DROP; `view_loss = infonce - λ·reg` (1fb91bd:agcl_ABIDE.py:133,141,149) | `budget` = mean(KEEP) is the original's equivalent | **C** |
| L5 | graph topology | "adjacency matrix is initialized with all 1's" (paper:271-272) → COMPLETE | `if 'positive' in file` (:52,93) + `.nonzero()` (:73,110) → SPARSE, subject-specific | DENSE signed, 8100 edges, identical topology (abideDataset.py:96-101) | **PAPER≠CODE** |
| L6 | ALFF scaling | joint min-max over "all the 3 channels" (paper:272-274) | RAW, only `nan_to_num` (1fb91bd:abideDataset.py:61-63) | HEAD: PER-BAND min-max (abideDataset.py:71-74). AUDIT M1_B: JOINT min-max (s12a1_core.py:41) | **3-WAY SPLIT** |
| L7 | labels | not specified | ASD `y=0`, HC `y=1` (1fb91bd:abideDataset.py:77,114) | ASD `label=1`, NC `label=0` (abideDataset.py:121-122) | **INVERTED** |
| L8 | folds | "5-fold cross-validation" (paper:1473-1474) | `KFold(n_splits=10, shuffle=True, random_state=None)` (1fb91bd:embedding_evaluation.py:226,241) | frozen `StratifiedKFold` 5-fold, fixed seed (s11_core.py:93) | **PAPER≠CODE** |
| L9 | self-loop | A initialized all 1's incl. diagonal (paper:271-272) | `(1+eps)*x_r` added (1fb91bd:wgin_conv.py:42); diag present via `.nonzero()` when FC[i,i]≠0 | diag kept at 1.0 AND `(1+eps)*x_r` → own features counted TWICE | **C (logged fork)** |
| L10 | edge normalisation | "/ maximum of absolute values" → [-1,1] (paper:275-276) | NOT done | HEAD divides (abideDataset.py:85-87); AUDIT does not (s12a1_core.py:46,50) | **no-op, verified** |
| L11 | proj head dim | not specified | `GInfoMinMax(enc, args.emb_dim=32)` POSITIONAL (1fb91bd:agcl_ABIDE.py:50-53) vs class default 300 (ginfominmax.py:6) | same positional 32 | **C** |
| L12 | best-epoch reporting | "5-fold CV" mean±std (paper:1473) | `best_test_epoch = argmax(test_curve)`, reported (1fb91bd:agcl_ABIDE.py:254,277) | audit never selects on test | **ORIGINAL SELECTS ON TEST** |

## MANDATORY ROW VERDICTS

**L1 NODE FEATURES — CONFIRMED.** Paper: "3 bands of ALFF as node features and
Pearson's correlation coefficients (PCC) ... as edge weights" (paper:48-49); "X =
{x_v ∈ R^3}" (paper:268). Code: the ONLY assignment to `x` is
`x = nf['alff_value_cache']` (1fb91bd:abideDataset.py:61 ASD, :100 HC); FC enters
solely as `edge_weight` (:70,76,107,113). Nothing else reaches the nodes.

**L2 EVALUATED REPRESENTATION — CONFIRMED, and we differ.** Original passes
`model.encoder` (1fb91bd:agcl_ABIDE.py:72 and :192) → `encoder.get_embeddings`
(1fb91bd:embedding_evaluation.py:28) → `TUEncoder.forward` returns `xpool` = **h**.
The projection head is applied only in `GInfoMinMax.forward`
(1fb91bd:ginfominmax.py:26-30), which evaluation never calls. Ours defaults to
`z` (agcl_ABIDE.py:437), which is trained ONLY by InfoNCE.

**L3 AUGMENTED VIEW — CONFIRMED, and we differ.** Original:
`x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight)`
(1fb91bd:agcl_ABIDE.py:129 view half, :172 model half). `batch_aug_edge_weight` is
`sigmoid(gate_inputs)` (:127,170) — the MASK ALONE. FC is never multiplied back in.

**L4 REGULARISER SIGN — CONFIRMED.** `edge_drop_out_prob = 1 - batch_aug_edge_weight`
(:133); `reg` = per-graph mean of that (:136,141,147); `view_loss = calc_loss - λ·reg`
(:149); ascent via `(-view_loss).backward()` (:153). Maximising `infonce - λ·mean(DROP)`
minimises the drop probability, i.e. **penalises dropping**. A `mean(KEEP)` with the
same minus sign penalises KEEPING and collapses — which is what our `paper_keep`
did. Our `budget` mode is the mathematically equivalent form of the original.

**L5 GRAPH — CONFIRMED, and it is the largest gap.** Original selects only files
whose NAME contains `'positive'` (1fb91bd:abideDataset.py:52,93) — a pre-thresholded
positive-only matrix — then `edge_index.nonzero(as_tuple=False).t()` (:73,110) with
`edge_weight` taken from `coo_matrix(...).data` (:69-70,106-107). Result: **sparse,
subject-specific topology**. Ours builds the complete 90×90 grid, all 8100 directed
edges, identical for every subject (abideDataset.py:96-101). NOTE the paper sides
with US ("adjacency matrix is initialized with all 1's", paper:271-272), so this is
a PAPER-vs-RELEASED-CODE conflict, not simply our deviation. **A2 has never been run.**

**L6 ALFF SCALING — 3-WAY SPLIT (new finding, not in the commission).**
Paper: normalise "by subtracting the minimum from all the 3 channels and dividing by
the difference between the maximum and the minimum" (paper:272-274) = **JOINT**
min-max. Original code: **RAW**, only `nan_to_num` (1fb91bd:abideDataset.py:61-63).
Our HEAD loader: **PER-BAND** (`x.min(dim=0)`, abideDataset.py:71-74). Our frozen
AUDIT tensor M1_B: **JOINT** (`M1.min((1,2))`, s12a1_core.py:41) — i.e. the audit
data matches the PAPER, while the production loader matches neither. The commission
states "Ours uses per-subject min-max"; that is true of the HEAD loader but the
S16 experiments will read the frozen audit tensor, which is joint. The C6 ablation
must therefore test **raw vs per-band vs joint vs z-across**, not three variants.

**L7 LABELS — CONFIRMED INVERTED.** Original ASD `y=0`, HC `y=1`
(1fb91bd:abideDataset.py:77,114). Ours ASD `label=1`, NC `label=0`
(abideDataset.py:121-122). AUC is invariant; **sensitivity and specificity are
swapped** relative to the paper's Table 2. Every such column must be labelled.

**L8 FOLDS — CONFIRMED, and paper and code disagree.** Code:
`KFold(n_splits=10, shuffle=True, random_state=None)`
(1fb91bd:embedding_evaluation.py:241, default `folds=10` at :226) plus
`train_test_split(..., random_state=None)` (:246). Both unseeded, and
`kf_embedding_evaluation` is called at every `eval_interval`
(1fb91bd:agcl_ABIDE.py:190-192), so **folds are re-drawn at each of the ~20
evaluations**. It is `KFold`, NOT stratified. The paper says 5-fold (paper:1474).

**L9 SELF-LOOP — CONFIRMED (logged fork, not to be fixed).**
`out += (1 + self.eps) * x_r` with `eps=0`, non-trainable
(1fb91bd:wgin_conv.py:19,23,42). Our edge set includes (i,i) with FC[i,i]=1.0, so a
node's own features are counted TWICE. S12A5 ran with this; comparability requires
keeping it.

## TWO ADDITIONAL FINDINGS (not commissioned, recorded because they bear on S16)

**L12 — the original selects the reported score ON TEST.**
`best_test_epoch = np.argmax(np.array(test_curve))` (1fb91bd:agcl_ABIDE.py:254) and
the reported `BestTestScore` uses it (:277). Combined with L8's re-randomised folds,
the headline is a maximum over ~20 re-drawn evaluations. S12B measured this
inflation directly at **+0.044 ± 0.006**.

**The original's headline metric is ACCURACY, not AUC.** `eval_metric = 'accuracy'`
(1fb91bd:abideDataset.py:18); paper reports 80.65% accuracy on ABIDE I / AAL1
(paper:1478-1480) over 5 metrics. Our audit's primary number is AUC. **The paper's
80.65% and our 0.7565 are not the same quantity** and must never be compared directly.

## UNVERIFIED — I DID NOT READ THESE
- `1fb91bd:unsupervised/encoder/tu_encoder.py` — not on the C0 list; the HEAD version
  was read in earlier stages but I did not diff it against 1fb91bd in this session.
- `1fb91bd:unsupervised/view_learner.py` — not on the C0 list.
- The subject-matching loop at 1fb91bd:abideDataset.py:58-63 has no `break`, so `x`
  becomes the LAST filename whose digits substring-match. I observed this but did not
  quantify how many subjects it mispairs; that would need the original raw filenames.

## PLAIN ENGLISH
I read the published paper and the authors' original released code side by side with
our own version, and wrote down every place they differ. The most important findings
are these. First, the paper's method really does put only three ALFF numbers on each
brain region, with connectivity used only as edge weights — so our earlier low scores
for that setup were testing the right thing. Second, the authors' actual code builds
a much sparser brain graph than either the paper describes or we have ever used, and
nobody has tested that version — it is the biggest untested gap. Third, there are
three different recipes in circulation for scaling the ALFF numbers (the paper's, the
code's, and ours), so that ablation needs four variants rather than three. Fourth,
the authors' code picks its reported score by looking at the test set and re-shuffles
its data splits every time it measures, which inflates results. Finally, the paper's
headline 80.65% is an accuracy figure, not the AUC we have been measuring, so the two
numbers cannot be compared.

**C0 complete. Please read audit/s16/LEDGER.md. Type GO C1.**
