# Corrections Log — against `docs/novelex_agcl_fork_audit.md`

This document tracks what was changed, what was deliberately left unchanged (and why), and
what is still open, in response to the pre-submission audit in
`docs/novelex_agcl_fork_audit.md`. Section numbers below match that document's numbering.

## Section 1 — carried over from the prior audit (`docs/agcl_official_repo_audit.md`)

The novelex audit verified these were already fixed correctly before it ran. Kept here for a
complete record.

| Issue | Fix | Where |
|---|---|---|
| `SyntaxError`, unterminated call | Call restored, return arity matched | `unsupervised/embedding_evaluation.py` |
| `TUEvaluator` not exported | Re-export added | `datasets/__init__.py` |
| `from scipy import interp` (removed API, unused) | `from numpy import interp` | `agcl_ABIDE.py`, `agcl_ABIDE_queue.py`, `unsupervised/embedding_evaluation.py` |
| `running_time` unbound when `flag=False` | Timing moved out of the `if flag:` guard | `unsupervised/embedding_evaluation.py` |
| `test_id` `NameError` | Dead line deleted | `unsupervised/embedding_evaluation.py` |
| Eval discarded edge weights | `self.forward(batch, x, edge_index, None, edge_weight)` | `unsupervised/encoder/tu_encoder.py:98` |
| Subject↔feature filename collision | Exact `_adj.mat` → `_nf.mat` swap, `isfile` guard (replaces fragile digit-substring matching) | `datasets/abideDataset.py` |
| Wrong `.mat` keys (`corr_each_sub`/`alff_value_cache` vs. actual `cropped_matrix`/`norm_matrix`) | Loader updated to real keys | `datasets/abideDataset.py` |
| Paper says 5-fold, code was 10 | `folds=5` | `unsupervised/embedding_evaluation.py` |
| §2.1 normalisation absent | Node min–max over all 3 channels; edge weights ÷ max\|w\| | `datasets/abideDataset.py` |
| AUC not computable (`LinearSVC` has no `predict_proba`) | `decision_function` → `roc_auc_score` | `unsupervised/embedding_evaluation.py` |
| Sensitivity/F1 were w.r.t. controls, not ASD | `ASD=1, NC=0` | `datasets/abideDataset.py` |
| Memory-bank sign inverted (no real adversarial structure) | Model minimises `cr_loss`, view maximises | `agcl_ABIDE_queue.py` |

**Also fixed during this work, not flagged by either audit:** `unsupervised/convs/wgin_conv.py`'s
`# propagate_type:` comment only declared `x`, not `edge_weight`. Modern `torch_geometric`
(2.6.1) compiles `propagate()`'s accepted kwargs from that comment, so it was silently rejecting
`edge_weight` — `TypeError: propagate() got an unexpected keyword argument 'edge_weight'`. This
broke training regardless of device (confirmed on both CPU and GPU/L40S) and would have made the
`tu_encoder.py:98` fix above ineffective. Fixed by adding `edge_weight: OptTensor` to the
annotation and importing `OptTensor`.

## Section 2 — Blocker: the `reg` sign flip

**Status: reverted. Left matching upstream (`-`).**

The audit's own repair pass had flipped `view_loss = model.calc_loss(x, x_aug) - (reg_lambda * reg)`
to `+` in both `agcl_ABIDE.py` and `agcl_ABIDE_queue.py`, to match the paper's prose ("we want to
minimize R" / "force more edges to be dropped"). This audit found that flip removes the only term
that opposes edge-dropping in the view learner's adversarial objective — under gradient ascent,
`+reg_lambda*reg` and `calc_loss` both push toward dropping more edges, with no interior optimum.
This repo is a fork of AD-GCL, where this term functions as a perturbation *budget*, not an
encouragement to drop.

Confirmed empirically from our own smoke-test log: gates initialise at `sigmoid(0)=0.5`, so `reg`
starts at exactly `0.5`. Under the `+` version, `reg` climbed to `0.525` after one epoch — moving
*up*, i.e. toward dropping everything, exactly as predicted. The `-` version is also what the
paper's own default `--reg_lambda 2.0` was calibrated against (Fig. 6 sweep); carrying that value
over to a flipped sign transfers no comparability to Table 2.

**Change:** reverted `+` back to `-` in `agcl_ABIDE.py:155` and `agcl_ABIDE_queue.py:207`. The
separate memory-bank sign fix (`+ cr_lambda * cr_loss`, Section 1 above) was *not* reverted — this
audit confirmed that one is a genuine correction, not a paper-vs-code inconsistency.

## Section 3 — Blocker: memory-bank variant never executed

**Status: verified. No code change required.**

`agcl_ABIDE_queue.py` (memory bank, λ₂=0.4, bank length 256 — the paper's headline method) had
never been run, not even to see if it started. Two specific risks were flagged:

- `MemoryBank_Q.__init__` builds `self.memory` via `torch.zeros(..., requires_grad=True).to(device)`,
  which attaches a fixed, one-time autograd node (`ToCopyBackward`) to a tensor reused every
  iteration — a plausible "backward through the graph a second time" crash risk, since
  `cr_loss.backward()` traverses that same node on every step.
- Cold-start negatives: the bank's zero-initialised rows contribute `exp(0)=1` to the InfoNCE
  denominator until the queue fills.

Ran `agcl_ABIDE_queue.py` for 2 full epochs (CPU, ~8 minutes; chosen deliberately over 1 epoch so
the memory-bank backward path executes more than once). Completed cleanly — `cr_loss.backward()`
ran dozens of times through the memory bank with no error. The double-backward crash does not
manifest on this torch/environment combination. Log saved at
`logs/agcl-smoke-queue-cpu_20260808_1836.log`.

**No code change.** The cold-start negatives point is real but minor (self-corrects after
`max_length / batch_size` steps) and was left as-is by choice.

## Section 4 — Verify: the PCC diagonal

**Status: verified, left unchanged. Diagonal is retained.**

Confirmed on real data: `cropped_matrix`'s diagonal is 1.0 (self-correlation), and it is the
maximum absolute value in every subject's matrix (`max|w| == 1.0000` exactly, checked across 10
subjects). Two consequences were flagged: the edge-weight max-abs normalisation added for §2.1 is
a no-op (dividing by exactly 1.0 changes nothing), and each graph carries 90 redundant self-loop
edges on top of `WGINConv`'s own internal `(1 + eps) * x_r` self-term.

**Decision to leave as-is:** the paper's own Section 2.1 states the graph is denoted
`G = (V, A, X, E)` with *"the adjacency matrix is initialized with all 1's"* — an M×M matrix of
1's is by definition all-ones including the diagonal, i.e. the paper's own construction includes
self-loops from the start. Zeroing the diagonal would have been a deviation from the paper's
stated graph definition, not an alignment with it. A brief search of the broader FC-graph
literature also found no consensus either way (some works retain self-connections, some drop
them, some threshold-sparsify entirely) — so there was no external convention pulling toward
zeroing it either.

**Consequence to state in methods:** the edge-weight normalisation added for §2.1 is a no-op
given this diagonal, and node self-information is represented twice (once via the diagonal
self-loop, once via `WGINConv`'s built-in root term). This is a known, documented property of the
current setup, not an oversight.

## Section 5 — Verify: nothing counted subjects

**Status: fixed.**

`_load_class` silently `continue`s when a subject's NF file has no match, with no log of how many
subjects were dropped. Added:

- In `datasets/abideDataset.py`, `_load_class` now prints `"{adj_folder}: loaded {N} / {M} adj
  files"` and asserts the counts match before returning.
- In both `agcl_ABIDE.py` and `agcl_ABIDE_queue.py`, after dataset construction:
  `logging.info("N = %d (ASD = %d, NC = %d)", ...)`.

Verified against the real data: `ASD_ADJ: loaded 455 / 455 adj files`, `NC_ADJ: loaded 501 / 501
adj files`, `N = 956 (ASD = 455, NC = 501)` — no exclusion cascade in this dataset.
`data/processed/data.pt` regenerated to pick up the change.

## Section 6 — Batch-size-scaled printed losses

**Status: fixed.**

`model_loss_all` and `view_loss_all` accumulate `loss.item() * batch.num_graphs` per batch (a
sum), but were divided by `len(dataloader)` (number of *batches*, not graphs) when computing the
epoch summary — inflating the printed loss by roughly `batch_size×`. Confirmed against our own
smoke-test log: `Model Loss 117.22` at `batch_size=32` ÷ 32 ≈ 3.66, essentially the InfoNCE chance
floor `log(31) ≈ 3.43` for an untrained epoch-1 model — consistent with the bug, not a real
signal. Purely a logging artefact; `.backward()` itself uses the correctly-scoped per-batch loss,
so training was never affected — only the human-readable summary number was wrong, and would have
been misleading in any batch-size sweep.

**Change:** denominator corrected to `len(dataloader) * args.batch_size` (accounts for
`drop_last=True`) in both `agcl_ABIDE.py` and `agcl_ABIDE_queue.py`. `fin_reg` was already correct
(accumulated as a per-batch mean, not a sum) and was left unchanged.

## Section 7 — Evaluation protocol

**Status: left unchanged, by decision — reproducing the paper's own protocol, not a stricter one.**

Three properties confirmed present, all inherited from upstream:

- `KFold`/`train_test_split` use `random_state=None` — evaluation is not reproducible run-to-run
  even with a fixed `--seed`, since the seed is never threaded into these calls.
- `best_test_epoch` (and the per-metric equivalents, now including `best_auc_test_epoch`) is
  computed independently per metric via `argmax` directly on the *test* curve — the reported
  `BestTestScore` line mixes results from whichever epoch happened to score highest on each
  individual metric, not one coherent epoch. This is an optimistic epoch-selection bias (picking
  the best of many test-set evaluations directly), the same mechanism the first audit identified
  as one reason the paper's 80.65% headline number would not reproduce as an honest, single
  estimate.
- The encoder is self-supervised on the full dataset (test folds included) before any train/test
  split — transductive, standard for this evaluation protocol (InfoGraph/AD-GCL-style), not label
  leakage since no labels are used at that stage.

**Decision:** kept unchanged, since the goal here is to reproduce the paper's own method and
evaluation as published, not to substitute a more statistically rigorous protocol. Documented here
so the behaviour is explicit rather than silently inherited.

## Section 8 — Housekeeping

**Status: not yet addressed.**

- README still documents the upstream data format (`alff_value_cache`, `corr_each_sub`,
  `sub_0050002_aal3_all_positive.mat`) — our loader reads `norm_matrix`, `cropped_matrix`, and
  `*_adj.mat`/`*_nf.mat`. Needs updating.
- `data` is committed as a symlink to `/users/3171356m/sharedscratch/A-GCL/data`; `.gitignore` has
  `data/`, which does not match a symlink named `data`. `git rm --cached data` needed.
- Committed `.pyc` files — `__pycache__/` should be added to `.gitignore` and removed from the
  index.
- `from numpy import interp` remains in three files, still unused (dead import, harmless but
  should be deleted).
- `torch.load(path)` in `abideDataset.py:22` should pass `weights_only=False` explicitly — the
  current `FutureWarning` becomes a hard error on future torch versions once the default flips.

## Section 9 — Modelling choices to state explicitly

**Status: not code changes — to be written into the methods section.**

- **90 ROIs, not 116.** `cropped_matrix` confirms cropping was applied; the paper's AAL1 (116
  ROIs, cerebrum + cerebellum + vermis) is what produced the 80.65% headline number. Our 90-ROI
  data is a defensible but different atlas configuration and is not directly comparable to
  Table 2.
- **Per-subject min–max on node features**, rather than dataset-wide. This is the correct choice
  to avoid test-fold leakage into training statistics, but note the consequence: it removes
  between-subject amplitude information entirely, leaving node features as pure per-subject
  spatial pattern.
