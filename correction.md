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

**Status: reverted at the time, then superseded by a later, separate decision. See update below —
this section's original conclusion is no longer what the code does.**

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

**Change made at the time:** reverted `+` back to `-`, keeping `reg` defined as *drop*-probability
(`mean(1-mu)`). With that definition, `-reg_lambda*reg` is the correct budget direction: ascending
it pushes drop-probability down, i.e. keep-probability up, opposing `calc_loss`'s pressure to drop
more. The separate memory-bank sign fix (`+ cr_lambda * cr_loss`, Section 1 above) was *not*
reverted — that one is a genuine correction, not a paper-vs-code inconsistency.

**Update — this section is stale, corrected here:** a later commit (documented in
`docs/changes.md`'s "regularizer: paper-literal R(mu)" section) redefined `reg` from *drop*-probability
(`mean(1-mu)`) to the paper's own literal *keep*-probability notation, `R(mu) = mean(mu)`, while
leaving the surrounding `-reg_lambda*reg` sign unchanged. That combination — `-` paired with
`mean(mu)` instead of `mean(1-mu)` — is mathematically the *same* budget-breaking direction this
section originally reverted (`+reg_lambda*mean(1-mu)` and `-reg_lambda*mean(mu)` differ only by a
constant), just reached by changing the multiplied quantity instead of the sign in front of it. That
redefinition was a deliberate, explicit choice to follow the paper's own formula over the AD-GCL
budget reading — see `docs/changes.md` for the full reasoning — and it is what the code runs today.
**The collapse finding referenced in Section 10 below comes from that later decision, not from this
section** — this section's own conclusion (`-` sign, budget-preserving) was superseded, not extended,
by it. Kept here, corrected in place, rather than deleted, so the history of the reversal is legible.

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

## Section 10 — `training_profile`: paper_exact vs corrected, side-by-side

**Status: added, additive only. No prior fix in this document was reverted or altered.**

Every section above resolved a paper-vs-code question (regularizer sign, memory-bank timing/exclusion,
mask symmetry) by picking one convention and applying it globally. That worked as long as each question
came up once, but the same "paper's literal formula vs. the tested/corrected code" tension kept
resurfacing across independent pieces, and picking a single global answer meant any future re-litigation
would require touching already-verified code again. Resolved instead with a runtime switch:
`--training_profile paper_exact | corrected` on both `agcl_ABIDE.py` and `agcl_ABIDE_queue.py`.

**Nothing in Sections 1–9 above was reverted.** `corrected` (selectable explicitly, and the behaviour
`apply_training_profile` leaves untouched) is exactly the code this document already describes. `paper_exact`
is implemented as new, additive components selected via an `is_paper_exact` branch, not a rewrite of the
existing ones:

- `unsupervised/training_profiles.py` (new) — `PAPER_EXACT_OVERRIDES` forces the paper's Section 2.4
  hyperparameters, the paper-literal GIN ablation flags Section 9 above already exposed as
  toggles (`normalize_nodes`/`message_relu`/`post_bn_relu` → `False`), temperature 1.0 wherever a
  temperature is undocumented in the paper, `sym=False`, and `eval_representation='z'`.
- `sample_ordered_concrete_mask()` (`unsupervised/view_learner.py`, new) — literal per-directed-edge
  Concrete relaxation, independent noise per direction, no reverse-edge symmetrization. Added alongside
  `symmetrize_edge_logits`/`sample_symmetric_logistic_noise` (Bug 2's fix, still the `corrected` default).
- `PaperMemoryBank_Q` / `calc_regloss_paper()` (`agcl_ABIDE_queue.py`, new) — deliberately reproduce, under
  `paper_exact` only, the exact behaviours Section 1's audit table fixed as bugs: zero-filled cold-start
  queue with no validity mask, push-before-loss ordering, no same-subject exclusion. `MemoryBank_Q`/
  `calc_regloss` (the corrected versions) are unchanged and remain the default.
- `paper_five_fold_evaluation()` (`unsupervised/embedding_evaluation.py`, new) — plain (non-stratified)
  `KFold` directly on embeddings, no separate held-out test split, matching this document's Section 7
  description of the paper's own (non-reproducible, non-held-out) evaluation protocol. Used only for
  `paper_exact`'s final evaluation (`PaperFiveFoldScore`); `kf_embedding_evaluation` (fixed stratified
  splits, val-only checkpoint selection, held-out test) remains `corrected`'s final evaluation unchanged.
- Dataset preflight check — `paper_exact` logs a warning (not a hard error, by default) that this
  project's real data (956 subjects / 90 ROIs — Section 9 above) does not match the paper's own ABIDE-I
  configuration (987 subjects / 116 ROIs). `--allow_dataset_mismatch` defaults to `True` for exactly this
  reason: on our data, selecting `paper_exact` always tests the paper-literal *code path* faithfully, but
  its result is never an exact reproduction of the paper's own reported experiment. `--allow_dataset_mismatch
  false` hard-requires an exact match instead, for use against a properly-cropped 116-ROI AAL1 dataset.
- Notation correction: this project's regularizer (`docs/changes.md`'s "regularizer: paper-literal
  R(mu)" entry — see the corrected Section 2 above for how that decision superseded this section's
  original, different conclusion) is `R(mu) = mean(mu)`, the mean *keep*-probability. Some earlier
  internal notes wrote this as `R(f;B)` — the view-learner function `f` and sampled mask `B` as its
  argument — which is imprecise, since the quantity actually averaged is the deterministic
  keep-probability `mu = sigmoid(edge_logits)`, not a function of the sampled mask `B`. Corrected here:
  it is `R(mu)`, not `R(f;B)`.
- Keep-probability collapse under `paper_exact`'s regularizer remains the accepted, documented consequence
  of the printed objective established by that later decision (`docs/changes.md`'s "regularizer:
  paper-literal R(mu)" entry, *not* Section 2 above, which reached the opposite, budget-preserving
  conclusion before being superseded) — the profile system does not change that finding, only makes
  choosing to reproduce it an explicit opt-in rather than an accidental default now that
  `--training_profile` defaults to `corrected` (see the argparse default in `agcl_ABIDE.py`/
  `agcl_ABIDE_queue.py`, flipped from `paper_exact` for exactly this reason).
- **FIXED — a real budget `regularizer_mode` now exists.** `'budget'` (new default for `corrected`;
  `paper_exact` still forces `'paper_keep'` regardless, via `PAPER_EXACT_OVERRIDES`) uses the same
  `reg = R(mu) = mean(mu)` as `paper_keep`, but flips the sign it enters `view_loss` with: `calc_loss
  + reg_lambda*reg` instead of `calc_loss - reg_lambda*reg`. Ascending `+reg_lambda*mean(mu)` pushes
  keep-probability *up*, opposing the drop-pressure, instead of reinforcing it — mathematically the
  same budget mechanism the original Section 2 above restored (`-reg_lambda*mean(1-mu)`), reached via
  a sign flip on the added term rather than redefining `reg` back to drop-probability.
  Verified on real data, 6 epochs, submitted properly via SLURM this time (`scripts/verify_budget_regularizer.slurm`,
  job 1842531) — an earlier attempt to verify this directly in the interactive session was abandoned
  mid-run after it was found to be consuming 16 of the shared login node's cores via `GridSearchCV`'s
  `n_jobs=16`, a real HPC-etiquette mistake distinct from the fix itself. Result: KeepProb goes
  `0.4937 → 0.4939 → 0.4883 → 0.4475 → 0.4126 → 0.4327` — an ~12% relative decline over 6 epochs and,
  critically, **not monotonic** (epoch 6 reverses upward), versus `paper_keep`'s ~51% relative decline
  that never reverses. The reversal is the actual signature of an opposing force, not just a slower
  version of the same one-way collapse.
  **Not yet resolved:** this same 6-epoch run still selected epoch 0 (untrained) as best-on-validation
  — expected at this length (6 epochs is far too short for a real accuracy signal regardless of the
  regularizer), but it means this fix is verified to stop the collapse dynamic, not yet verified to
  produce a checkpoint that beats the untrained baseline over a full run. That requires an actual full
  200-epoch run to confirm.

**Also caught while wiring this in:** the `score_fmt` string used for `FinalFitScore`/`FinalTestScore`
mixed `str.format()`'s `{}` placeholders with a `logging`-style `%d` prefix, then passed everything to
`logging.info(msg, *args)` — `logging`'s lazy `msg % args` formatting only recognises the one real `%d`
directive, so either line would raise `TypeError: not all arguments converted during string formatting`
the moment it actually fired. Rewritten as a fully `%`-style string (`%.4f` per field). Pre-existing bug,
unrelated to the profile system itself, caught while adding the equivalent `PaperFiveFoldScore` log line.

files: `unsupervised/training_profiles.py` (new), `unsupervised/view_learner.py`,
`unsupervised/embedding_evaluation.py`, `agcl_ABIDE.py`, `agcl_ABIDE_queue.py`
