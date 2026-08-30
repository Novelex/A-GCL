# S17 SWEEP — DEVIATION 01: LOSO full-grid -> LOSO top-K

**Written 2026-08-30 03:10 BST, BEFORE any Stage-B aggregate existed.**

## Proof of pre-registration
At the moment of writing:
- `aggBC` (job 1876183) state = PENDING, never started.
- No `agg*.json` and no `report*` file exists anywhere under `runs/sweep/`.
- Stage-B units on disk: LAB/SITE only (units 0-1439 block, partially complete).
  Zero LOSO units (1440-4175) had been written.
Therefore the rule below was fixed with no knowledge of any Stage-B result.

## What is being changed
Stage B as submitted evaluates the full `grid_B()` (540 configs) on all 29 outer
folds of all three protocols. LOSO accounts for 2736 of the 4176 units (66%) and
~16 h of the ~18.5 h remaining wall clock.

LOSO full-grid evaluation is replaced by **LOSO top-K evaluation**:

> **Design: select on LAB+SITE, evaluate on LOSO.**

## The selection rule (FIXED — no discretion after results)
1. Aggregate Stage-B LAB and SITE units only.
2. For each config `c` in `grid_B()`, compute `s(c)` = the mean **inner-fold** AUC
   of `c`, pooled over every (protocol, fold, input) cell in {lab, site}.
   Inner folds only. Outer-fold AUC is NEVER used for selection.
3. Rank configs by `s(c)` descending. Ties broken by **catalogue order**
   (ascending index in `grid_B()`), identical to the existing aggregator.
4. **K = 30.** Take the top 30 configs. K is fixed here and is not revisable.
5. Evaluate exactly those 30 configs on all **19** LOSO folds and all **8**
   frozen inputs (`B/inputs.json`, already frozen by aggA).
   Units = 19 x 8 x 1 chunk = 152.

K=30 was chosen to equal the existing `CHUNK` size so the new wave reuses the
per-unit contract byte-for-byte. It is not tuned.

## What is NOT changed
- LAB and SITE remain **full grid, 540 configs**. Tasks 0-287 run to completion
  untouched. No LAB/SITE unit is cancelled, re-run, or altered.
- Stage A, aggA, and `B/inputs.json` are untouched and already complete.
- Stage C (job 1876182) is untouched and runs to completion.
- The LAB gate (`gate_flat_linsvm_s55grid` must reproduce 0.7565) is unchanged
  and still blocks the report (`exit 5` on failure).
- Frozen tensors, S16, and all prior evidence are untouched.
- All fold definitions, the per-edge z-score, and the PROD optimiser recipe are
  unchanged.

## Honest statement of the cost
LOSO is now a **confirmatory** arm, not an exploratory one. It answers
"do the configs selected on LAB+SITE transfer across sites?" It can no longer
answer "which config is best on LOSO?", because 510 of 540 configs are never
evaluated there. Any LOSO number in the report must be read as
select-on-LAB/SITE-evaluate-on-LOSO, and must be labelled as such.

This is a strictly weaker claim than the original design. It is also a less
overfit-prone one: the original would have selected among 540 configs using
LOSO inner folds, which is a wider selection surface than the confirmatory
design used here.

## Operational consequences (all handled, none silent)
- `scancel 1876181_[288-999]` cancels only PENDING LOSO tasks. Verified at
  03:10 that RUNNING indices span 71-231, i.e. entirely LAB/SITE.
- Cancelled array elements make `afterok:1876181_*` unsatisfiable, so the
  pending `final` job (1876183) is cancelled too and re-submitted without that
  dependency once the top-K wave completes.
- `report.py:13` and `aggBC.py:81` iterate `L.protocols()` including `loso`.
  They consume the top-K LOSO units, which are written in the same on-disk
  contract, so no aggregator change is required beyond pointing at them.

---

# AMENDMENT A1 — 2026-08-30 03:35 BST

## Correction to "Operational consequences"
The original note claimed:

> "They consume the top-K LOSO units, which are written in the same on-disk
>  contract, so no aggregator change is required beyond pointing at them."

**That was wrong.** `aggBC.load_stage()` enforced
`if set(r["inner"]) != set(CFG_B): rows.pop(key)` — a row carrying only 30 of the
540 configs is treated as ABSENT. Every top-K LOSO row would have been silently
dropped, LOSO would have reported `complete: False`, the blend would have been
skipped, and `aggBC` would have exited 4. The failure would have been silent in
the sense that no LOSO number would appear at all, rather than a wrong one.

## The aggregator change actually made
`aggBC.py`, three marked hunks, backed up to
`agcl_audit_s0/s17/sweep/aggBC.py.pre_dev01`:

1. New `expected_grid(stage, p)`. Returns the frozen `B/TOPK.json` config set for
   `(B, loso)` only; the full 540-config grid for every other (stage, protocol).
2. `load_stage()` calls `expected_grid(stage, p)` instead of hard-coding `CFG_B`.
3. `main()` stamps the LOSO report block with `B_selection`, `B_estimand`, and
   `B_topk_cutoff_mean_inner`, so a top-K LOSO number can never be read as if it
   came from a 540-config search.

No other protocol, stage, gate, or threshold is affected. LAB and SITE still
require the full grid and are popped if incomplete, exactly as before.

## Where K=30 is enforced
- `selK.py` sets `K = 30` as a module constant and writes `B/TOPK.json` exactly
  once, refusing to overwrite an existing file.
- `sb_D.sh` refuses to launch unless `TOPK.json` has `K == 30` and 30 unique
  configs, and unless `inputs.json` has exactly 8 inputs.
- `taskD.py` and `aggBC.expected_grid()` assert the `DEVIATION_01` tag and
  `len(configs) == K`. They deliberately do NOT hard-code 30, so the identical
  code path can be smoke-tested with a small K; production strictness comes from
  `selK.py` and `sb_D.sh` above.

## Also corrected
`selK.py` refuses to run unless LAB and SITE are BOTH complete for all
(fold, input) cells with a full 540-config grid. A straggler task therefore
cannot shrink the pool the top-K is drawn from.

---

# AMENDMENT A2 — 2026-08-30 15:40 BST — SELECTION RULE CORRECTED BEFORE USE

## The defect
A1's rule pooled every health-valid inner run across all lab+site cells and took
the mean. Run on the complete LAB+SITE data it produced a top-30 dominated by
configs that almost never train successfully:

    w512_d2_p0.3_lr0.0001_wd0.01   inner=0.7537   n=9   valid inner runs of 400
    w128_d1_p0.0_lr0.0001_wd0.01   inner=0.7372   n=4   valid inner runs of 400

A config surviving 4 of 400 inner runs was outranking one surviving all 400. The
pooled mean rewards configs whose few successful runs happen to be lucky, because
`n` is not held constant across configs. This is a small-sample selection
artifact, not a performance signal.

That TOPK.json was NEVER used: no Stage-D task was submitted against it. It is
quarantined unmodified at
`agcl_audit_s0/s17/sweep/TOPK.REJECTED_A1.json` as evidence.

## The corrected rule (A2) — supersedes rule step 2 of the original note
The rule now reuses `aggBC`'s OWN pre-existing health definitions verbatim
rather than any new criterion invented here:

1. A cell `(protocol, fold, input)` is ELIGIBLE for config `c` iff
   `aggBC.run_valid(row, c)` — the OUTER refit passes health AND at least
   `MIN_VALID_INNER = 3` of the 5 inner trainings pass health.
2. `c` is ADMISSIBLE iff it is eligible in **all 80** lab+site cells
   (10 folds x 8 inputs). This holds `n` constant across every ranked config.
3. `s(c)` = mean over those 80 cells of `aggBC.inner_score(cell, c)`, the mean
   inner AUC over the health-valid inner runs of that cell. Inner folds only;
   outer-fold AUC is never read.
4. Rank ADMISSIBLE configs by `s(c)` descending, ties by grid_B catalogue order.
5. **K = 30, unchanged.**

## Feasibility, measured before the rule was fixed
Counts only were inspected — no config names, no scores:

    eligible-cell count per config: min=0  median=74  max=80
    eligible in ALL 80 cells : 150 configs
    eligible in 0 cells      :  57 configs

150 admissible configs is a 5x pool for K=30, so the strict all-80 requirement
costs nothing in search breadth.

## Why this is not rule-shopping
The A1 rule was not abandoned because its answer was unwelcome; it was abandoned
because `n` ranged from 4 to 400 across the configs it ranked, which makes its
means incomparable by construction. The replacement introduces no new threshold:
`run_valid`, `MIN_VALID_INNER = 3`, and `inner_score` are all pre-existing
`aggBC` definitions used unchanged throughout Stage A and Stage B. The
all-cells requirement mirrors `aggBC`'s existing optimistic arm, which already
skips any config not valid across every fold of an input.

---

# AMENDMENT A3 — 2026-08-30 15:55 BST — A2 ALSO REJECTED, RULE CORRECTED AGAIN

## The defect in A2
A2 required a config to be health-eligible in ALL 80 lab+site cells. Applied to
the complete data it selected only narrow networks (widths 16/32/64, mean inner
0.7094-0.7207) and excluded **every** config that actually wins a fold:

    per-fold honest winners also present in the A2 top-30: 0 / 10

The ten winners and their eligibility:

    lab0  w512_d2_p0.5_lr0.001_wd0.01      inner 0.7600   67/80 cells
    lab1  w512_d2_p0.3_lr0.001_wd0.001     inner 0.7692   71/80
    lab2  w128_d1_p0.2_lr0.0003_wd0.001    inner 0.7589   69/80
    lab3  w512_d2_p0.3_lr0.001_wd0.0001    inner 0.7459   70/80
    lab4  w512_d2_p0.2_lr0.001_wd0.0001    inner 0.7602   67/80
    site0 w256_d2_p0.1_lr0.0003_wd0.0001   inner 0.7307   62/80
    site1 w512_d2_p0.0_lr0.001_wd0.001     inner 0.7838   61/80
    site2 w256_d1_p0.5_lr0.0001_wd0.01     inner 0.7698   23/80
    site3 w512_d2_p0.2_lr0.0003_wd0.01     inner 0.7970   31/80
    site4 w512_d1_p0.1_lr0.001_wd0.01      inner 0.7532   60/80

Wide networks fail the health rule in 10-20% of cells (more for two of them), so
an all-cells requirement selects for RELIABILITY and discards PERFORMANCE. A
"select on LAB+SITE, evaluate on LOSO" design whose selected set contains none of
the configs the selection procedure actually chose is not that design at all.

Stage D job 1877798 was submitted against the A2 set and ran for ~5 minutes. It
was cancelled before writing any unit (0 LOSO units on disk, verified) and the
LOSO tree was removed. LAB/SITE units (720+720) were never touched. The A2
TOPK.json is quarantined at `agcl_audit_s0/s17/sweep/TOPK.REJECTED_A2.json`.

## The corrected rule (A3) — supersedes A2
`TOPK` is the union of two parts, capped at K = 30:

**(i) The honest per-fold winners, unconditionally.** Every config selected by
`aggBC.nested_B` for any outer fold of LAB or SITE is included. These are the
literal output of the pre-existing honest nested procedure; no eligibility
threshold is applied to them, because the procedure already selected them under
its own `run_valid` rule in the cell where it chose them. This is 10 configs.

**(ii) Padding to K by performance.** The remaining 20 slots go to the highest
`mean inner_score over eligible cells` among configs eligible in at least
**40 of 80** cells (a simple majority), excluding those already in (i). Ties by
grid_B catalogue order.

Inner folds only throughout; outer-fold AUC is never read. K = 30, unchanged.

## Why the majority threshold is not rule-shopping
The threshold in (ii) cannot change whether any winner is included: winners enter
through (i) regardless of their eligibility count. (ii) governs only the padding,
and "a simple majority of cells" is a threshold chosen for being unarguable
rather than for its effect on any particular config. A config eligible in 40
cells has at least 120 health-valid inner runs, which controls the small-sample
artifact that killed A1 without reintroducing the robustness filter that killed A2.

## Residual limitation, stated plainly
Configs in the padding set still have differing `n` (40 to 80 cells), so their
mean inner scores are not perfectly comparable to one another. This affects only
which configs fill slots 11-30; it cannot affect the presence of the ten configs
the honest procedure actually selected.

## Rule-change log
- A1 pooled all valid inner runs -> ranked on n from 4 to 400. REJECTED, unused.
- A2 required eligibility in all 80 cells -> excluded 10/10 winners. REJECTED,
  submitted then cancelled before any output was written.
- A3 winners unconditionally + majority-eligible padding. IN USE.

---

# NOTE N1 — 2026-08-30 18:45 BST — final job 1877861 failed, fixed, resubmitted

`aggBC.main()` stamped `rep["B_topk_cutoff_mean_inner"]` by indexing
`TOPK.json["cutoff_mean_inner"]`. That key existed under rules A1 and A2 but the
A3 rewrite of `selK.py` replaced it with a winners/padding structure, so the key
is absent. `aggBC` raised `KeyError: 'cutoff_mean_inner'` on the LOSO block after
LAB and SITE had already aggregated successfully. Exit 1, no AGG_BC.json written.

This was a defect in the stamping code only. No gate, threshold, health rule or
selection rule was involved, and no result was affected: the crash occurred while
recording provenance metadata, after the LOSO numbers had been computed.

Fix: read the TOPK provenance keys with `.get()` and additionally stamp
`amendment`, `n_winners`, `min_cells_for_padding` and `rule`. The frozen
`TOPK.json` was NOT modified. Resubmitted as a fresh final job.
