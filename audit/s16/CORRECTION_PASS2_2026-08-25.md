# S16 Final Pre-Submission Correction — Pass 2

**Baseline** `bd0a0b45ee45e4512ce1219d848628c2d36273d0`, branch `s16-verify`,
clean tree, 0 jobs in queue.
**Scope** fourteen phases P1-P14. Fifteen defects recorded as D32-D46 in
`DEFECTS_FOUND.md`.

## What was NOT done

- No cluster job was submitted, cancelled or requeued.
- C2, E2E and C6 were not run. No scientific computation was performed.
- No production code outside `audit/s16/`, no cache, no prior evidence was modified.
- No commit was amended and nothing was force-pushed.
- The one surviving prior E2E artifact (`main_A7_signed_plain_s0__lab0`, 2026-08-25
  01:58) was read but not altered or deleted.

## Frozen identities, re-verified after every change

| invariant | value |
|---|---|
| expected ledger | 159 units x 9 folds = **1,431 cells**, hash `8587b1ca36553408` |
| PROD policy hash | `798ed7790c1ddabc` (400 epochs, folds 3/3/3) |
| E2E policy hash | `9971b85a2cd0b449` (4 epochs, folds 1/0/0) |
| TEST policy hash | `0053ac4f238a9b8f` |
| contracted fields | 28 MATCH_KEYS; `worktree_clean` is a GUARD, never compared (D27) |

## The two findings that would have destroyed the study

**D35 — 810 of 1,431 cells.** The collector loaded the `signed` cache once and
validated every unit against it. Every `abs`, `pos_zero` and `shift` cell — the
entire non-signed two thirds of the four-level E factor — carried the wrong `h_fc`
and `cache_file` and would have been rejected as a provenance failure. A real
159-unit fixture with distinct per-E hashes now collects all 1,431 cells; replaying
the old binding rejects exactly 810, every one an `h_fc`/`cache_file` mismatch.

**D37 — the report could never run.** `refusals()` unpacked four values from a
function returning three, so `s16_report.py` raised `ValueError` on its first
statement regardless of input. No headline could ever have been produced.

## Pre-registered decision bands

Declared 2026-08-25, **before any C6 result exists** (no C6 run has ever produced
a result):

- **C-PERM**: protocol-level mean AUC must lie in **[0.45, 0.55]**. An AUC of
  exactly 0.500 PASSES — that is the expected outcome for permuted labels, and
  treating it as a failure was defect D9.
- **shift vs signed**: for BNT the identity is exact, so the **paired** difference
  must lie within **+/-0.01**. A3 is excluded because its node features ARE the FC
  rows, so E rewrites the inputs rather than only the graph weights.

Both are now hard gates that exit 4. They were previously printed and ignored.

## Validity pass rule, stated explicitly (D38)

- decision level: `(arm, E, mode, fold_protocol)` — one verdict per estimand.
- statistic: average across the folds of a seed first, then across seeds.
- C-RAND reference: same architecture AND same protocol, same aggregation.
- INTERPRETABLE requires movement_max > 0.10, clip_rate < 0.30 and
  (auc - crand) >= 0.03; otherwise UNTRAINED, or DESCRIPTIVE where no admissible
  reference exists (A7, per `CRAND_MAPPING.md`).

lab, site and loso are three separate estimands and are never pooled. LOSO runs
about 0.044 below the ordinary protocols, so a pooled C-RAND reference silently
imported that shift into every verdict.

## Single source of truth (D34)

`P.contract_fields()` builds the 28 contracted fields in one place. The worker's
resume expectation, `build_manifest()` and the collector's expectation all call it,
and all derive the epoch and optimizer blocks from the same `ExecPolicy` that drives
training. `ExecPolicy.train_consts()` and `P.model_cfg()` are likewise single owners.
Reading the one real pre-refactor E2E manifest confirmed the diagnosis exactly: the
manifest already carried `batch`, so the divergence was worker-side only.

That same manifest also shows `worktree_clean: False` — it was produced from a dirty
tree and is correctly refused by the D27 guard — and its result record carries no
explicit `namespace`, which the D34 change now rejects. Both are the intended
behaviours for a 4-epoch artifact that must never enter a PROD collection.

## Requeue truthfulness (D45)

`--signal=B:USR1@300` delivers to the batch shell, not the python child, so the
worker's SIGUSR1 handler could never fire; and `sys.exit(0)` is a successful exit,
which SLURM never requeues. Status `requeued` was therefore a false claim on a code
path that was unreachable anyway. The requeue is now explicit, its real outcome is
recorded including `scontrol`'s own return code and stderr, a requeue that did not
happen is reported as `stopped_not_requeued`, and the launchers forward the signal
to the child and propagate its exit status. Verified with a mocked `scontrol` and a
live forwarding harness; no real job was touched.

## Blocker for submission — disk quota

`/users` is over quota: **270G against a 250G soft quota and a 300G hard limit**,
4 days of grace remaining, and 1,427k files against a 1,500k soft limit. Writes were
failing silently mid-fixture during this pass. A PROD wave writes ~1,431 sealed
bundles at roughly 16 MB each, about **23 GB**, against ~30 GB of headroom. The space
is dominated by unrelated projects, not by this audit. **This must be resolved before
the wave is submitted.**

## Evidence

| suite | rc | PASS | FAIL |
|---|---|---|---|
| `test_pass2_p1p2.py` (P1, P2) | 0 | 26 | 0 |
| `test_pass2_p3.py` (P3) | 0 | 18 | 0 |
| `test_pass2_p4.py` (P4) | 0 | 7 | 0 |
| `test_pass2_p6789.py` (P6-P9) | 0 | 23 | 0 |
| `test_pass2_p11.py` (P11) | 0 | 11 | 0 |
| `test_pass2_p13.py` (P13) | 0 | 19 | 0 |
| `test_gate2.py` | 0 | 14 | 0 |
| `test_gate3.py` | 0 | 26 | 0 |
| `test_gate5.py` | 0 | 17 | 0 |
| `test_gate6.py` | 0 | 16 | 0 |
| `test_gate7.py` | 0 | 18 | 0 |
| `test_gate8.py` | 0 | 26 | 0 |
| `test_final.py` | 0 | 50 | 0 |
| `test_gate4.py` | 0 | retired stub, prints its supersession | - |

**Total: 271 assertions, 0 failures.** `test_pass2_p4.py` and `test_final.py` each
build and collect a full 1,431-cell fixture through the real collector; `test_final`
reports `CLEAN AND COMPLETE [test]: 1431 cells, 6291 rows`.

D46 is the defect this evidence run itself caught: the P3 fix initially duplicated
`policy_hash` in the worker's result record — `dict() got multiple values for keyword
argument`, the exact failure class that killed all 288 S15 units. `test_gate3.py`
caught it before any submission. That is the E2E-gate lesson working as intended.

## Status

Not submitted. Awaiting independent review and explicit submission authorization.
