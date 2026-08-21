# R0 EXECUTION REPAIRS — minimal, execution-only (Phase 1 freeze)
Rule: every repair makes the released procedure RUNNABLE; none may alter training tensors,
predictions, split generation or metric values. Evidence class noted per repair.

R1. Complete the malformed evaluator call (line 256).
    Failure without: SyntaxError, module unimportable.
    Evidence: kf bytecode call sequence — embedding_evaluation(encoder, train_loader,
    valid_loader, test_loader, flag).
    Patch: close the call with exactly those arguments.
    Proof of neutrality: restores the argument list the working binary used; no numeric change.
R2. Define `test_id = []` before the fold loop.
    Failure without: NameError at line 243. Evidence: STORE_FAST test_id in bytecode.
    Neutrality: test_id is only appended to; never read for computation.
R3. Restore the 15-value return contract: compute fpr,tpr = roc_curve-derived arrays? —
    AMBIGUITY GUARD: the bytecode proves roc_curve was called and fpr/tpr returned, but the
    exact inputs (which split's scores) are not recoverable with certainty. fpr/tpr are
    NEVER USED downstream in kf (only unpacked). DECISION: return placeholder empty arrays
    for fpr/tpr, preserving the 15-slot contract. This CANNOT change accuracy: the two
    slots are write-only at the call site. (Two plausible roc_curve reconstructions exist,
    but both are dead values — so this is not an accuracy-affecting ambiguity.)
R4. `running_time = 0.0` when flag=False (timing-only slot).
    Failure without: NameError on the pretraining evaluation. Evidence: bytecode shows the
    same conditional store — the bug is genuine in both artifacts. Neutrality: value feeds
    only a printed mean of timings.
R5. Invoke with `--batch_size 32` while documenting the README's `--batch-size` typo.
    Failure without: argparse SystemExit. Neutrality: same value the README intends.
R6. Dependency-compatibility shims only: none required beyond MPLBACKEND=Agg (headless
    matplotlib import) on this environment. torch/PyG API usage in the O-contract path is
    already compatible (proven by the S8 parity work). No other shim applied.
No other change is permitted or made. No repair touches tensors, RNG, splits or metrics.
