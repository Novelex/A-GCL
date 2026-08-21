# R0 SOURCE DISCREPANCIES — Phase 0 (read-only, all verified on this machine)
live local HEAD at Phase 0: 6ec18cda2f0dd8d4dbc774026b43fb164d60cfab (worktree clean)
upstream commit verified present: 08339b719642a5d886688330a4b562a031961091 (sjzhang, 2023-11-03)

## COMPILE RESULT (committed evaluator)
python -m py_compile unsupervised/embedding_evaluation.py @08339b7
-> SyntaxError: '(' was never closed (line 256). THE COMMITTED EVALUATOR DOES NOT COMPILE.

## VERIFIED CLAIMS (12/12 confirmed, none contradicted)
 1. Incomplete call: line 256 `... = self.embedding_evaluation(encoder,` — argument list
    and closing paren absent. CONFIRMED (SyntaxError above).
 2. `test_id` undefined: only occurrence is line 243 `test_id.append(test_index)`; no
    definition anywhere in committed source. CONFIRMED.
 3. return/unpack disagree on fpr/tpr: `embedding_evaluation` returns 13 values (line 224,
    no fpr/tpr, never computed in committed source); line 256 unpacks 15 incl. fpr,tpr.
    CONFIRMED.
 4. `running_time` undefined when flag=False: stored only inside `if flag:` (lines 185-188)
    yet returned unconditionally (line 224). Initial pretraining call (queue line 115) uses
    the default flag=False. CONFIRMED — and the SHIPPED BYTECODE HAS THE SAME CONDITIONAL
    STORE (single STORE_FAST at offset 110 inside the flag branch), so even the authors'
    working binary would NameError on flag=False.
 5. README uses `--batch-size` (lines 185, 234); argparse defines `--batch_size` (line 358).
    CONFIRMED — the README command as printed exits with an argparse error.
 6. argparse epochs default = 1 (line 362). CONFIRMED.
 7. README invocation specifies --epochs 100. CONFIRMED.
 8. eval every 5 epochs: `if epoch % args.eval_interval == 0` (line 243), default 5 (366).
    CONFIRMED.
 9. metric appends every epoch outside the eval condition (lines 273-295 at loop indent).
    CONFIRMED — epochs between evaluations re-append the stale scores.
10. initial pretraining evaluation before the loop (line 115). CONFIRMED.
11. downstream CV = KFold(n_splits=10, shuffle=True, random_state=None) (line 241) with
    inner train_test_split(test_size=0.2, random_state=None) (line 246). CONFIRMED.
12. paper reports 5-fold CV ("calculated based on 5-fold cross-validation", Table 2
    caption) and metrics Accuracy/AUC/Precision/Recall/F1; released code computes
    accuracy/F1/sensitivity/specificity on shuffled 10-fold, no AUC, no precision.
    CONFIRMED — metric sets and protocol do not match.

## SHIPPED BYTECODE (evidence class: bytecode-derived, NOT committed source)
  embedding_evaluation.cpython-38.pyc  magic 550d (3.8),  mtime 2023-05-12 17:00:38 UTC, src_size 18393
  embedding_evaluation.cpython-310.pyc magic 6f0d (3.10), mtime 2023-05-12 17:00:38 UTC, src_size 18393
  committed embedding_evaluation.py                                             size 10903
  => the bytecode was compiled from an 18,393-byte source; the committed file is 10,903
     bytes. The authors ran a LONGER evaluator that was never committed.
  Disassembly (matching CPython 3.10.19) recovers the missing control flow:
    - kf: STORE_FAST test_id (test_id was defined; also tprs/aucs/mean_fpr exist)
    - kf: UNPACK_SEQUENCE 15 — matches the committed 15-value unpack
    - embedding_evaluation: BUILD_TUPLE 15 return, names include roc_curve;
      locals include fpr, tpr — the working source computed roc_curve and returned
      (…, fpr, tpr, running_time): the committed 13-value return is the stale half of an edit.
    - call signature recovered from kf bytecode:
      self.embedding_evaluation(encoder, train_loader, valid_loader, test_loader, flag)
    - running_time store remains conditional on flag even in bytecode (see claim 4).

## DECISION TABLE
| item              | paper          | README        | committed .py       | shipped .pyc        | S8-O mirror      | R0 decision |
|-------------------|----------------|---------------|---------------------|---------------------|------------------|-------------|
| downstream CV     | 5-fold         | n/a           | KFold(10,sh,None)   | 10 folds (varnames) | frozen S3C 5-fold| 10-fold, shuffled, random_state=None (released behaviour) |
| epochs            | 200 (Sec 2.4)  | 100           | default 1           | n/a                 | 200              | 100 (README invocation) |
| batch flag        | n/a            | --batch-size  | --batch_size        | n/a                 | n/a              | --batch_size (documented README typo) |
| eval cadence      | n/a            | n/a           | pre + every 5       | same                | final only       | pre + every 5 (21 events, 100-entry curve) |
| eval return       | n/a            | n/a           | 13 values, broken   | 15 incl fpr/tpr     | n/a              | 15-value contract restored from bytecode |
| test_id           | n/a            | n/a           | UNDEFINED           | defined (STORE_FAST)| n/a              | test_id=[] |
| running_time      | n/a            | n/a           | undefined flag=False| same bug in bytecode| n/a              | timing-only value when flag=False |
| downstream repr   | ambiguous      | n/a           | h via encoder       | h                   | h (weighted+slip)| h, UNWEIGHTED (released get_embeddings slip) |
| metrics           | Acc/AUC/P/R/F1 | n/a           | Acc/F1/Sen/Spe      | + fpr/tpr           | AUC              | Acc/F1/Sen/Spe upstream-first; AUC diagnostic only |
No factual claim was contradicted. Phase 0 gate: PASS — proceed to Phase 1.
