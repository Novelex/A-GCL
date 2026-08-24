# S16 PROTOCOL ADDENDA — applied at C0, before any C1 code exists
Three additions received after C0. Recorded verbatim in effect, with arithmetic
recomputed and open questions flagged. NO code written. NO job submitted.

## ADDITION 1 — edge/profile treatment becomes a 4-level factor E
E ∈ {signed, abs, pos_zero, shift}
  signed   FC as-is (current, unchanged)
  abs      |FC|
  pos_zero fc[fc<0]=0 then .nonzero() -> sparse, subject-specific graph
  shift    (fc+1)/2
WGIN: E transforms the EDGE WEIGHTS. BNT: E transforms the CONNECTION-PROFILE
NODE FEATURES (BNT uses no edge weights; they are never injected into attention).
Old arm A2 is DELETED — it is exactly A1 at E=pos_zero.
FIVE ARMS REMAIN:
  A1 WGIN + ALFF(3)            A3 WGIN + FC-row(90)      A4 WGIN + ALFF+FC-row(93)
  A5 BNT  + FC-row(90)         A6 BNT  + ALFF+FC-row(93)

## ADDITION 2 — C1 builds FOUR caches, one per E
Each separately hashed, each with its own full Gate assertion block.
For pos_zero additionally report edges-per-subject (min / median / max / % of the
8100 retained) and flag loudly any subject with fewer than 90 edges.

## ADDITION 3 — pre-registered prediction (written before running)
shift must match signed within +/-0.01, because
  sum_j ((w+1)/2) * x_j = 1/2 * sum_j w*x_j + 1/2 * sum_j x_j
and on the complete graph the second term is identical for every node, so a
following linear layer plus LayerNorm absorbs it. If |shift - signed| > 0.01,
STOP and report — something is wired wrong.

## REVISED C6 GRID AND ARITHMETIC
> **SUPERSEDED BY AMENDMENT A5 (2026-08-25).** The arithmetic below predates the
> addition of arm A7 (EdgeMLP) at all four E. CURRENT: **21 configs**, 126 MAIN +
> 24 CTRL + 9 ABL = **159 units**, 9 folds each = **1,431 fold-runs**
> (ledger hash `8587b1ca36553408`). The figures in this section are retained only as
> the record of what was agreed at the time.

configs   = A1,A4,A5,A6 x 4 E = 16, plus A3 at E=signed = **17**  [now 21 with A7]
runs      = 17 x {plain, fused} = **34**  [now 42]
folds     = ordinary 0-2 + site-strat 0-2 + LOSO 0-2 = **9**
seeds     = 20260818/19/20 = **3**
MAIN fold-runs = 34 x 9 x 3 = **918**
CONTROLS  = 4 (random-encoder twin, permuted labels, shuffled columns, shuffled
            ROI order) x 2 architectures x 9 folds x 3 seeds = **216**
TOTAL     = **1,134 fold-runs**  [SUPERSEDED: now 1,431]
All submitted in ONE wave, no dependencies between arms.

## OPEN QUESTIONS — I CANNOT PROCEED PAST THESE WITHOUT AN ANSWER (RULE 1)

**Q1. For A4 (WGIN with FC-row node features), does E apply to the edges only, or
to the FC-row node features as well?**
The addition says "For WGIN, E transforms the edge weights" and "For BNT, E
transforms the connection-profile node features". A4 is a WGIN arm that ALSO
carries FC-row node features, and it is run at all four E values, so the rule as
written is silent on half of its FC. A1 is unaffected (ALFF nodes only, E hits
edges). A3 is unaffected (run only at E=signed). Only A4 is ambiguous.
My recommendation: apply E to FC EVERYWHERE it appears, i.e. A4 gets transformed
edges AND transformed profile columns, so that "E" names one consistent treatment
of connectivity rather than two different ones depending on where FC sits. State
which you want.

**Q2. What is "the C4 fusion correction"?**
I have no message containing a C4 correction. I am reading it as the C4 FLOOR TEST
already in the S16 commission — repr = concat(raw_FC_edges(4005), learned(d)), with
the requirement that zeroing the learned block reads EXACTLY 0.7565 to four
decimals. "fused" in the C6 text is then the Stage-B floor arm. If you meant
something else, send it; I have not silently invented a correction.

## TWO TECHNICAL FLAGS ON ADDITION 3 (raised now so a real effect is not misread
## as a wiring bug, and a wiring bug is not excused as a real effect)

**F1. The identity is exact for BNT and only approximate for WGIN.**
For BNT, E acts on NODE FEATURES, and shift is a per-feature affine map. The input
layer Linear(D,H) can represent it EXACTLY (W' = 2W, b' = b - W*1), so shift and
signed are the same function class and should agree to optimisation noise. The
+/-0.01 band is well founded here.
For WGIN, E acts on EDGE WEIGHTS. The constant term 1/2 * sum_j x_j is identical
across NODES within a subject but VARIES ACROSS SUBJECTS. A per-subject constant
vector added to every node is NOT removed by LayerNorm (which normalises per node
across features) and is NOT absorbed by a bias (which is per-feature, not per
subject). So for WGIN the prediction is empirical, not an identity. I will report
|shift - signed| for BNT and WGIN SEPARATELY. A WGIN breach of 0.01 with BNT intact
indicates the per-subject constant, not miswiring; a BNT breach indicates miswiring.
Note also that message_relu=True means the aggregate is sum_j relu(x_j)*w, so the
decomposition is over relu(x_j); the constant-across-nodes property still holds.

**F2. The "<90 edges per subject" flag can almost never fire, and does not test
what it is meant to test.**
diag(FC[i]) == 1.0 EXACTLY for all i (verified in this project's Gate-C:
fc_diag_dev = 0.0). All 90 self-loops are positive, so every subject retains at
least 90 edges under pos_zero by construction, and the flag is vacuous.
The stated intent — "that subject has isolated regions" — is a PER-NODE property.
I will therefore ALSO report, and flag loudly, any NODE whose degree is 1 (self-loop
only, i.e. no positive connection to any other region), with counts per subject.
The commissioned per-subject flag is kept as written; this is an addition, not a
substitution.

## PLAIN ENGLISH
You asked for three changes. First, how we treat the connectivity numbers becomes a
proper four-way experiment (keep the negatives, take absolute values, throw the
negatives away, or slide everything up into a positive range) applied to five model
setups instead of six — the old arm A2 disappears because it is just one of these
new settings. Second, the data preparation step now builds four separate datasets,
one per setting, each independently checked. Third, one of those four settings
("slide everything up") is predicted in advance to give the same answer as the
current one, and if it does not, that means we have wired something wrong rather
than discovered something. I have flagged two things about that prediction: it is
mathematically guaranteed for the transformer but only approximately true for the
graph network, so I will report those two separately; and the proposed check for
"isolated brain regions" cannot actually fire as written, so I will add the check
that does what was intended. I also need two answers before I can start.

**Addenda recorded. C0 remains complete. Awaiting answers to Q1 and Q2, then GO C1.**
