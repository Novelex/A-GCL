# S16 C-RAND REFERENCE MAPPING — predeclared, before any C6 result

The validity gate says an arm must "beat its own C-RAND random-encoder twin by
>= 0.03". **The frozen grid does not contain a per-arm twin.** It contains exactly two
C-RAND units, one per architecture, both at `E=signed`:

| C-RAND unit | architecture | node features | kh |
|---|---|---|---|
| `ctrl_A6_C-RAND_signed_plain_s{0,1,2}` | BNT | FC-row + ALFF (93) | K=32 |
| `ctrl_A4_C-RAND_signed_plain_s{0,1,2}` | WGIN | FC-row + ALFF (93) | hidden=128 |

**The phrase "each arm's own random twin" is therefore withdrawn.** The mapping below
is predeclared instead; the grid is NOT changed.

## Reference mapping
| arm | architecture | node features | C-RAND reference | inputs identical? |
|---|---|---|---|---|
| A1 | WGIN | ALFF (3) | `ctrl_A4_C-RAND` | **NO** — reference uses FC-row+ALFF |
| A3 | WGIN | FC-row (90) | `ctrl_A4_C-RAND` | **NO** — reference adds ALFF |
| A4 | WGIN | FC-row+ALFF (93) | `ctrl_A4_C-RAND` | **YES** — exact |
| A5 | BNT | FC-row (90) | `ctrl_A6_C-RAND` | **NO** — reference adds ALFF |
| A6 | BNT | FC-row+ALFF (93) | `ctrl_A6_C-RAND` | **YES** — exact |
| A7 | EDGEMLP | FC upper triangle (4005) | **NONE EXISTS** | — |

## Stated limitation
Only **A4 and A6** have an exactly-matched random reference. For A1, A3 and A5 the
reference differs in NODE-FEATURE INPUT, so the comparison bounds the architecture's
untrained baseline but does not isolate the arm's own. Every such comparison is
labelled **INPUT-MISMATCHED REFERENCE** in the report and may not be quoted as a
per-arm twin comparison.

## A7
The grid contains **no EDGEMLP C-RAND unit**, so **A7 cannot satisfy the C-RAND
criterion**. A7's validity is therefore assessed on movement and clipping only, and it
is **DESCRIPTIVE ONLY for architecture-level claims**. Adding an A7 control would
change the frozen grid and is **not done silently**; it requires separate
authorisation.

## Applied at reporting time
The validity gate is evaluated per arm as: movement_max > 0.10 **AND** clip_rate <
0.30 **AND** (C-RAND delta >= 0.03 where a reference exists). An arm failing any
component is reported **UNTRAINED** and excluded from architecture verdicts while
still appearing in the full grid.
