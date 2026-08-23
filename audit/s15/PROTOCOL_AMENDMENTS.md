# S15 PROTOCOL AMENDMENTS (timestamped, justified, never silent)

## A1 — 2026-08-24 — WGIN width levels {64,128} -> {128,256} (BEFORE any result)
CONFLICT (internal to the commission, found at model-construction time, no result
had been computed): Section 5 mandates for WGIN-R
    "inp : Linear(D, hidden). ASSERT hidden >= D.   (fixes B2)"
while Section 9 specifies the sweep level set hidden{64,128}. Arms W2 (D=90) and
W3 (D=93) therefore REQUIRE hidden >= 90/93, and hidden=64 is a COMPRESSION —
precisely the B2 defect the assert exists to forbid. The two sections cannot both
be satisfied at hidden=64. Verified concretely: WGINR(D=90, hidden=64) raises
"hidden(64) must be >= D(90): fixes B2".

RESOLUTION: WGIN width levels become {128, 256}, applied UNIFORMLY to W1, W2 and
W3. Rationale, in order of priority:
 (i) The hard assert wins. "Never a compression" is a scientific requirement of
     this wave (it is the repair of B2); the specific integer 64 is not.
 (ii) 128 is already in the commissioned set, so only the lower level moves.
 (iii) UNIFORM levels keep W1/W2/W3 mutually comparable. The ragged alternative
     ({64,128} for W1 where D=3, {96,128} for W2/W3) would make the width
     contrast mean something different in each arm and would break the
     cross-arm reading of decision rule R3.
 (iv) {128,256} preserves a clean 2x width contrast (1.4x and 2.8x of D), which
     is what the level set is for — the WGIN analogue of R8's K=8 vs K=32.
CONSEQUENCE: WGIN unit count and every decision rule are unchanged. R8's WGIN-side
reading is now "hidden 256 vs 128" instead of "128 vs 64". No other section,
threshold, arm, seed, or training setting is touched.

## A2 — 2026-08-24 — fold count is 29, not 27 (declared in PROTOCOL.md at write time)
F-LOSO yields 19 evaluable sites (both classes present), so per-unit folds are
5 (F-LAB) + 5 (F-SITE) + 19 (F-LOSO) = 29, versus the 27 estimated in the
commission. The builder confirms: lab 5, site 5, loso 19. No site fell below the
SMALL_SITE_MIN=10 threshold, so the small-site pooling bucket is EMPTY and no
site was pooled for F-SITE stratification.
