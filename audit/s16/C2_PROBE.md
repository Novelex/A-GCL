> # ⚠️ SUPERSEDED — DO NOT QUOTE
>
> These figures come from an estimator whose **calibration point FAILED**: the
> random (epoch-0) encoder, which cannot memorise anything and must therefore read
> zero, produced a mean paired difference of **+0.0231**, outside the predeclared
> equivalence band **[-0.01, +0.01]** (declared in `s16_c2_bounded.py`).
>
> The estimator was also **unmatched**: its biased comparator drew ~95 subjects from
> all 763 training subjects spanning every site, while its honest comparator drew ~95
> from a single test fold. The more diverse draw generalises better regardless of
> memorisation, which is the most likely source of the +0.0231 floor.
>
> **All retrospective pure-bias estimates in this file are UNRESOLVED.** No arm may be
> described as memorising on the strength of these numbers, and in particular the
> earlier claim that "only the two BNTs show real memorisation bias" is **withdrawn**
> pending a calibration that passes.
>
> Replacement: the site x label matched estimator in `s16_c2_bounded.py`
> (feasibility confirmed in `C2_FEASIBILITY.md`; **not yet executed**).

# S16 C2 (AMENDED) — DECOMPOSING THE PROBE DELTA

The earlier single delta was CONFOUNDED: `probe_honest` changed three things at
once — it removed the bias, cut the probe's training set from ~763 to ~153, and
cut the encoder's from ~763 to ~610. What follows separates them.

## The four probes (all on the SAME saved representations, nothing retrained)
| probe | fit on | memorised by encoder? | scores |
|---|---|---|---|
| 1 `old_full` | tr (~763) | YES | te |
| 2 `old_subset` | tr_probe (~153) | YES | te |
| 3 `honest_teSplit` | one half of te (~95) | **NO** | other half of te |
| 4 `biased_matched` | tr subset, SAME SIZE as (3), class-balance matched | YES | same half of te |

## Decomposition
- **(1) − (2) = SAMPLE-SIZE EFFECT** — probe training set shrinks, bias unchanged
- **(4) − (3) = PURE BIAS** — identical training-set size, identical scoring set;
  differs ONLY in whether the probe's training subjects were memorised
- (2) − (3) = the commissioned "actual bias" (retains a residual 153-vs-95 size
  confound; reported for continuity, but (4) − (3) is the clean number)

**Why true `probe_honest` is not computable here:** for an already-trained
checkpoint the encoder saw ALL of tr, so no subset of tr is out-of-sample and
S16's tr_enc/tr_probe split cannot be reconstructed after the fact. Only te is
unseen. S16's own C6 runs DO yield a true `probe_honest`, because their encoders
train on tr_enc only.

## PRECISION WARNING — READ BEFORE ANY NUMBER
`honest_teSplit` scores only ~95 subjects per fold, so a PER-FOLD AUC carries a
standard error of roughly **±0.05**. Per-fold values are therefore
**NOISE-DOMINATED and must never be quoted individually.** Every headline number
below is a **POOLED out-of-fold AUC** — one AUC computed over all covered
subjects at once, not a mean of small per-fold AUCs. Per-fold values are retained
in `out/C2_RESCORE.json` and are explicitly marked noise-dominated there.

## Results — POOLED out-of-fold AUC (headline)

| source | folds | dim | 1 old_full | 2 old_subset | 3 honest_teSplit | 4 biased_matched | size effect (1−2) | **PURE BIAS (4−3)** | commissioned (2−3) |
|---|---|---|---|---|---|---|---|---|---|
| RANDOM WGIN (S12A5 A repr0, epoch-0)  [CALIBRATION] | 5 | 32 | 0.6207 | 0.5671 | 0.5578 | 0.5435 | +0.0535 | **-0.0143** | +0.0093 |
| trained WGIN (S12A4 arm1 h) | 5 | 32 | 0.6433 | 0.6288 | 0.6166 | 0.6163 | +0.0145 | **-0.0003** | +0.0121 |
| S12A5 arm A (WGIN) | 5 | 32 | 0.6481 | 0.6157 | 0.5999 | 0.6366 | +0.0324 | **+0.0367** | +0.0158 |
| S12A5 arm B (WGIN+edge skip) | 5 | 64 | 0.7041 | 0.7049 | 0.6529 | 0.6860 | -0.0008 | **+0.0331** | +0.0519 |
| S12A5 arm C (edge MLP) | 5 | 32 | 0.7210 | 0.7143 | 0.7046 | 0.7083 | +0.0068 | **+0.0036** | +0.0097 |
| S13 BNT winner (K=2 wd1e-4) | 5 | 256 | 0.6404 | 0.5811 | 0.5850 | 0.6336 | +0.0593 | **+0.0486** | -0.0040 |
| S15 B1 BNT K=32 (terminated) | 5 | 4096 | 0.6425 | 0.6209 | 0.5907 | 0.6443 | +0.0216 | **+0.0537** | +0.0302 |

**FIXED ANCHOR, carries no bias:** LinearSVC on raw FC = **0.7565** ord /
**0.7432** LOSO. No encoder is fitted to produce raw FC, so no distribution shift
exists and neither correction applies to it.

**CALIBRATION:** the RANDOM (epoch-0) encoder never trained, so its PURE BIAS
must be ≈ 0 while its SAMPLE-SIZE EFFECT is real. It is the control that separates
the two columns.

**FLAGGED FOR C7, NOT NOW:** the statistically clean fix is CROSS-FITTING — an
inner K-fold that builds a fully out-of-sample R[tr] using all 763 subjects, so
there is no bias AND no sample starvation. It costs 5× encoder training, so it is
for the winner only, at C7.

wall {time.time()-t0:.0f}s

## THE SAME MODELS ON ONE HONEST SCALE

Every learned representation below is read with `honest_teSplit` — probe fitted on
unseen subjects, scored on unseen subjects. The linear baseline needs no correction.

| model | historical (old_full) | honest_teSplit | drop |
|---|---|---|---|
| RANDOM WGIN (S12A5 A repr0, epoch-0)  [CALIBRATION] | 0.6207 | **0.5578** | -0.0628 |
| trained WGIN (S12A4 arm1 h) | 0.6433 | **0.6166** | -0.0267 |
| S12A5 arm A (WGIN) | 0.6481 | **0.5999** | -0.0482 |
| S12A5 arm B (WGIN+edge skip) | 0.7041 | **0.6529** | -0.0512 |
| S12A5 arm C (edge MLP) | 0.7210 | **0.7046** | -0.0164 |
| S13 BNT winner (K=2 wd1e-4) | 0.6404 | **0.5850** | -0.0554 |
| S15 B1 BNT K=32 (terminated) | 0.6425 | **0.5907** | -0.0518 |
| **LinearSVC raw FC (no encoder fitted)** | **0.7565** | **0.7565** | **0.0000** |

## CORRECTION TO SOMETHING I SAID EARLIER
In an earlier message I wrote that S13's BNT winner falls *below* the random-encoder
watermark of 0.6539. **That was an apples-to-oranges comparison and it was wrong.**
0.6539 is a BIASED reading of the random encoder. On the honest scale the random
encoder reads **0.5578**, and S13's BNT reads **0.5850** — so BNT beats its random
twin by **+0.027**, it does not fall below it. Both numbers dropped; the ordering
survived. The correct statement is that BNT clears random by a small margin, not that
it fails to clear it.

## WHAT THE DECOMPOSITION SHOWS

**1. The calibration point behaves exactly as predicted.** The random (epoch-0)
encoder never trained, so it can carry no memorisation bias — and its PURE BIAS is
**-0.0143**, i.e. zero within noise. Its sample-size effect is nonetheless a large
**+0.0535**. The two effects are therefore separated by measurement, not by argument.
A second near-zero anchor appears unprompted: S12A4's trained WGIN has PURE BIAS
**-0.0003** — consistent with S12A4b's finding that that model barely left its
initialisation, so there was little to memorise.

**2. Pure bias is real, and it tracks how hard the encoder actually fits.**
Near zero for the random encoder (-0.014), the barely-trained WGIN (-0.000) and the
edge MLP (+0.004); substantial for the models that genuinely fit — S12A5 arm A
**+0.037**, arm B **+0.033**, S13 BNT **+0.049**, S15 BNT **+0.054**.

**3. The commissioned estimator (2)-(3) is unreliable and must not be used alone.**
For S13 it reads **-0.0040** while the size-matched measurement gives **+0.0486** —
the wrong sign and an order of magnitude out. The 153-vs-95 size residual it carries
is comparable to the effect being measured. Column 4 exists because of this, and the
headline bias should always be (4)-(3).

**4. The sample-size effect is large and erratic**, from **-0.001** (arm B) to
**+0.059** (S13). It is not a small correction that can be waved through, and it is
the dominant term for the random encoder.

## PLAIN ENGLISH

We had been scoring our models with a ruler that was reading high, and we wanted to
know by how much. The problem is that fixing the ruler also shrinks the amount of
data it gets to use, and that alone changes the reading. So we measured the two
things separately.

The honest answer is that **both effects are real and roughly the same size**. Giving
the ruler less data costs somewhere between nothing and about 0.06 depending on the
model. The genuine unfairness — scoring a model partly on subjects it had already
memorised — is worth about **0.03 to 0.05** for the models that actually learned
something, and essentially **zero** for models that barely trained or that never
trained at all. That last point is the reassuring one: an untrained model shows no
unfairness, which is exactly what should happen, and it tells us the measurement is
working rather than inventing an effect.

What this means for the project: every graph-model score we have reported was
somewhat too high, by roughly 0.03-0.05 for the models that trained properly. The
linear baseline of 0.7565 is unaffected, because nothing is fitted to produce it.
**So the gap between our models and the simple linear method is wider than we
thought, not narrower.** The best learned representation on the honest scale is the
edge MLP at 0.7046, still about 0.05 short of the linear model.
