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

# S16 C2 — PRECISION REFIT (20 repeats of BOTH random draws)

Every pure-bias value in the first C2 pass rested on ONE draw of the te
half-split and ONE draw of the matched tr subset. At ~95 scoring subjects the
paired difference carries SE of roughly +-0.02-0.03 — the SAME ORDER as the
+0.033..+0.054 effects being reported. Both draws are now repeated over
**20 seeds** (20260818..20260837), the frozen probe refitted on the same
saved representations each time. Nothing retrained.

| source | dim | honest (mean±SE) | matched (mean±SE) | **PURE BIAS (mean±SE)** | sd | sign flips |
|---|---|---|---|---|---|---|
| RANDOM WGIN (S12A5 A repr0, epoch-0)  [CALIBRATION] | 32 | 0.5267±0.0033 | 0.5499±0.0046 | **+0.0231±0.0067** | 0.0298 | 4/20 (20%) |
| trained WGIN (S12A4 arm1 h) | 32 | 0.5990±0.0040 | 0.6243±0.0019 | **+0.0253±0.0048** | 0.0217 | 3/20 (15%) |
| S12A5 arm A (WGIN) | 32 | 0.5985±0.0039 | 0.6328±0.0014 | **+0.0343±0.0043** | 0.0192 | 1/20 (5%) |
| S12A5 arm B (WGIN+edge skip) | 64 | 0.6801±0.0037 | 0.6850±0.0014 | **+0.0049±0.0044** | 0.0196 | 9/20 (45%) |
| S12A5 arm C (edge MLP) | 32 | 0.7068±0.0021 | 0.7056±0.0021 | **-0.0012±0.0033** | 0.0147 | 12/20 (60%) |
| S13 BNT winner (K=2 wd1e-4) | 256 | 0.5756±0.0036 | 0.6190±0.0020 | **+0.0433±0.0043** | 0.0192 | 0/20 (0%) |
| S15 B1 BNT K=32 (terminated) | 4096 | 0.5646±0.0041 | 0.6262±0.0026 | **+0.0616±0.0045** | 0.0199 | 0/20 (0%) |

**How to read the sign-flip column.** It is the fraction of the 20 draws in which
the measured bias came out NEGATIVE. For a genuine positive effect it should be
near 0; for a true zero it should sit near 50%. It is the non-parametric check
that does not depend on the SE being well estimated.

**Quotability rule.** Report mean ± SE. A single draw is NOT quotable: the
per-draw sd column shows how far one draw can land from the mean.

wall {time.time()-t0:.0f}s
