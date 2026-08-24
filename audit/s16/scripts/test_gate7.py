"""S16 Gate 7: bounded C2 estimator — synthetic and unit tests. Executes NO refit."""
import sys, os, json, numpy as np
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_c2_bounded as CB, s16_data as DAT
F=[]
def ck(n,ok,d=""):
    print(("PASS " if ok else "FAIL ")+n+(" | "+d if d else ""),flush=True)
    if not ok: F.append(n)

# ---------- synthetic cohort with a deliberately scarce site ----------
print("--- synthetic: matching, no replacement, infeasibility ---")
y = np.array([0,1]*30); sites = np.array(["A"]*40 + ["B"]*16 + ["TINY"]*4)
tr = np.arange(0,44); te = np.arange(44,60)
rng = np.random.default_rng(0)
te_a = np.array([44,45,46,47])
req = CB.cell_counts(te_a, y, sites)
sel = CB.matched_draw(te_a, tr, y, sites, rng)
ck("S1_matched_counts_exact", sel is not None and CB.cell_counts(sel,y,sites)==req,
   f"requested {dict(req)} == drawn {dict(CB.cell_counts(sel,y,sites)) if sel is not None else None}")
ck("S2_no_replacement", sel is not None and len(set(sel.tolist()))==len(sel),
   f"{len(sel)} drawn, {len(set(sel.tolist()))} unique")
ck("S3_drawn_only_from_tr", sel is not None and set(sel.tolist())<=set(tr.tolist()),
   "every drawn subject is in tr")
# infeasible cell -> refuse, do not repair
te_scarce = np.array([56,57,58,59])          # all TINY, none of which are in tr
rows, ok = CB.feasibility(te_scarce, tr, y, sites)
ck("S4_infeasible_detected", not ok, f"rows {rows}")
ck("S5_refuses_rather_than_repairs",
   CB.matched_draw(te_scarce, tr, y, sites, rng) is None,
   "matched_draw returns None — no replacement, no pooling, no weighting")
# determinism across the predefined seeds
d1 = CB.matched_draw(te_a, tr, y, sites, np.random.default_rng(7))
d2 = CB.matched_draw(te_a, tr, y, sites, np.random.default_rng(7))
ck("S6_deterministic_per_seed", np.array_equal(np.sort(d1),np.sort(d2)),
   "same seed -> same draw")
ck("S7_twenty_predefined_seeds", len(CB.SEEDS)==20 and CB.SEEDS[0]==DAT.BASE,
   f"{CB.SEEDS[0]}..{CB.SEEDS[-1]}")
# half splits are label-stratified and disjoint
hs = CB.half_split(te, y, CB.SEEDS[0])
a,b = hs[0]
ck("S8_halves_disjoint_and_cover", len(set(a)&set(b))==0 and len(a)+len(b)==len(te),
   f"|a|={len(a)} |b|={len(b)} |te|={len(te)}")
ck("S9_both_halves_scored", len(hs)==2, "both directions are used, then pooled")

# ---------- uncertainty vocabulary ----------
print("\n--- Monte Carlo reporting vocabulary ---")
summ = CB.monte_carlo_summary([0.02,-0.01,0.03,0.00,0.015])
need = {"mean_paired_difference","sd_across_seeds","monte_carlo_se_of_mean",
        "empirical_interval_min","empirical_interval_max","sign_flips_descriptive"}
ck("U1_required_fields", need <= set(summ), f"{sorted(need & set(summ))}")
txt = json.dumps(summ).lower()
banned = [w for w in ("sigma","p-value","p_value","significant","confidence interval",
                      "population se","reject") if w in txt]
ck("U2_no_forbidden_language", not banned, f"banned terms present: {banned}")
ck("U3_explicitly_not_population_se", "not a population standard error"
   in summ["interpretation"].lower(), summ["interpretation"][:70]+"...")
ck("U4_sign_flips_descriptive", summ["sign_flips_descriptive"]==1,
   f"{summ['sign_flips_descriptive']}/5 draws negative — descriptive count only")

# ---------- predeclared calibration band ----------
print("\n--- predeclared calibration band ---")
ck("B1_band_predeclared", CB.RANDOM_ENCODER_EQUIVALENCE_BAND==(-0.01,0.01),
   f"{CB.RANDOM_ENCODER_EQUIVALENCE_BAND}")
v_in  = CB.calibration_verdict(0.004); v_out = CB.calibration_verdict(0.023)
ck("B2_band_pass", v_in["passed"] is True, v_in["consequence"][:60])
ck("B3_band_fail_blocks_interpretation", v_out["passed"] is False
   and "UNRESOLVED" in v_out["consequence"],
   f"mean 0.023 -> {v_out['consequence'][:80]}")
ck("B4_prior_calibration_would_fail",
   not CB.calibration_verdict(0.0231)["passed"],
   "the earlier unmatched estimator's random-encoder mean of +0.0231 lies OUTSIDE "
   "the band, so its bias estimates remain UNRESOLVED")

# ---------- residual confounds recorded ----------
ck("R1_residual_confounds_recorded",
   set(CB.RESIDUAL_CONFOUNDS)>= {"sex","age","TR"} and
   any("motion" in c for c in CB.RESIDUAL_CONFOUNDS),
   ", ".join(CB.RESIDUAL_CONFOUNDS))

print(f"\n=== GATE 7 SUMMARY: {len(F)} failures ===")
for f in F: print("  FAIL "+f)
sys.exit(1 if F else 0)
