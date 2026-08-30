"""Final report: numbers only. Reads AGG_A.json and AGG_BC.json."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_lib as L
from aggA import boot
A = json.load(open(L.ROOT + "A/AGG_A.json")); BC = json.load(open(L.ROOT + "AGG_BC.json")) if os.path.exists(L.ROOT + "AGG_BC.json") else {}
y = L.data()["y"]; out = []; P = out.append
def f4(v): return "n/a" if v is None else f"{v:.4f}"       # None-safe (families absent in a run)
P("# S17 ACCURACY SWEEP — REPORT (numbers only)\n")
P(f"reference means: {A.get('_meta', {}).get('reference_mean_implementations', 'n/a')}")
P("neural optimiser recipe: S16 PROD policy 798ed7790c1ddabc (AdamW, warmup 0.10, cosine floor 0.05, label smoothing 0.05, batch 32, adaptive clip p90/200 after 50 steps, min 80 / patience 50 / max 400 epochs); only lr and wd swept.")
P("health rule: a config is eligible in a fold only if its OUTER refit satisfies movement_max > 0.10 and clip_rate < 0.30 and at least 3 of its 5 inner-split trainings do; inner runs failing health are excluded from the config's inner mean and counted as discarded runs. A fold with no eligible config is reported UNRESOLVED and no pooled number is given.")
for p in L.protocols():
    a = A.get(p, {}); b = BC.get(p, {})
    P(f"\n## {p.upper()}" + ("  (descriptive: test folds hold 20-173 subjects)" if p == "loso" else ""))
    if not a.get("complete"): P(f"Stage A INCOMPLETE: {a.get('units_present')}/{a.get('units_expected')}"); continue
    P(f"gate flat+LinearSVC (S5.5 grid) nested OOF: **{f4(a['gate_flat_linsvm_s55grid'])}**" + (f"  pass={a.get('gate_pass')}" if p == "lab" else ""))
    sh = a['shuffle_arm_auc_vs_permuted_labels']
    P(f"shuffled-label arm: {f4(sh)}" + ("" if sh is None or 0.45 <= sh <= 0.55 else "  **OUTSIDE [0.45, 0.55]**"))
    P(f"\n### Honest nested OOF (selection on inner folds only)")
    P(f"- Stage A, all {a['n_configs_searched']} classical configs: **{f4(a['honest_nested_all'])}**")
    P(f"- flat + LinearSVC, wide C grid: {f4(a['honest_flat_linsvm_widegrid'])}")
    for kd, v in a["honest_nested_per_clf"].items(): P(f"- Stage A, {kd} only: {f4(v)}")
    if b.get("complete"):
        P(f"- Stage B, neural (all inputs): **{f4(b['B_honest_nested_all'])}**  (runs discarded by health: {b['B_n_runs_discarded_by_health']} (fold,input,config) triples; unresolved folds: {b.get('B_unresolved_folds') or 'none'})")
        for i, v in b["B_honest_per_input"].items(): P(f"  - input {i}: {f4(v)}")
        P(f"- Stage C, mALFF-90 MLP alone: {f4(b['C_alff_mlp_honest_nested'])}  (discarded runs {b.get('C_n_runs_discarded_by_health')}; unresolved folds: {b.get('C_unresolved_folds') or 'none'})")
        if "blend_fused_honest_auc" not in b: P(f"- Blend: {b.get('blend')}")
        if "blend_fused_honest_auc" in b: P(f"- Blend w1 z(A-winner) + w2 z(ALFF-MLP): **{b['blend_fused_honest_auc']:.4f}**, w2 per fold {b['blend_w2_per_fold']}, w2≠0 in {b['blend_w2_nonzero_folds']}/{len(b['blend_w2_per_fold'])} folds; blend−A: {b['blend_boot_vs_A_winner']['delta']:+.4f} [{b['blend_boot_vs_A_winner']['lo']:+.4f},{b['blend_boot_vs_A_winner']['hi']:+.4f}]")
    P(f"\n### Optimistic (best single config by OUTER OOF — selection effect visible)")
    o = a["optimistic_best_single"]; P(f"- Stage A best-single: {o['outer_oof_auc']:.4f}  ({o['rep']} | {o['cfg']}; inner mean {o['inner_mean']:.4f})  → selection effect vs honest: +{a['selection_effect']:.4f}")
    if b.get("complete") and b.get("B_optimistic_best_single"):
        o2 = b["B_optimistic_best_single"]; P(f"- Stage B best-single: {o2['outer_oof_auc']:.4f}  (in{o2['input_idx']} {o2['rep']} alff={o2['alff']} | {o2['cfg']}; inner {o2['inner_mean']:.4f})  → selection effect: +{b['B_selection_effect']:.4f}")
    P(f"\n### Honest winner vs flat+LinearSVC (S5.5 grid), paired bootstrap 2000")
    bb = a["boot_winner_vs_flat_linsvm_s55"]; P(f"- Stage A honest winner: {bb['delta']:+.4f} [{bb['lo']:+.4f}, {bb['hi']:+.4f}]")
    if b.get("complete") and "B_boot_vs_flat_linsvm_s55" in b:
        for lbl, key in (("Stage B honest winner", "B_boot_vs_flat_linsvm_s55"), ("Stage C ALFF-MLP", "C_boot_vs_flat_linsvm_s55"), ("Blend", "blend_boot_vs_flat_linsvm_s55")):
            bx = b[key]; P(f"- {lbl}: {bx['delta']:+.4f} [{bx['lo']:+.4f}, {bx['hi']:+.4f}]")
    P(f"\n### Per-fold winners (Stage A)")
    for w in a["per_fold_winners"]: P(f"- {w['fold']}: {w['rep']} | {w['cfg']} (inner {w['inner']:.4f})")
    P(f"boundary pins: {a['boundary_pins'] or 'none'}")
    if b.get("complete"):
        P(f"\n### Per-fold winners (Stage B)")
        for w in b["B_per_fold_winners"]: P(f"- {w['fold']}: in{w['input_idx']} {w['rep']} alff={w['alff']} | {w['cfg']} (inner {w['inner']:.4f}, mv {w['movement_max']:.3f}, clip {w['clip_rate']:.3f}, ep {w['best_epoch']})")
        P(f"width won: {b['B_width_won']}  dropout won: {b['B_dropout_won']}  pins: {b['B_boundary_pins'] or 'none'}")
        P(f"ALFF in Stage-B winners: {b['B_alff_in_winners']}/{len(b['B_per_fold_winners'])} folds; in top-20 optimistic: {b['B_alff_in_top20']}/20")
        P(f"\n### Per-fold winners (Stage C, mALFF-90 MLP)")
        for w in b["C_per_fold_winners"]: P(f"- {w['fold']}: {w['cfg']} (inner {w['inner']:.4f}, mv {w['movement_max']:.3f}, clip {w['clip_rate']:.3f}, ep {w['best_epoch']})")
    P(f"\n### Top-20 configs by OUTER OOF AUC (OPTIMISTIC), Stage A")
    P("| rank | rep | cfg | outer OOF | inner mean |"); P("|---|---|---|---|---|")
    for i, d in enumerate(a["top20_optimistic"], 1): P(f"| {i} | {d['rep']} | {d['cfg']} | {d['outer_oof_auc']:.4f} | {d['inner_mean']:.4f} |")
    if b.get("complete"):
        P(f"\n### Top-20 configs by OUTER OOF AUC (OPTIMISTIC), Stage B (health-valid in every fold)")
        P("| rank | input | rep | alff | cfg | outer OOF | inner mean | mv med | ep med |"); P("|---|---|---|---|---|---|---|---|---|")
        for i, d in enumerate(b["B_top20_optimistic"], 1): P(f"| {i} | in{d['input_idx']} | {d['rep']} | {d['alff']} | {d['cfg']} | {d['outer_oof_auc']:.4f} | {d['inner_mean']:.4f} | {d['movement_med']:.3f} | {d['best_epoch_med']:.0f} |")
    P(f"\ntop-3 representations by mean INNER score (Stage-B hand-off): {a['top3_reps_by_mean_inner']}   [by honest nested, for reference: {a['top3_reps_by_honest_nested']}]")
open(L.ROOT + "REPORT.md", "w").write("\n".join(out)); print("\n".join(out))
