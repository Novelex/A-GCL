"""S16 collector. Refuses to write CSV, report, plot or headline unless the wave is
COMPLETE and UNCONTAMINATED against the exact expected ledger.

A top-level scheduler COMPLETED state is NOT scientific completion. SLURM reports
COMPLETED whenever the process exits 0, which a unit does even when every fold inside
it failed. Completion is decided cell by cell against s16_ledger.expected_ledger()."""
import sys, os, json, glob, collections, numpy as np, pandas as pd
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_prov as P, s16_ledger as L
NS = os.environ.get("S16_NS","prod")

REJECTIONS = ("missing_unit","missing_fold","duplicate_fold","unexpected_cell",
              "failed_record","malformed_json","provenance_absent_or_incompatible",
              "poison_marker","tally_result_disagreement",
              "skipped_without_validated_reuse","wrong_namespace","grid_shape")

def audit(ns=NS):
    """-> (problems dict, rows, tally_totals). Never writes anything."""
    prob = collections.defaultdict(list)
    shape_probs, cells, units, tags = L.assert_grid_shape()
    for p in shape_probs: prob["grid_shape"].append(p)

    # ---- unresolved poison ----
    if os.path.exists(P.poison_path(ns)):
        prob["poison_marker"].append(open(P.poison_path(ns)).read().strip()[:300])
    for t in sorted(glob.glob(P.jobs_dir(ns)+"*/POISON")):
        prob["poison_marker"].append(f"{os.path.basename(os.path.dirname(t))}: "
                                     + open(t).read().strip()[:200])

    # ---- scan every result cell ----
    seen = collections.Counter(); rows=[]; per_unit_ok = collections.Counter()
    for f in sorted(glob.glob(P.jobs_dir(ns)+"*/fold_*.json")):
        uid = os.path.basename(os.path.dirname(f))
        try:
            rec = json.load(open(f))["rec"]
        except Exception as e:
            prob["malformed_json"].append(f"{f}: {e!r}"); continue
        tag = rec.get("fold")
        if rec.get("status") != "OK":
            prob["failed_record"].append(f"{uid}/{tag}: status={rec.get('status')!r}")
            continue
        if rec.get("namespace", ns) != ns:
            prob["wrong_namespace"].append(f"{uid}/{tag}: {rec.get('namespace')!r}")
        cell = (uid, tag)
        if cell not in cells:
            prob["unexpected_cell"].append(f"{uid}/{tag}")
            continue
        seen[cell]+=1
        if seen[cell] > 1:
            prob["duplicate_fold"].append(f"{uid}/{tag} x{seen[cell]}"); continue
        # ---- provenance must exist and be compatible ----
        mfp = P.feat_dir(ns)+f"{uid}__{tag}.npz.prov.json"
        if not os.path.exists(mfp):
            prob["provenance_absent_or_incompatible"].append(f"{uid}/{tag}: no manifest")
        else:
            try:
                m=json.load(open(mfp))
                if m.get("namespace")!=ns:
                    prob["wrong_namespace"].append(f"{uid}/{tag} manifest ns={m.get('namespace')!r}")
                if m.get("status")!="OK":
                    prob["provenance_absent_or_incompatible"].append(
                        f"{uid}/{tag}: manifest status={m.get('status')!r}")
                if m.get("schema")!="s16-prov-1":
                    prob["provenance_absent_or_incompatible"].append(
                        f"{uid}/{tag}: schema={m.get('schema')!r}")
            except Exception as e:
                prob["malformed_json"].append(f"{mfp}: {e!r}")
        per_unit_ok[uid]+=1
        rows.append((uid,tag,rec))

    # ---- missing units and folds ----
    have_units = {u for u,_ in seen}
    for uid,_,_ in units:
        if uid not in have_units:
            prob["missing_unit"].append(uid)
    for cell in sorted(cells):
        if cell not in seen and cell[0] in have_units:
            prob["missing_fold"].append(f"{cell[0]}/{cell[1]}")

    # ---- tallies must agree with results ----
    tot = dict(expected=len(cells), newly_attempted=0, newly_successful=0, failed=0,
               validated_reused=0, remaining=0)
    for t in sorted(glob.glob(P.jobs_dir(ns)+"*/TALLY.json")):
        uid=os.path.basename(os.path.dirname(t))
        try: d=json.load(open(t))
        except Exception as e: prob["malformed_json"].append(f"{t}: {e!r}"); continue
        if d.get("namespace") not in (None,ns):
            prob["wrong_namespace"].append(f"tally {uid}: {d.get('namespace')!r}")
        tot["newly_attempted"]  += d.get("attempted",0)
        tot["newly_successful"] += d.get("newly_succeeded",0)
        tot["failed"]           += d.get("failed",0)
        tot["validated_reused"] += d.get("validated_reused",0)
        claimed = d.get("validated_reused",0)+d.get("newly_succeeded",0)
        if claimed != per_unit_ok.get(uid,0):
            prob["tally_result_disagreement"].append(
                f"{uid}: tally claims {claimed} ok, {per_unit_ok.get(uid,0)} result rows")
        if d.get("accounting_ok") is False:
            prob["skipped_without_validated_reuse"].append(
                f"{uid}: reused+new != expected-failed")
        exp_u = len(tags)
        skipped = exp_u - d.get("attempted",0) - d.get("validated_reused",0)
        if skipped > 0:
            prob["skipped_without_validated_reuse"].append(
                f"{uid}: {skipped} folds neither attempted nor validated-reused")
    tot["remaining"] = tot["expected"] - len(seen)
    return prob, rows, tot

def main():
    prob, rows, tot = audit(NS)
    print(f"=== S16 LEDGER AUDIT [ns={NS}] ===")
    print(f"expected cells {tot['expected']} (ledger hash {L.ledger_hash()}) | "
          f"present {tot['expected']-tot['remaining']} | remaining {tot['remaining']}")
    print(f"tally: expected {tot['expected']} | newly_attempted {tot['newly_attempted']} | "
          f"newly_successful {tot['newly_successful']} | failed {tot['failed']} | "
          f"validated_reused {tot['validated_reused']} | remaining {tot['remaining']}")
    if prob:
        print("\n*** REFUSING TO WRITE CSV / REPORT / PLOT / HEADLINE ***", file=sys.stderr)
        print("The wave is contaminated or incomplete. A scheduler COMPLETED state is "
              "NOT scientific completion.", file=sys.stderr)
        for k in REJECTIONS:
            if k in prob:
                v=prob[k]
                print(f"  [{k}] {len(v)}", file=sys.stderr)
                for x in v[:5]: print(f"      {x}", file=sys.stderr)
                if len(v)>5: print(f"      ... and {len(v)-5} more", file=sys.stderr)
        sys.exit(3)
    # ---- only here may anything be written ----
    R=[]
    for uid,tag,rec in rows:
        base={k:rec.get(k) for k in ("unit","branch","arm","E","arch","mode","seed",
              "fold","fold_protocol","control","alff_mode","kh","svm_tr_enc","n_tr",
              "n_tr_enc","n_tr_probe","movement_max","clip_rate","verdict","best_epoch",
              "total_steps","repr_dim_used","sparse","ema_delta","evaluated_state",
              "h_fc","cache_file","node","wall_s")}
        for g,v in (rec.get("movement") or {}).items(): base["movement_"+g]=v
        for pt in ("head","head_ema","probe_honest","probe_old_full"):
            if isinstance(rec.get(pt),dict): R.append({**base,"eval_point":pt,**rec[pt]})
        fu=rec.get("fusion")
        if fu: R.append({**base,"eval_point":"fused","auc":fu["fused_auc"],
                "alpha_selected":fu["alpha_selected"],"stack_auc":fu["stack_auc"],
                "delta_vs_svm_tr_enc":fu["delta_vs_svm_tr_enc"],
                "delta_vs_0p7565_SECONDARY":fu["delta_vs_0p7565_SECONDARY"],
                "alpha1_ok":fu["alpha1_equals_svm_tr_enc"]})
    df=pd.DataFrame(R); df.to_csv(P.results_path(NS),index=False)
    print(f"\nCLEAN AND COMPLETE [{NS}]: {len(rows)} cells, {len(df)} rows -> "
          f"{P.results_path(NS)}")

if __name__=="__main__": main()
