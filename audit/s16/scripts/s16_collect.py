"""S16 collector. Validates EVERY expected cell against an independently
reconstructed contract, then writes atomically — or writes nothing at all.

A top-level scheduler COMPLETED state is NOT scientific completion."""
import sys, os, json, glob, collections, math, numpy as np, pandas as pd
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_prov as P, s16_ledger as L, s16_grid as G, s16_policy as PL
NS = os.environ.get("S16_NS","prod")

REJECTIONS = ("grid_shape","poison_marker","missing_unit","missing_fold",
              "duplicate_fold","unexpected_cell","failed_record","malformed_json",
              "bundle_invalid","wrong_namespace","identity_mismatch",
              "fusion_invalid","plain_masquerade","nonfinite_field",
              "tally_missing","tally_duplicate","tally_disagreement",
              "unit_done_missing","status_not_terminal","accounting_mismatch")

def _cfg_for(u, policy):
    return P.model_cfg(u)   # single source (defect D34)

def expected_contract(ns, uid, u, tag, seed, MAN, ent, policy):
    """Reconstructed from grid + ledger + policy + data manifest. NEVER from the
    manifest under validation."""
    cfg = _cfg_for(u, policy)
    return P.contract_fields(ns, u, cfg, uid, tag, seed,
                             tag.rstrip("0123456789"), MAN, ent,
                             P.expected_repr_dim(u["arch"], u["kh"], cfg["H"],
                                                 cfg["readout"]),
                             policy, policy.train_consts())

REQUIRED_FINITE = ("svm_tr_enc","svm_tr_full","size_delta_paired")

def audit(ns=NS, data=None):
    prob = collections.defaultdict(list)
    shape_probs, cells, units, tags = L.assert_grid_shape()
    for p in shape_probs: prob["grid_shape"].append(p)
    policy = PL.get(ns)
    if os.path.exists(P.poison_path(ns)):
        prob["poison_marker"].append(open(P.poison_path(ns)).read().strip()[:300])
    for t in sorted(glob.glob(P.jobs_dir(ns)+"*/POISON")):
        prob["poison_marker"].append(os.path.basename(os.path.dirname(t)))
    umap = {uid:(br,u) for uid,br,u in units}
    # PER-E METADATA (defect D35). This loaded the "signed" cache ONCE and then
    # validated every unit against it, so h_fc and cache_file were structurally
    # wrong for all 810 abs / pos_zero / shift cells: the entire non-signed two
    # thirds of the study would have been rejected as provenance failures.
    # `data` may be a {E: (MAN, ent)} map, or a bare (MAN, ent) pair meaning signed.
    _meta_cache = {}
    if isinstance(data, dict): _meta_cache.update(data)
    elif data is not None:     _meta_cache["signed"] = data
    def meta_for(E):
        if E not in _meta_cache:
            import s16_data as DAT
            _d, m, e = DAT.load(E, where="collect")
            _meta_cache[E] = (m, e)
        return _meta_cache[E]
    seen = collections.Counter(); rows=[]; per_unit_ok = collections.Counter()

    for f in sorted(glob.glob(P.jobs_dir(ns)+"*/fold_*.json")):
        dir_uid = os.path.basename(os.path.dirname(f))
        fname_tag = os.path.basename(f)[len("fold_"):-len(".json")]
        try: rec = json.load(open(f))["rec"]
        except Exception as e: prob["malformed_json"].append(f"{f}: {e!r}"); continue
        tag = rec.get("fold")
        if tag != fname_tag:
            prob["identity_mismatch"].append(f"{dir_uid}: filename tag {fname_tag!r} != record fold {tag!r}")
            continue
        if rec.get("unit") != dir_uid:
            prob["identity_mismatch"].append(f"{dir_uid}: record unit {rec.get('unit')!r} != directory")
            continue
        if "namespace" not in rec:
            prob["wrong_namespace"].append(f"{dir_uid}/{tag}: field ABSENT "
                "(a record without an explicit namespace is not attributable)"); continue
        if rec["namespace"] != ns:
            prob["wrong_namespace"].append(f"{dir_uid}/{tag}: {rec.get('namespace')!r}"); continue
        if rec.get("status") != "OK":
            prob["failed_record"].append(f"{dir_uid}/{tag}: status={rec.get('status')!r}"); continue
        cell = (dir_uid, tag)
        if cell not in cells: prob["unexpected_cell"].append(f"{dir_uid}/{tag}"); continue
        seen[cell]+=1
        if seen[cell] > 1: prob["duplicate_fold"].append(f"{dir_uid}/{tag}"); continue
        br,u = umap[dir_uid]
        for k,want in (("arm",u["arm"]),("arch",u["arch"]),("E",u["E"]),
                       ("mode",u["mode"]),("control",u.get("control")),
                       ("alff_mode",u.get("alff_mode")),
                       ("seed",G.SEEDS[u["seed_idx"]]),
                       ("fold_protocol",tag.rstrip("0123456789"))):
            if rec.get(k) != want:
                prob["identity_mismatch"].append(f"{dir_uid}/{tag}: {k}={rec.get(k)!r} expected {want!r}")
        MAN_E, ent_E = meta_for(u["E"])          # metadata for THIS unit's E level
        exp = expected_contract(ns, dir_uid, u, tag, G.SEEDS[u["seed_idx"]],
                                MAN_E, ent_E, policy)
        okb, why = P.validate_bundle(ns, dir_uid, tag, exp,
            P.feat_dir(ns)+f"{dir_uid}__{tag}.npz", P.ckpt_dir(ns)+f"{dir_uid}__{tag}.pt",
            P.feat_dir(ns)+f"{dir_uid}__{tag}.npz.prov.json", f,
            P.feat_dir(ns)+f"{dir_uid}__{tag}.pred.json")
        if not okb: prob["bundle_invalid"].append(f"{dir_uid}/{tag}: {why}"); continue
        for k in REQUIRED_FINITE:
            v = rec.get(k)
            if v is None or not np.isfinite(v):
                prob["nonfinite_field"].append(f"{dir_uid}/{tag}: {k}={v!r}")
        fu = rec.get("fusion")
        if u["mode"] == "fused":
            ok, why2 = validate_fusion(fu, rec, ns, dir_uid, tag)
            if not ok: prob["fusion_invalid"].append(f"{dir_uid}/{tag}: {why2}"); continue
        else:
            if fu: prob["plain_masquerade"].append(f"{dir_uid}/{tag}: plain cell carries a fusion block")
        per_unit_ok[dir_uid]+=1; rows.append((dir_uid,tag,rec,u))

    have = {u for u,_ in seen}
    for uid,_,_ in units:
        if uid not in have: prob["missing_unit"].append(uid)
    for c in sorted(cells):
        if c not in seen and c[0] in have: prob["missing_fold"].append(f"{c[0]}/{c[1]}")

    tot = dict(expected=len(cells), newly_attempted=0, newly_successful=0, failed=0,
               validated_reused=0, remaining=0)
    for uid,_,_ in units:
        jd = P.jobs_dir(ns)+uid
        # SHARED unit-completion contract (defect D48) — the same function the E2E
        # checker calls, so "this unit finished" means one thing in both places.
        okc, whyc = P.validate_unit_completion(ns, uid, len(tags))
        if not okc:
            for w in whyc:
                key = ("poison_marker" if "POISON" in w else
                       "tally_missing" if "TALLY.json absent" in w else
                       "malformed_json" if "unreadable" in w else
                       "unit_done_missing" if "UNIT.done" in w else
                       "status_not_terminal" if "STATUS" in w else
                       "accounting_mismatch")
                prob[key].append(f"{uid}: {w}")
        tl = glob.glob(jd+"/TALLY.json")
        if len(tl)==0: continue
        if len(tl)>1: prob["tally_duplicate"].append(uid); continue
        try: d=json.load(open(tl[0]))
        except Exception: continue
        if d.get("unit")!=uid: prob["tally_disagreement"].append(f"{uid}: tally unit {d.get('unit')!r}")
        exp_u = len(tags)
        got = dict(reused=d.get("validated_reused",0),
                   new=d.get("newly_successful", d.get("newly_succeeded",0)))
        tot["newly_attempted"]  += d.get("newly_attempted", d.get("attempted",0))
        tot["newly_successful"] += got["new"]; tot["failed"] += d.get("failed",0)
        tot["validated_reused"] += got["reused"]
        if got["reused"]+got["new"] != per_unit_ok.get(uid,0):
            prob["tally_disagreement"].append(
                f"{uid}: tally claims {got['reused']+got['new']} ok, {per_unit_ok.get(uid,0)} valid rows")
    for extra in sorted(glob.glob(P.jobs_dir(ns)+"*/TALLY.json")):
        uid=os.path.basename(os.path.dirname(extra))
        if uid not in umap: prob["tally_duplicate"].append(f"unexpected tally dir {uid}")
    tot["remaining"] = tot["expected"] - len(seen)
    return prob, rows, tot

def validate_fusion(fu, rec, ns, uid, tag):
    if not isinstance(fu, dict): return False, "fusion is not a dict"
    for k in ("alpha_curve","alpha_curve_inner","alpha_selected","fused_auc",
              "alpha1_equals_svm_tr_enc","alpha1_bitwise_equals_zsFC",
              "delta_vs_svm_tr_enc","delta_vs_svm_tr_full","delta_is_unclamped"):
        if k not in fu: return False, f"fusion missing {k!r}"
    if len(fu["alpha_curve"]) != 21: return False, f"{len(fu['alpha_curve'])} test alpha points, need 21"
    if len(fu["alpha_curve_inner"]) != 21: return False, f"{len(fu['alpha_curve_inner'])} inner alpha points, need 21"
    import s16_feat as FT
    grid = [round(float(a),4) for a in FT.ALPHA_GRID]
    if round(float(fu["alpha_selected"]),4) not in grid:
        return False, f"alpha_selected {fu['alpha_selected']} not on the grid"
    repro = max(fu["alpha_curve_inner"], key=lambda r:(r["auc"], r["alpha"]))["alpha"]
    if round(float(repro),4) != round(float(fu["alpha_selected"]),4):
        return False, (f"conservative tie-breaking not reproducible: inner curve gives "
                       f"{repro}, record says {fu['alpha_selected']}")
    if fu["alpha1_equals_svm_tr_enc"] is not True: return False, "alpha=1 AUC identity false"
    if fu["alpha1_bitwise_equals_zsFC"] is not True: return False, "alpha=1 bitwise identity false"
    if fu["delta_is_unclamped"] is not True: return False, "delta_is_unclamped is not True"
    pf = P.feat_dir(ns)+f"{uid}__{tag}.pred.json"
    try: pred=json.load(open(pf))
    except Exception as e: return False, f"prediction unreadable: {e!r}"
    if pred.get("score_fused") is None: return False, "prediction has no fused score"
    y=np.asarray(pred["label_used"]); sf=np.asarray(pred["score_fused"],float)
    if len(np.unique(y))>1:
        from sklearn.metrics import roc_auc_score
        rec_auc=float(roc_auc_score(y,sf))
        if abs(rec_auc-float(fu["fused_auc"]))>1e-9:
            return False, (f"fused AUC does not recompute from predictions: "
                           f"{rec_auc:.10f} vs {fu['fused_auc']:.10f}")
        if abs((rec_auc-float(rec["svm_tr_enc"]))-float(fu["delta_vs_svm_tr_enc"]))>1e-9:
            return False, "delta_vs_svm_tr_enc does not recompute"
    return True, "ok"

CSV_REQUIRED = ["svm_tr_enc","svm_tr_full","size_delta_paired","delta_vs_svm_tr_enc",
                "delta_vs_svm_tr_full","alpha_selected","alpha1_equals_svm_tr_enc",
                "alpha1_bitwise_equals_zsFC","fold_protocol","seed","arm","E","mode",
                "control","alff_mode","eval_point","evaluated_state",
                "ocread_entropy"]

def build_rows(rows):
    R=[]
    for uid,tag,rec,u in rows:
        base={k:rec.get(k) for k in ("unit","branch","arm","E","arch","mode","seed",
              "fold","fold_protocol","control","alff_mode","kh","svm_tr_enc",
              "svm_tr_full","size_delta_paired","n_tr","n_tr_enc","n_tr_probe",
              "movement_max","clip_rate","verdict","best_epoch","total_steps",
              "repr_dim_used","sparse","ema_delta","evaluated_state","policy_name",
              "ocread_entropy","flag_best_epoch_1","integrity_loss_decreased",
              "h_fc","cache_file","node","wall_s")}
        for g,v in (rec.get("movement") or {}).items(): base["movement_"+g]=v
        fu=rec.get("fusion")
        for pt in ("head","head_ema","probe_honest","probe_old_full"):
            if isinstance(rec.get(pt),dict):
                R.append({**base,"eval_point":pt,**rec[pt],
                          "alpha_selected":None,
                          "alpha1_equals_svm_tr_enc":None,
                          "alpha1_bitwise_equals_zsFC":None,
                          "delta_vs_svm_tr_enc":None,"delta_vs_svm_tr_full":None})
        if fu:
            R.append({**base,"eval_point":"fused","auc":fu["fused_auc"],
                "alpha_selected":fu["alpha_selected"],"stack_auc":fu.get("stack_auc"),
                "delta_vs_svm_tr_enc":fu["delta_vs_svm_tr_enc"],
                "delta_vs_svm_tr_full":fu["delta_vs_svm_tr_full"],
                "alpha1_equals_svm_tr_enc":fu["alpha1_equals_svm_tr_enc"],
                "alpha1_bitwise_equals_zsFC":fu["alpha1_bitwise_equals_zsFC"],
                "delta_is_unclamped":fu["delta_is_unclamped"]})
    return R

def main():
    prob, rows, tot = audit(NS)
    print(f"=== S16 LEDGER AUDIT [ns={NS}] ===")
    print(f"expected {tot['expected']} (hash {L.ledger_hash()}) | present "
          f"{tot['expected']-tot['remaining']} | remaining {tot['remaining']}")
    print(f"tally: expected {tot['expected']} | newly_attempted {tot['newly_attempted']} "
          f"| newly_successful {tot['newly_successful']} | failed {tot['failed']} "
          f"| validated_reused {tot['validated_reused']} | remaining {tot['remaining']}")
    if prob:
        print("\n*** REFUSING TO WRITE CSV / REPORT / PLOT / HEADLINE ***", file=sys.stderr)
        print("A scheduler COMPLETED state is NOT scientific completion.", file=sys.stderr)
        for k in REJECTIONS:
            if k in prob:
                v=prob[k]; print(f"  [{k}] {len(v)}", file=sys.stderr)
                for x in v[:5]: print(f"      {x}", file=sys.stderr)
                if len(v)>5: print(f"      ... and {len(v)-5} more", file=sys.stderr)
        sys.exit(3)
    # ---- ALL validation passed. Build the table, THEN write atomically. ----
    try:
        R = build_rows(rows)
        df = pd.DataFrame(R)
        missing = [c for c in CSV_REQUIRED if c not in df.columns]
        if missing: raise KeyError(f"required CSV columns absent: {missing}")
    except Exception as e:
        print(f"\n*** TABLE CONSTRUCTION FAILED: {e!r} — NOTHING WRITTEN ***",
              file=sys.stderr); sys.exit(4)
    out = P.results_path(NS); tmp = out + ".tmp"
    df.to_csv(tmp, index=False)
    pd.read_csv(tmp)                       # validate readback before publishing
    os.replace(tmp, out)                   # atomic
    print(f"\nCLEAN AND COMPLETE [{NS}]: {len(rows)} cells, {len(df)} rows -> {out}")

if __name__=="__main__": main()
