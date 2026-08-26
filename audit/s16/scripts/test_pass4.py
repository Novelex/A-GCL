"""S16 Pass 4: adversarial tests for D53-D56.

D53 strict tally typing   D54 unpaired cells stop the report
D55 per-cell evaluation contract   D56 no expected-failing test_* file

No scientific data fitting: every fixture is synthetic. Bundles built here are
INTERNALLY HASH-CONSISTENT — the manifest is rebuilt after each mutation — so a
rejection can only come from the contract under test, never from a stale hash."""
import sys, os, json, glob, shutil, subprocess, collections, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
OK = []
def ck(n, c, d=""):
    OK.append(bool(c)); print(f"  [{'PASS' if c else 'FAIL'}] {n}" + (f" | {d}" if d else ""))
import s16_prov as P, s16_policy as PL, s16_ledger as L, s16_grid as G, s16_collect as C

# ================================================================= D53
print("=== D53: TALLY fields must be present, integer, non-boolean, in range ===")
NS, UID, F = "test", "u", 9
def tally(**over):
    shutil.rmtree(P.root(NS), ignore_errors=True); P.ensure(NS)
    jd = P.jobs_dir(NS)+UID; os.makedirs(jd, exist_ok=True)
    t = dict(unit=UID, namespace=NS, expected=F, validated_reused=0,
             newly_successful=F, failed=0, remaining=0)
    t.update(over)
    for k in [k for k, v in list(t.items()) if v is ...]: t.pop(k)
    json.dump(t, open(jd+"/TALLY.json", "w"))
    json.dump(dict(state="done"), open(jd+"/STATUS.json", "w"))
    open(jd+"/UNIT.done", "w").write("d")
    return P.validate_unit_completion(NS, UID, F)

ok, why = tally(); ck("a well-formed tally validates", ok, f"{why}")
for name, over, expect in (
    ("`remaining` ABSENT",        dict(remaining=...),        "'remaining' is ABSENT"),
    ("`remaining` null",          dict(remaining=None),       "not an integer"),
    ("`remaining` boolean False", dict(remaining=False),      "is a boolean"),
    ("`remaining` boolean True",  dict(remaining=True),       "is a boolean"),
    ("`remaining` string '0'",    dict(remaining="0"),        "not an integer"),
    ("`remaining` nonzero",       dict(remaining=3),          "must be exactly 0"),
    ("`remaining` negative",      dict(remaining=-1),         "is negative"),
    ("`failed` ABSENT",           dict(failed=...),           "'failed' is ABSENT"),
    ("`failed` boolean False",    dict(failed=False),         "is a boolean"),
    ("`failed` nonzero",          dict(failed=2),             "must be 0"),
    ("`expected` ABSENT",         dict(expected=...),         "'expected' is ABSENT"),
    ("`expected` string",         dict(expected="9"),         "not an integer"),
    ("`validated_reused` ABSENT", dict(validated_reused=...), "'validated_reused' is ABSENT"),
    ("`validated_reused` string", dict(validated_reused="0"), "not an integer"),
    ("`newly_successful` ABSENT", dict(newly_successful=...), "ABSENT"),
    ("a single count exceeds expected", dict(validated_reused=F+1, newly_successful=0), "exceed expected_folds"),
    ("counts sum above expected",  dict(validated_reused=F, newly_successful=F), "reused 9 + new 9"),
    ("accounting identity off",   dict(newly_successful=F-1), "reused 0 + new 8"),
):
    ok2, why2 = tally(**over)
    ck(f"rejects {name}", (not ok2) and any(expect in w for w in why2),
       ("; ".join(why2)[:92] if why2 else "ACCEPTED"))
ok3, why3 = tally(newly_successful=..., newly_succeeded=F)
ck("accepts the `newly_succeeded` spelling", ok3, f"{why3}")
shutil.rmtree(P.root(NS), ignore_errors=True)

# ================================================================= D54
print("\n=== D54: unpaired cells stop the report; AUC magnitude does not ===")
import s16_report as RP
tab = pd.DataFrame([dict(arch="BNT", protocol="lab", n_pairs=10, paired_diff=0.30)])
ck("fully paired + LARGE AUC diff (0.30) passes", RP.shift_gate(tab, []) == [],
   "the withdrawn +/-0.01 gate is NOT restored")
f1 = RP.shift_gate(tab, ["300 signed, 299 shift, 299 paired"])
ck("a nonempty unpaired list FAILS", len(f1) == 1 and "unpaired shift/signed cells" in f1[0],
   f1[0][:88] if f1 else "no failure")
f2 = RP.shift_gate(pd.DataFrame([dict(arch="BNT", protocol="lab", n_pairs=0, paired_diff=0.0)]), [])
ck("a zero-pair group FAILS", len(f2) == 1 and "zero pairs" in f2[0], f2[0][:70] if f2 else "")
ck("an empty table FAILS", len(RP.shift_gate(pd.DataFrame(), [])) == 1)
ck("`unpaired` is actually read", "unpaired" in RP.shift_gate.__code__.co_varnames
   and "for u in (unpaired" in open(f"{HERE}/s16_report.py").read())

# ================================================================= D55
print("\n=== D55: per-cell evaluation contract, hash-consistent bundles ===")
POL = PL.get(NS)
E_META = {E: ({"h_alff":"a","h_labels":"l","h_subject_order":"o","h_folds_lab":"fl",
               "h_folds_site":"fs","h_folds_loso":"fo"},
              {"h_fc":f"h_{E}", "cache_file":f"fc_{E}.npz", "sparse":(E=="pos_zero")})
          for E in ("signed","abs","pos_zero","shift")}

def write_cell(uid, u, tag, mutate_rec=None):
    """Write one COMPLETE sealed bundle, then rebuild the manifest so its hashes
    match whatever the mutation produced. Any rejection is therefore the contract's
    doing, never a stale hash."""
    MAN, ent = E_META[u["E"]]; cfg = P.model_cfg(u); seed = G.SEEDS[u["seed_idx"]]
    jd = P.jobs_dir(NS)+uid; os.makedirs(jd, exist_ok=True)
    fp  = P.feat_dir(NS)+f"{uid}__{tag}.npz"; ckp = P.ckpt_dir(NS)+f"{uid}__{tag}.pt"
    prd = P.feat_dir(NS)+f"{uid}__{tag}.pred.json"; res = jd+f"/fold_{tag}.json"
    np.savez_compressed(fp[:-4], repr=np.zeros((4,4), np.float32))
    open(ckp,"wb").write(b"CKPT")
    import s16_feat as FT
    from sklearn.metrics import roc_auc_score
    yv = np.array([0,1,0,1]); sf = np.array([0.1,0.9,0.2,0.8])
    fauc = float(roc_auc_score(yv, sf))
    pred = {"unit":uid,"fold":tag,"namespace":NS,"label_used":yv.tolist()}
    rec = dict(status="OK", unit=uid, fold=tag, namespace=NS, arm=u["arm"], E=u["E"],
        arch=u["arch"], mode=u["mode"], seed=int(seed),
        fold_protocol=tag.rstrip("0123456789"), control=u.get("control"),
        alff_mode=u.get("alff_mode"), kh=u["kh"], svm_tr_enc=0.75, svm_tr_full=0.75,
        size_delta_paired=0.0, policy_hash=POL.policy_hash(), policy_name=POL.name,
        ocread_entropy=float("nan"), movement_max=0.5, clip_rate=0.05,
        head=dict(auc=0.60), head_ema=dict(auc=0.59),
        probe_honest=dict(auc=0.61), probe_old_full=dict(auc=0.62),
        evaluated_state=("raw=validation-best checkpoint; EMA(0.999) reported "
                         "alongside; selection by VALIDATION only "
                         "(S15 PROTOCOL.md:186)"))
    if u["mode"] == "fused":
        pred["score_fused"] = sf.tolist()
        grid = [round(float(a),4) for a in FT.ALPHA_GRID]
        inner = [{"alpha":a,"auc":0.5} for a in grid]; inner[-1]["auc"] = 0.9
        rec["fusion"] = dict(alpha_curve=[{"alpha":a,"auc":0.5} for a in grid],
            alpha_curve_inner=inner, alpha_selected=grid[-1], fused_auc=fauc,
            alpha1_equals_svm_tr_enc=True, alpha1_bitwise_equals_zsFC=True,
            delta_vs_svm_tr_enc=fauc-0.75, delta_vs_svm_tr_full=fauc-0.75,
            delta_is_unclamped=True)
    if mutate_rec: mutate_rec(rec)
    json.dump(pred, open(prd,"w")); json.dump({"rec":rec,"curve":[]}, open(res,"w"))
    man = P.build_manifest(NS, u, cfg, uid, tag, seed, tag.rstrip("0123456789"),
        MAN, ent, P.expected_repr_dim(u["arch"], u["kh"], cfg["H"], cfg["readout"]),
        fp, ckp, "OK", POL.train_consts(), policy=POL, result_path=res, pred_path=prd)
    man["worktree_clean"] = True                     # hashes rebuilt AFTER mutation
    P.atomic_json(man, fp+".prov.json")

def one_unit_fixture(mutate_rec=None, mode="fused"):
    shutil.rmtree(P.root(NS), ignore_errors=True); P.ensure(NS)
    cells, units, tags = L.expected_ledger()
    uid, br, u = next((x for x in units if x[2]["mode"] == mode))
    write_cell(uid, u, tags[0], mutate_rec)
    return uid, u, tags[0]

def accepted(mutate_rec=None, mode="fused"):
    uid, u, tag = one_unit_fixture(mutate_rec, mode)
    rec = json.load(open(P.jobs_dir(NS)+uid+f"/fold_{tag}.json"))["rec"]
    ok, why = P.validate_eval_contract(rec, u["mode"])
    shutil.rmtree(P.root(NS), ignore_errors=True)
    return ok, why

ok, why = accepted(); ck("a complete fused cell satisfies the contract", ok, f"{why}")
ok, why = accepted(mode="plain"); ck("a complete plain cell satisfies the contract", ok, f"{why}")

for pt in P.EVAL_POINTS_PLAIN:
    ok2, why2 = accepted(lambda r, p=pt: r.pop(p))
    ck(f"rejects: {pt} dictionary REMOVED", (not ok2) and any(pt in w for w in why2),
       "; ".join(why2)[:80])
for pt in P.EVAL_POINTS_PLAIN:
    for bad, lbl in ((float("nan"), "NaN"), (float("inf"), "Inf"), (1.5, "out of [0,1]")):
        ok2, why2 = accepted(lambda r, p=pt, b=bad: r.__setitem__(p, dict(auc=b)))
        ck(f"rejects: {pt} auc={lbl}", (not ok2) and any(p_ in w for w in why2 for p_ in (pt,)),
           "; ".join(why2)[:70])
for k in ("movement_max", "clip_rate"):
    ok2, why2 = accepted(lambda r, kk=k: r.__setitem__(kk, float("nan")))
    ck(f"rejects: {k} = NaN", (not ok2) and any(k in w for w in why2), "; ".join(why2)[:70])
    ok2, why2 = accepted(lambda r, kk=k: r.pop(kk))
    ck(f"rejects: {k} absent", (not ok2) and any(k in w for w in why2), "; ".join(why2)[:70])
ok2, why2 = accepted(lambda r: r.__setitem__("evaluated_state", "EMA chosen after seeing test"))
ck("rejects: evaluated_state not the frozen protocol", (not ok2)
   and any("frozen raw/EMA protocol" in w for w in why2), "; ".join(why2)[:80])
ok2, why2 = accepted(lambda r: r.pop("evaluated_state"))
ck("rejects: evaluated_state absent", not ok2, "; ".join(why2)[:70])
ok2, why2 = accepted(lambda r: r["fusion"].__setitem__("fused_auc", float("nan")))
ck("rejects: fused_auc = NaN", (not ok2) and any("fused_auc" in w for w in why2), "; ".join(why2)[:70])
ok2, why2 = accepted(lambda r: r.pop("fusion"))
ck("rejects: fused cell with NO fusion block", (not ok2)
   and any("no fusion block" in w for w in why2), "; ".join(why2)[:70])
ok2, why2 = accepted(lambda r: r.__setitem__("fusion", dict(fused_auc=0.7)), mode="plain")
ck("rejects: PLAIN cell carrying a fusion block", (not ok2)
   and any("plain cell carries" in w for w in why2), "; ".join(why2)[:70])

print("\n=== D55: the REPORT independently verifies the per-cell eval-point set ===")
import s16_report as RP
cells, units, tags = L.expected_ledger()

def full_csv(mutate=None):
    """A complete, ledger-exact results table with every evaluation point present."""
    rows = []
    for uid, br, u in units:
        for tag in tags:
            proto = tag.rstrip("0123456789")
            base = dict(unit=uid, branch=br, arm=u["arm"], E=u["E"], arch=u["arch"],
                mode=u["mode"], seed=G.SEEDS[u["seed_idx"]], fold=tag,
                fold_protocol=proto, control=u.get("control"),
                alff_mode=u.get("alff_mode"), kh=u["kh"], svm_tr_enc=0.7565,
                svm_tr_full=0.7565, size_delta_paired=0.0, movement_max=0.50,
                clip_rate=0.05, ocread_entropy=np.nan, evaluated_state="raw")
            ctl = u.get("control"); sh = -0.044 if proto == "loso" else 0.0
            a = 0.50 if ctl == "C-PERM" else (0.5578+sh if ctl == "C-RAND" else 0.62+sh)
            for pt in P.EVAL_POINTS_PLAIN:
                rows.append({**base, "eval_point": pt, "auc": a, "alpha_selected": None,
                    "alpha1_equals_svm_tr_enc": None, "alpha1_bitwise_equals_zsFC": None,
                    "delta_vs_svm_tr_enc": None, "delta_vs_svm_tr_full": None})
            if u["mode"] == "fused":
                rows.append({**base, "eval_point": "fused", "auc": 0.76,
                    "alpha_selected": 1.0, "delta_vs_svm_tr_enc": 0.0035,
                    "delta_vs_svm_tr_full": 0.0035, "alpha1_equals_svm_tr_enc": True,
                    "alpha1_bitwise_equals_zsFC": True, "delta_is_unclamped": True})
    df = pd.DataFrame(rows)
    if mutate: df = mutate(df)
    return df

def run_report(df):
    shutil.rmtree(P.root(NS), ignore_errors=True); P.ensure(NS)
    df.to_csv(P.results_path(NS), index=False)
    r = subprocess.run([sys.executable, f"{HERE}/s16_report.py"], capture_output=True,
        text=True, env={**os.environ, "S16_NS": NS,
                        "PYTHONPYCACHEPREFIX": "/users/3171356m/agcl_audit_s0/pycache"})
    shutil.rmtree(P.root(NS), ignore_errors=True)
    return r

r0 = run_report(full_csv())
ck("1. complete paired table -> exit 0", r0.returncode == 0,
   f"exit {r0.returncode} {r0.stderr.strip().splitlines()[:1]}")

u0 = units[0][0]; t0 = tags[0]
def drop_one(df):
    m = (df.unit == u0) & (df.fold == t0) & (df.eval_point == "probe_honest")
    return df[~m]
r1 = run_report(full_csv(drop_one))
ck("removing ONE evaluation row -> nonzero exit", r1.returncode != 0, f"exit {r1.returncode}")
ck("   ...and the reason names the missing eval point",
   "eval_point_set_wrong" in r1.stderr and "probe_honest" in r1.stderr,
   [l.strip()[:88] for l in r1.stderr.splitlines() if "eval_point_set_wrong" in l][:1])

def dup_one(df):
    m = (df.unit == u0) & (df.fold == t0) & (df.eval_point == "head")
    return pd.concat([df, df[m]], ignore_index=True)
r2 = run_report(full_csv(dup_one))
ck("duplicating one evaluation row -> nonzero exit", r2.returncode != 0, f"exit {r2.returncode}")
ck("   ...and the reason names the duplicate",
   "duplicate_eval_point" in r2.stderr, [l.strip()[:80] for l in r2.stderr.splitlines()
                                          if "duplicate_eval_point" in l][:1])

def add_unexpected(df):
    m = (df.unit == u0) & (df.fold == t0) & (df.eval_point == "head")
    extra = df[m].copy(); extra["eval_point"] = "probe_secret"
    return pd.concat([df, extra], ignore_index=True)
r3 = run_report(full_csv(add_unexpected))
ck("an UNEXPECTED evaluation point -> nonzero exit", r3.returncode != 0, f"exit {r3.returncode}")
ck("   ...and the reason names it", "probe_secret" in r3.stderr,
   [l.strip()[:80] for l in r3.stderr.splitlines() if "probe_secret" in l][:1])

def plain_fused(df):
    pl = df[(df["mode"] == "plain")].unit.iloc[0]
    m = (df.unit == pl) & (df.fold == t0) & (df.eval_point == "head")
    extra = df[m].copy(); extra["eval_point"] = "fused"
    return pd.concat([df, extra], ignore_index=True)
r4 = run_report(full_csv(plain_fused))
ck("a PLAIN cell carrying a fused row -> nonzero exit", r4.returncode != 0, f"exit {r4.returncode}")

print("\n=== D54 dynamic: unpaired probe_honest row stops the headline ===")
def drop_shift(df):
    sh = df[(df.E == "shift") & (df.eval_point == "probe_honest") & (df.control.isna())]
    if not len(sh): return df
    return df.drop(sh.index[:1])
r5 = run_report(full_csv(drop_shift))
ck("removing one shift probe_honest row -> nonzero exit", r5.returncode != 0, f"exit {r5.returncode}")
ck("   ...reason is an eval-set or unpaired failure",
   ("eval_point_set_wrong" in r5.stderr) or ("unpaired shift/signed cells" in r5.stderr),
   [l.strip()[:88] for l in r5.stderr.splitlines()
    if "unpaired" in l or "eval_point_set_wrong" in l][:1])

def big_shift(df):
    m = (df.E == "shift") & (df.eval_point == "probe_honest") & (df.control.isna())
    df.loc[m, "auc"] = df.loc[m, "auc"] + 0.30      # huge, but FULLY paired
    return df
r6 = run_report(full_csv(big_shift))
ck("3. large but FULLY PAIRED AUC difference -> exit 0", r6.returncode == 0,
   f"exit {r6.returncode} — the withdrawn +/-0.01 gate stays withdrawn")
ck("   ...and the wording is descriptive", "DESCRIPTIVE DIAGNOSTIC ONLY" in r6.stdout)

# ================================================================= D56
print("\n=== D56: no test_*.py may be expected to fail ===")
ck("test_pass3_repro.py is gone", not os.path.exists(f"{HERE}/test_pass3_repro.py"))
ck("repro_pass3_historical.py exists outside the suite",
   os.path.exists(f"{HERE}/repro_pass3_historical.py"))
ME = os.path.basename(os.path.abspath(__file__))
for f in sorted(glob.glob(f"{HERE}/test_*.py")):
    if os.path.basename(f) == ME: continue      # this file states the pattern, so it
                                                # would match itself; exit codes for
                                                # every test_*.py are verified by the
                                                # Pass-4 sweep, not by string search
    t = open(f).read()
    ck(f"{os.path.basename(f)} declares no reproduction helper",
       "def repro(" not in t)

print(f"\n{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
