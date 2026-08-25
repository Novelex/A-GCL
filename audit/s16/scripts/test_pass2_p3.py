"""Pass 2, P3: worker / manifest / collector must agree by construction.

Defect D34: the worker's resume path reconstructed `epoch_policy` and
`optimizer_recipe` inline and its `optimizer_recipe` omitted `batch`, so a
worker-built expectation could never equal a build_manifest() record: every
resumable cell would be silently recomputed, and the collector's third
independent copy of the same dict was a standing divergence risk.

Fix: contract_fields() builds the MATCH_KEYS in ONE place; worker, manifest and
collector all call it, and all derive the epoch/optimizer blocks from the same
ExecPolicy that drives training."""
import sys, os, json, copy, tempfile, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s16_prov as P, s16_policy as PL, s16_collect as C
OK = []
def check(c, m): OK.append(bool(c)); print(f"  [{'PASS' if c else 'FAIL'}] {m}")

POL = PL.PROD
UNIT = dict(arm="A4", arch="WGIN", E="signed", mode="M1", control=None,
            alff_mode="raw", kh=128, seed=20260818)
UID  = "A4_WGIN_signed_M1"
CFG  = P.model_cfg(UNIT)      # the SAME builder the worker and collector use
MAN  = {f"h_{k}": f"hash_{k}" for k in ("alff","labels","subject_order",
                                        "folds_lab","folds_site","folds_loso")}
ENT  = dict(h_fc="hash_fc", cache_file="fc_signed.npz", sparse=False)
TC   = POL.train_consts()
ARGS = ("prod", UNIT, CFG, UID, "lab0", 20260818, "lab", MAN, ENT, 11520)

print("=== 1. contract_fields is the same object for every caller ===")
exp = P.contract_fields(*ARGS, POL, TC)
check(set(exp) == set(P.MATCH_KEYS),
      f"contract_fields returns EXACTLY the {len(P.MATCH_KEYS)} MATCH_KEYS")
check("batch" in exp["optimizer_recipe"],
      f"optimizer_recipe carries batch={exp['optimizer_recipe'].get('batch')} "
      f"(the omitted key that rejected all 1,431 cells)")
check(exp["optimizer_recipe"] == POL.optimizer_manifest(CFG["lr"], CFG["wd"], CFG["loss"]),
      "optimizer_recipe is policy.optimizer_manifest() verbatim")
check(exp["epoch_policy"] == POL.epoch_manifest(), "epoch_policy is policy.epoch_manifest()")

print("\n=== 2. round trip: build_manifest -> validate_reuse REUSES ===")
with tempfile.TemporaryDirectory() as td:
    fp, ckp, mfp = f"{td}/f.npz", f"{td}/c.pt", f"{td}/f.npz.prov.json"
    open(fp, "wb").write(b"FEATBYTES"); open(ckp, "wb").write(b"CKPTBYTES")
    man = P.build_manifest("prod", UNIT, CFG, UID, "lab0", 20260818, "lab", MAN, ENT,
                           11520, fp, ckp, "OK", TC, policy=POL)
    man["worktree_clean"] = True          # simulate the clean-tree production rule
    P.atomic_json(man, mfp)
    for k in P.MATCH_KEYS:
        if man[k] != exp[k]:
            check(False, f"manifest[{k}] != worker expectation"); break
    else:
        check(True, f"all {len(P.MATCH_KEYS)} contracted fields agree, field by field")
    ok, why = P.validate_reuse(mfp, exp, fp, ckp)
    check(ok, f"validate_reuse -> {ok} ({why})")

    print("\n=== 3. mutation tests: each contracted field must REJECT ===")
    for field, bad in (("config_hash", "deadbeef"),
                       ("policy_hash", "0000000000000000"),
                       ("optimizer_recipe", {"opt": "SGD"}),
                       ("epoch_policy", {"max_epochs": 4})):
        m2 = copy.deepcopy(man); m2[field] = bad
        P.atomic_json(m2, mfp)
        ok2, why2 = P.validate_reuse(mfp, exp, fp, ckp)
        check((not ok2) and why2.startswith(f"{field} mismatch"),
              f"mutated {field} -> rejected: {why2[:72]}")
    # a missing contracted field must also reject
    m3 = copy.deepcopy(man); m3.pop("model_state_rule")
    P.atomic_json(m3, mfp)
    ok3, why3 = P.validate_reuse(mfp, exp, fp, ckp)
    check((not ok3) and "missing field" in why3, f"missing field -> rejected: {why3}")
    # informational fields must NOT reject
    m4 = copy.deepcopy(man); m4["node"] = "elsewhere"; m4["wall_s"] = 999
    m4["policy_name"] = "renamed"; m4["environment"] = {"x": 1}
    P.atomic_json(m4, mfp)
    ok4, why4 = P.validate_reuse(mfp, exp, fp, ckp)
    check(ok4, f"informational-only edits still validate ({why4})")
    # tampered artifact must reject
    P.atomic_json(man, mfp); open(fp, "wb").write(b"TAMPERED")
    ok5, why5 = P.validate_reuse(mfp, exp, fp, ckp)
    check((not ok5) and "hash does not match" in why5, f"tampered feature file -> {why5}")

print("\n=== 4. collector expectation == worker expectation ===")
cexp = C.expected_contract("prod", UID, UNIT, "lab0", 20260818, MAN, ENT, POL)
diff = [k for k in P.MATCH_KEYS if cexp.get(k) != exp.get(k)]
check(not diff, f"collector and worker expectations identical (diff={diff})")

print("\n=== 5. namespace is REQUIRED in the result record ===")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "s16_collect.py")).read()
check('rec.get("namespace", ns)' not in src, "no silent rec.get('namespace', ns) fallback")
check('if "namespace" not in rec' in src, "absent namespace is an explicit rejection")
wsrc = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "s16_worker.py")).read()
check("namespace=ns," in wsrc and "policy_hash=policy.policy_hash()" not in wsrc,
      "worker writes an explicit namespace into the result record; policy_hash is "
      "NOT re-passed (it arrives via **info from the same policy object — passing "
      "both raises the duplicate-keyword TypeError that killed all 288 S15 units)")
tsrc = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "s16_train.py")).read()
check("info = dict(policy_name=policy.name, policy_hash=policy.policy_hash()," in tsrc,
      "policy_name/policy_hash reach the record through train_fold's info dict")

print(f"\n{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
