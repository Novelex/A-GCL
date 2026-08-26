"""S16 Correction Gate 3 tests: namespace isolation + provenance-safe resume.
No cluster jobs, no scientific results, no deletion of existing artifacts."""
import sys, os, json, glob, shutil, copy, tempfile, numpy as np
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_prov as P, s16_grid as G
F=[]
def ck(n,ok,d=""):
    print(("PASS " if ok else "FAIL ")+n+(" | "+d if d else ""),flush=True)
    if not ok: F.append(n)

# ---------------- A. NAMESPACE ISOLATION ----------------
print("--- A. namespace isolation ---")
ck("A1_roots_disjoint", P.root("prod")!=P.root("e2e")
   and not P.root("prod").startswith(P.root("e2e"))
   and not P.root("e2e").startswith(P.root("prod")),
   f"prod={P.root('prod')} e2e={P.root('e2e')}")
for ns in ("prod","e2e"): P.ensure(ns)
# a poison marker in e2e must not be visible to prod, and vice versa
open(P.poison_path("e2e"),"w").write("synthetic e2e poison\n")
ck("A2_poison_does_not_cross", os.path.exists(P.poison_path("e2e"))
   and not os.path.exists(P.poison_path("prod")),
   "e2e POISON present, prod POISON absent")
os.remove(P.poison_path("e2e"))
# legacy top-level artifacts must not satisfy either namespace
legacy = glob.glob("/users/3171356m/A-GCL/audit/s16/feat/*.npz")
ck("A3_legacy_outside_namespaces",
   all(not f.startswith(P.feat_dir("prod")) and not f.startswith(P.feat_dir("e2e"))
       for f in legacy),
   f"{len(legacy)} legacy npz live at the OLD top-level path, invisible to both namespaces")

# ---------------- B. E2E CREATES NO PRODUCTION FILE ----------------
print("\n--- B. E2E creates no production file ---")
def snap(ns): return set(glob.glob(P.root(ns)+"**/*", recursive=True))
before_prod = snap("prod")
import s16_train as TR, s16_data as DAT
TR.MAX_EPOCHS, TR.MIN_EPOCHS, TR.PATIENCE = 2, 1, 1
import s16_worker as W
DAT.folds_orig = DAT.folds
DAT.folds = lambda d,p: (DAT.folds_orig(d,p)[:1] if p=="lab" else [])
i = next(k for k,u in enumerate(G.MAIN) if u["arm"]=="A7" and u["mode"]=="plain"
         and u["E"]=="signed" and u["seed_idx"]==0)
W.run("main", i, ns="e2e")
after_prod = snap("prod"); e2e_files = snap("e2e")
ck("B1_e2e_wrote_nothing_to_prod", before_prod==after_prod,
   f"prod tree unchanged ({len(before_prod)} entries before and after)")
ck("B2_e2e_wrote_into_e2e", len(e2e_files)>0, f"{len(e2e_files)} entries under runs/e2e/")
uid = G.unit_id(G.MAIN[i])
fp  = P.feat_dir("e2e")+f"{uid}__lab0.npz"; mfp = fp+".prov.json"
ck("B3_manifest_written", os.path.exists(mfp), os.path.basename(mfp))

# ---------------- C. PROVENANCE-SAFE RESUME, 10 CASES ----------------
print("\n--- C. provenance-safe resume ---")
man = json.load(open(mfp)); ckp = P.ckpt_dir("e2e")+f"{uid}__lab0.pt"

# The live worktree is DIRTY (gate-2/3 edits are uncommitted), so every manifest
# written now carries worktree_clean=False and the dirty guard rejects reuse. That is
# CORRECT for production, but it would MASK every field test below — each case would
# be rejected before the field under test is even compared. Each mutation is therefore
# run against a SYNTHETIC CLEAN manifest, and the rejection REASON is asserted, not
# merely the fact of rejection. The dirty guard itself is tested as its own case.
clean_man = copy.deepcopy(man); clean_man["worktree_clean"]=True; clean_man["worktree_dirt"]=""
cmf = mfp+".clean"; json.dump(clean_man, open(cmf,"w"), default=str)
exp = {k: clean_man[k] for k in P.MATCH_KEYS}

ok,why = P.validate_reuse(cmf, exp, fp, ckp)
ck("C1_valid_resume", ok, why)

def case(label, field, value, expect_sub, expmod=None):
    """Mutate ONE field of a CLEAN manifest; assert the rejection names that field."""
    tmp = mfp+".case"; m2 = copy.deepcopy(clean_man)
    if field is not None: m2[field] = value
    json.dump(m2, open(tmp,"w"), default=str)
    e = copy.deepcopy(exp)
    if expmod: e.update(expmod)
    o,w = P.validate_reuse(tmp, e, fp, ckp); os.remove(tmp)
    ck(label, (not o) and (expect_sub in w), f"rejected: {w}")

case("C2_sha_mismatch","git_sha","0"*40,"git_sha mismatch")
# dirty-state guard, tested in isolation
tmp = mfp+".dirty"; md = copy.deepcopy(clean_man); md["worktree_clean"]=False
json.dump(md, open(tmp,"w"), default=str)
o,w = P.validate_reuse(tmp, exp, fp, ckp); os.remove(tmp)
ck("C3_dirty_state_mismatch", (not o) and "DIRTY" in w, f"rejected: {w}")
case("C4_config_mismatch","config_hash","deadbeefdeadbeef","config_hash mismatch")
ep = copy.deepcopy(clean_man["epoch_policy"]); ep["min_epochs"]=999
case("C5_epoch_mismatch","epoch_policy",ep,"epoch_policy mismatch")
o,w = P.validate_reuse(mfp+".nonexistent", exp, fp, ckp)
ck("C6_missing_manifest", (not o) and "missing manifest" in w, f"rejected: {w}")
case("C7_failed_record","status","FAILED","status")
bad = fp+".corrupt"; shutil.copy(fp,bad)
with open(bad,"ab") as f: f.write(b"\x00corrupt")
o,w = P.validate_reuse(cmf, exp, bad, ckp)
ck("C8_corrupted_feature", (not o) and "hash does not match" in w, f"rejected: {w}")
os.remove(bad)
e_prod = copy.deepcopy(exp); e_prod["namespace"]="prod"
o,w = P.validate_reuse(cmf, e_prod, fp, ckp)
ck("C9_e2e_artifact_refused_by_prod", (not o) and "namespace mismatch" in w, f"rejected: {w}")
case("C10_prod_artifact_refused_by_e2e","namespace","prod","namespace mismatch")
orc = copy.deepcopy(clean_man["optimizer_recipe"]); orc["lr"]=9.9
case("C11_optimizer_recipe_mismatch","optimizer_recipe",orc,"optimizer_recipe mismatch")
case("C12_model_state_rule_mismatch","model_state_rule","EMA only","model_state_rule mismatch")
case("C13_repr_dim_mismatch","repr_dim",7,"repr_dim mismatch")
case("C14_data_hash_mismatch","h_fc","0000000000000000","h_fc mismatch")
case("C15_worker_version_mismatch","worker_version","old worker","worker_version mismatch")
case("C16_seed_mismatch","seed",999,"seed mismatch")
case("C17_protocol_mismatch","protocol","loso","protocol mismatch")
os.remove(cmf)

# ---------------- D. ACCOUNTING ----------------
print("\n--- D. accounting ---")
t = json.load(open(P.jobs_dir("e2e")+uid+"/TALLY.json"))
ck("D1_accounting_identity",
   t["validated_reused"]+t["newly_succeeded"] == t["expected_folds"]-t["failed"],
   f"reused {t['validated_reused']} + new {t['newly_succeeded']} = "
   f"{t['validated_reused']+t['newly_succeeded']} ; expected {t['expected_folds']} "
   f"minus failed {t['failed']}")
# Second-run reuse cannot be exercised while the worktree is dirty — the guard
# (correctly) forces recomputation. Reuse is therefore proven against a manifest
# stamped clean, with the expectation stamped clean to match.
cm = copy.deepcopy(man); cm["worktree_clean"]=True; cm["worktree_dirt"]=""
cmf2 = mfp+".reuse"; json.dump(cm, open(cmf2,"w"), default=str)
e2 = {k: cm[k] for k in P.MATCH_KEYS}
o,w = P.validate_reuse(cmf2, e2, fp, ckp)
ck("D2_reuse_accepted_when_contract_matches", o, w)
os.remove(cmf2)
# D58: this assertion used to read `not P.worktree_clean()[0]` — it required the
# LIVE repository to be dirty, so it passed only while correction work was
# uncommitted and FAILED on the clean committed tree that production demands. The
# guard being tested is a property of the MANIFEST, not of the ambient checkout, so
# it is now exercised synthetically: a hash-consistent manifest stamped
# worktree_clean=False must be REFUSED by the real reuse validator, whatever state
# the live repository happens to be in.
dm = copy.deepcopy(man)
dm["worktree_clean"] = False
dm["worktree_dirt"] = " M audit/s16/scripts/s16_train.py\n M audit/s16/scripts/s16_prov.py"
dmf = mfp + ".dirty"
json.dump(dm, open(dmf, "w"), default=str)
e3 = {k: dm[k] for k in P.MATCH_KEYS}          # contract matches exactly
o3, w3 = P.validate_reuse(dmf, e3, fp, ckp)    # the REAL validator
os.remove(dmf)
ck("D3_dirty_tree_forces_recompute",
   (not o3) and ("DIRTY worktree" in w3),
   f"a manifest stamped worktree_clean=False is refused even though every "
   f"contracted field matches -> {w3!r}; the artifact, not the live checkout, "
   f"decides reuse (D58)")

print(f"\n=== GATE 3 SUMMARY: {len(F)} failures ===")
for f in F: print("  FAIL "+f)
sys.exit(1 if F else 0)
