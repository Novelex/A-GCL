"""S16 provenance: run namespaces + provenance-safe resume.
Filename existence is NEVER sufficient to reuse a fold."""
import sys, os, json, hashlib, subprocess, numpy as np

S16 = "/users/3171356m/A-GCL/audit/s16/"
REPO = "/users/3171356m/A-GCL"
WORKER_VERSION = "s16_worker v5 (final-preflight: policy-driven, bundle-sealed)"
COLLECTOR_VERSION = "s16_collect v3 (final-preflight: strict bundle validator)"
NAMESPACES = ("prod", "e2e", "test")   # "test" exists solely for gate tests

# ---------------------------------------------------------------- namespaces
def root(ns):
    if ns not in NAMESPACES: raise ValueError(f"unknown namespace {ns!r}")
    return f"{S16}runs/{ns}/"
def jobs_dir(ns): return root(ns)+"jobs/"
def feat_dir(ns): return root(ns)+"feat/"
def ckpt_dir(ns): return root(ns)+"ckpt/"
def poison_path(ns): return root(ns)+"POISON"
def results_path(ns): return root(ns)+"s16_results.csv"
def ensure(ns):
    for d in (jobs_dir(ns), feat_dir(ns), ckpt_dir(ns)): os.makedirs(d, exist_ok=True)
    return root(ns)

# ---------------------------------------------------------------- runtime git
def git_sha():
    return subprocess.run(["git","-C",REPO,"rev-parse","HEAD"],
                          capture_output=True,text=True).stdout.strip()
def worktree_clean():
    out = subprocess.run(["git","-C",REPO,"status","--porcelain"],
                         capture_output=True,text=True).stdout.strip()
    return (out == ""), out

def builder_sha():
    """SHA recorded by the CACHE BUILDER, kept SEPARATE from the runtime SHA so a
    cache built at one commit and consumed at another is visible, not conflated."""
    import json as _j
    try: return _j.load(open(S16+"CACHE_MANIFEST.json")).get("git_sha")
    except Exception: return None

def environment():
    import platform, numpy, sklearn, torch as _t
    return dict(python=platform.python_version(), numpy=numpy.__version__,
                sklearn=sklearn.__version__, torch=_t.__version__,
                omp=os.environ.get("OMP_NUM_THREADS"),
                mkl=os.environ.get("MKL_NUM_THREADS"),
                torch_threads=_t.get_num_threads(),
                s11_njobs=os.environ.get("S11_NJOBS"),
                slurm_cpus=os.environ.get("SLURM_CPUS_PER_TASK"),
                host=__import__("socket").gethostname())

def sha_file(p, n=16):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()[:n]

def cfg_hash(unit, cfg, extra=None):
    """Stable hash of everything that defines the computation."""
    payload = dict(unit={k:v for k,v in sorted(unit.items())},
                   cfg={k:v for k,v in sorted(cfg.items())},
                   extra=extra or {})
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]

# ---------------------------------------------------------------- manifest
def expected_repr_dim(arch, kh, H=128, readout="roi"):
    """Representation width computed INDEPENDENTLY from architecture + configuration.
    NEVER read back from the manifest under validation (defect: circular check)."""
    if arch == "EDGEMLP": return 32                    # net.3 output
    if arch == "BNT":     return int(kh) * int(H)      # K clusters x H
    if arch == "WGIN":
        if readout == "roi": return 90 * int(kh)       # ROI-concat over 90 nodes
        if readout == "sum": return int(kh)
        raise ValueError(f"unknown WGIN readout {readout!r}")
    raise ValueError(f"unknown arch {arch!r}")

def effective_config(unit, cfg):
    """The FULL EFFECTIVE configuration actually used, not only its hash.
    A7's dropout is HARD-CODED 0.3 inside EdgeMLP regardless of cfg['dropout'] (which
    is 0.10 generically); recording the generic value alone would be false."""
    eff = dict(cfg)
    if unit["arch"] == "EDGEMLP":
        eff["dropout_effective"] = 0.3
        eff["dropout_argument_ignored"] = cfg.get("dropout")
        eff["head"] = "plain Linear(32,1)"
    else:
        eff["dropout_effective"] = cfg.get("dropout")
    eff["arch"] = unit["arch"]; eff["arm"] = unit["arm"]; eff["E"] = unit["E"]
    eff["mode"] = unit["mode"]; eff["control"] = unit.get("control")
    eff["alff_mode"] = unit.get("alff_mode"); eff["kh"] = unit["kh"]
    return eff

def model_cfg(unit):
    """THE model/optimizer config for a unit. Worker and collector both call this;
    duplicating the literals in two files was the same drift risk as D34."""
    return dict(K_or_hidden=unit["kh"], lr=3e-4, wd=1e-3, loss="L-BCE",
                freeze_encoder=(unit.get("control") == "C-RAND"), readout="roi",
                dropout=0.10, H=128)

MODEL_STATE_RULE = ("raw = validation-best checkpoint; EMA(0.999) evaluated "
                    "alongside and reported with the delta; selection by "
                    "VALIDATION only (S15 PROTOCOL.md:186)")

def contract_fields(ns, unit, cfg, uid, fold, seed, protocol, data_man, data_ent,
                    repr_dim, policy, train_consts):
    """THE contractual subset — exactly the MATCH_KEYS, built in ONE place.

    Defect D34: the worker's resume path reconstructed `epoch_policy` and
    `optimizer_recipe` inline, and its inline `optimizer_recipe` omitted `batch`,
    so a worker-built expectation could NEVER equal a build_manifest() record.
    Resume validation was therefore guaranteed to fail on every cell. Both callers
    now derive these fields from the SAME policy object that drives training."""
    return dict(
        schema="s16-prov-1", namespace=ns, git_sha=git_sha(),
        worker_version=WORKER_VERSION,
        config_hash=cfg_hash(unit, cfg, train_consts),
        h_fc=data_ent["h_fc"], h_alff=data_man["h_alff"],
        h_labels=data_man["h_labels"], h_subject_order=data_man["h_subject_order"],
        h_folds_lab=data_man["h_folds_lab"], h_folds_site=data_man["h_folds_site"],
        h_folds_loso=data_man["h_folds_loso"], cache_file=data_ent["cache_file"],
        unit=uid, arm=unit["arm"], arch=unit["arch"], E=unit["E"], mode=unit["mode"],
        control=unit.get("control"), alff_mode=unit.get("alff_mode"),
        seed=int(seed), fold=fold, protocol=protocol,
        epoch_policy=(policy.epoch_manifest() if policy else
                      {k: train_consts[k] for k in
                       ("max_epochs","min_epochs","patience","min_delta")}),
        optimizer_recipe=(policy.optimizer_manifest(cfg["lr"], cfg["wd"], cfg["loss"])
                          if policy else None),
        model_state_rule=MODEL_STATE_RULE,
        repr_dim=int(repr_dim),
        policy_hash=(policy.policy_hash() if policy else None))

def build_manifest(ns, unit, cfg, uid, fold, seed, protocol, data_man, data_ent,
                   repr_dim, feat_path, ckpt_path, status, train_consts,
                   policy=None, result_path=None, pred_path=None, effective_cfg=None):
    clean, dirt = worktree_clean()
    m = contract_fields(ns, unit, cfg, uid, fold, seed, protocol, data_man, data_ent,
                        repr_dim, policy, train_consts)
    # INFORMATIONAL fields — recorded for forensics, deliberately NOT in MATCH_KEYS.
    # worktree_clean is a GUARD (must be True), never a compared field (defect D27).
    m.update(
        worktree_clean=bool(clean), worktree_dirt=dirt[:400],
        collector_version=COLLECTOR_VERSION, builder_sha=builder_sha(),
        environment=environment(), status=status,
        policy_name=(policy.name if policy else None),
        effective_config=effective_cfg,
        feat_sha=sha_file(feat_path) if os.path.exists(feat_path) else None,
        ckpt_sha=sha_file(ckpt_path) if os.path.exists(ckpt_path) else None,
        result_sha=sha_file(result_path) if result_path and os.path.exists(result_path) else None,
        pred_sha=sha_file(pred_path) if pred_path and os.path.exists(pred_path) else None)
    return m

MATCH_KEYS = ("schema","namespace","git_sha","worker_version",
              "config_hash","h_fc","h_alff","h_labels","h_subject_order",
              "h_folds_lab","h_folds_site","h_folds_loso","cache_file","unit","arm",
              "arch","E","mode","control","alff_mode","seed","fold","protocol",
              "epoch_policy","optimizer_recipe","model_state_rule","repr_dim",
              "policy_hash")

def validate_reuse(man_path, expected, feat_path, ckpt_path):
    """Returns (ok, reason). A fold is reusable ONLY if every contracted field
    matches AND the recorded artifact hashes still match the files on disk."""
    if not os.path.exists(man_path): return False, "missing manifest"
    try: m = json.load(open(man_path))
    except Exception as e: return False, f"unreadable manifest: {e!r}"
    if m.get("status") != "OK": return False, f"recorded status={m.get('status')!r}"
    # CLEAN-TREE RULE: the ARTIFACT must have been produced from a clean tree.
    # We do NOT compare this to the tree's CURRENT state — that is irrelevant to the
    # artifact's validity, and comparing it made validation impossible whenever the
    # working tree was dirty (manifest True mismatches expected False; manifest False
    # trips this guard). The guard alone is the rule; `worktree_clean` is therefore
    # NOT in MATCH_KEYS.
    if not m.get("worktree_clean", False):
        return False, "manifest was produced from a DIRTY worktree"
    for k in MATCH_KEYS:
        if k not in m: return False, f"manifest missing field {k!r}"
        if k in expected and m[k] != expected[k]:
            return False, f"{k} mismatch: manifest={m[k]!r} expected={expected[k]!r}"
    if not os.path.exists(feat_path): return False, "feature file absent"
    if m.get("feat_sha") != sha_file(feat_path):
        return False, "feature file hash does not match manifest (corrupted/replaced)"
    if m.get("ckpt_sha") is not None:
        if not os.path.exists(ckpt_path): return False, "checkpoint absent"
        if m["ckpt_sha"] != sha_file(ckpt_path):
            return False, "checkpoint hash does not match manifest"
    return True, "validated"

def atomic_json(o, p):
    json.dump(o, open(p+".tmp","w"), indent=1, default=str)
    json.load(open(p+".tmp")); os.replace(p+".tmp", p)


EVAL_POINTS_PLAIN = ("head", "head_ema", "probe_honest", "probe_old_full")
EVAL_POINTS_FUSED = EVAL_POINTS_PLAIN + ("fused",)

def expected_eval_points(mode):
    """The EXACT evaluation-point set a cell must carry — no more, no fewer."""
    return set(EVAL_POINTS_FUSED if mode == "fused" else EVAL_POINTS_PLAIN)

def _finite01(v):
    try: f = float(v)
    except (TypeError, ValueError): return False
    return f == f and abs(f) != float("inf") and 0.0 <= f <= 1.0

def _finite(v):
    try: f = float(v)
    except (TypeError, ValueError): return False
    return f == f and abs(f) != float("inf")

def validate_eval_contract(rec, mode):
    """THE per-cell evaluation contract (defect D55), shared by the collector and the
    post-C6 report so the two cannot disagree about what a complete cell looks like.

    Previously the collector validated the sealed bundle and three scalar fields, and
    the report only checked that (unit, fold) appeared SOMEWHERE in the CSV. A cell
    missing three of its four evaluation points, or carrying a NaN AUC, or a plain
    cell smuggling a fusion block, passed both.

    Returns (ok, [reasons])."""
    why = []
    for pt in EVAL_POINTS_PLAIN:
        m = rec.get(pt)
        if not isinstance(m, dict):
            why.append(f"evaluation point {pt!r} is absent or not a dict"); continue
        if "auc" not in m:
            why.append(f"{pt}: no 'auc' key"); continue
        if not _finite01(m["auc"]):
            why.append(f"{pt}: auc={m['auc']!r} is not finite within [0,1]")
    for k in ("movement_max", "clip_rate"):
        if k not in rec: why.append(f"{k} is absent")
        elif not _finite(rec[k]): why.append(f"{k}={rec[k]!r} is not finite")
    es = str(rec.get("evaluated_state", ""))
    if not (es.startswith("raw=validation-best checkpoint") and "EMA(0.999)" in es
            and "selection by VALIDATION only" in es):
        why.append(f"evaluated_state does not match the frozen raw/EMA protocol: "
                   f"{es[:90]!r}")
    fu = rec.get("fusion")
    if mode == "fused":
        if not isinstance(fu, dict):
            # wording keeps the pre-existing collector contract ("fusion is not a
            # dict", test_final H32) while naming the cell-level problem
            why.append("fused cell carries no fusion block (fusion is not a dict)")
        elif not _finite01(fu.get("fused_auc")):
            why.append(f"fused_auc={fu.get('fused_auc')!r} is not finite within [0,1]")
    else:
        if fu:
            why.append("plain cell carries a fusion block (plain cells must not be fused)")
    return (not why), why

def validate_unit_completion(ns, uid, expected_folds):
    """THE unit-completion contract, shared by the collector and the E2E checker
    (defect D48) so the two definitions cannot drift apart.

    A sealed five-file bundle proves ONE FOLD was produced correctly. It says nothing
    about whether the UNIT finished: the E2E checker approved targets on bundle
    evidence alone, with no reference to POISON, TALLY.json, STATUS.json or UNIT.done,
    so a unit that was poisoned, or that recorded failures, or that never reached a
    terminal state, was still reported PASS.

    Returns (ok, [reasons])."""
    import json as _j
    why = []
    jd = jobs_dir(ns) + uid
    if os.path.exists(poison_path(ns)):
        why.append(f"GLOBAL POISON marker present for namespace {ns!r}")
    if os.path.exists(jd + "/POISON"):
        why.append(f"unit POISON marker: {open(jd+'/POISON').read().strip()[:120]}")
    tp = jd + "/TALLY.json"
    if not os.path.exists(tp):
        why.append("TALLY.json absent")
    else:
        try:
            t = _j.load(open(tp))
        except Exception as e:
            t = None; why.append(f"TALLY.json unreadable: {e!r}")
        if t is not None:
            # STRICT TYPING (defect D53). `remaining` was permitted to be 0 OR None,
            # so an ABSENT key passed: dict.get returned None and None was allowed.
            # A unit that never recorded how much work was left was therefore
            # indistinguishable from one that finished. bool is excluded explicitly
            # because in Python False == 0 and True == 1, so a boolean would satisfy
            # every numeric comparison below while meaning nothing.
            def _int(key, *, alt=None):
                """Require the key to be PRESENT and a real non-boolean integer."""
                if key in t:            v = t[key]
                elif alt and alt in t:  v = t[alt]; key = alt
                else:
                    why.append(f"TALLY field {key!r} is ABSENT (a unit that does not "
                               f"record it cannot be called complete)"); return None
                if isinstance(v, bool):
                    why.append(f"TALLY {key}={v!r} is a boolean, not an integer"); return None
                if not isinstance(v, int):
                    why.append(f"TALLY {key}={v!r} is {type(v).__name__}, not an integer"); return None
                if v < 0:
                    why.append(f"TALLY {key}={v} is negative"); return None
                return v
            exp     = _int("expected")
            reused  = _int("validated_reused")
            new_ok  = _int("newly_successful", alt="newly_succeeded")
            failed  = _int("failed")
            rem     = _int("remaining")
            if t.get("unit") != uid: why.append(f"TALLY unit {t.get('unit')!r} != {uid!r}")
            if t.get("namespace") != ns: why.append(f"TALLY namespace {t.get('namespace')!r} != {ns!r}")
            if exp is not None and exp != expected_folds:
                why.append(f"TALLY expected {exp} != {expected_folds}")
            if failed is not None and failed != 0:
                why.append(f"TALLY failed={failed} (must be 0)")
            if rem is not None and rem != 0:
                why.append(f"TALLY remaining={rem} (must be exactly 0)")
            if None not in (reused, new_ok):
                if reused > expected_folds or new_ok > expected_folds:
                    why.append(f"TALLY counts exceed expected_folds: reused {reused}, "
                               f"new {new_ok}, expected {expected_folds}")
                if reused + new_ok != expected_folds:
                    # Wording preserves the pre-existing collector contract
                    # ("reused N + new M != expected K") so test_final's H49
                    # expectation holds unchanged, while naming the identity.
                    why.append(f"accounting identity violated: reused {reused} + new {new_ok} "
                               f"!= expected {expected_folds} (validated_reused + "
                               f"newly_succeeded must equal expected_folds)")
    sp = jd + "/STATUS.json"
    if not os.path.exists(sp):
        why.append("STATUS.json absent")
    else:
        try:
            st = _j.load(open(sp)).get("state")
            if st != "done": why.append(f"STATUS state {st!r} is not the terminal 'done'")
        except Exception as e:
            why.append(f"STATUS.json unreadable: {e!r}")
    if not os.path.exists(jd + "/UNIT.done"):
        why.append("UNIT.done absent")
    return (not why), why

# ------------------------------------------------------------------ bundle validator
BUNDLE = ("result JSON", "provenance manifest", "feature npz", "checkpoint", "prediction JSON")

def validate_bundle(ns, uid, fold, expected, feat_path, ckpt_path, man_path,
                    result_path, pred_path):
    """THE single strict validator, shared by resume and collection.

    A cell is valid only if the ENTIRE bundle exists and agrees. A feature/checkpoint
    pair without its result and prediction is NOT reusable and must be recomputed.
    `expected` is reconstructed by the CALLER from the grid, ledger, policy and data
    manifest — never from the manifest being validated."""
    import json as _j
    for label, pth in (("provenance manifest", man_path), ("result JSON", result_path),
                       ("feature npz", feat_path), ("checkpoint", ckpt_path),
                       ("prediction JSON", pred_path)):
        if not os.path.exists(pth): return False, f"{label} absent"
    try: m = _j.load(open(man_path))
    except Exception as e: return False, f"unreadable manifest: {e!r}"
    if m.get("status") != "OK": return False, f"recorded status={m.get('status')!r}"
    # CLEAN-TREE RULE: the ARTIFACT must have been produced from a clean tree.
    # We do NOT compare this to the tree's CURRENT state — that is irrelevant to the
    # artifact's validity, and comparing it made validation impossible whenever the
    # working tree was dirty (manifest True mismatches expected False; manifest False
    # trips this guard). The guard alone is the rule; `worktree_clean` is therefore
    # NOT in MATCH_KEYS.
    if not m.get("worktree_clean", False):
        return False, "manifest was produced from a DIRTY worktree"
    for k in MATCH_KEYS:
        if k not in m: return False, f"manifest missing field {k!r}"
        if k in expected and m[k] != expected[k]:
            return False, f"{k} mismatch: manifest={m[k]!r} expected={expected[k]!r}"
    if m.get("unit") != uid:  return False, f"manifest unit {m.get('unit')!r} != {uid!r}"
    if m.get("fold") != fold: return False, f"manifest fold {m.get('fold')!r} != {fold!r}"
    for label, pth, key in (("feature npz", feat_path, "feat_sha"),
                            ("checkpoint", ckpt_path, "ckpt_sha"),
                            ("result JSON", result_path, "result_sha"),
                            ("prediction JSON", pred_path, "pred_sha")):
        rec = m.get(key)
        if rec is None: return False, f"{label} hash not recorded ({key})"
        if rec != sha_file(pth): return False, f"{label} hash mismatch (corrupted/replaced)"
    return True, "validated"
