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

def build_manifest(ns, unit, cfg, uid, fold, seed, protocol, data_man, data_ent,
                   repr_dim, feat_path, ckpt_path, status, train_consts,
                   policy=None, result_path=None, pred_path=None, effective_cfg=None):
    clean, dirt = worktree_clean()
    return dict(
        schema="s16-prov-1", namespace=ns,
        git_sha=git_sha(), worktree_clean=bool(clean),
        worktree_dirt=dirt[:400], worker_version=WORKER_VERSION,
        collector_version=COLLECTOR_VERSION, builder_sha=builder_sha(),
        environment=environment(),
        config_hash=cfg_hash(unit, cfg, train_consts),
        h_fc=data_ent["h_fc"], h_alff=data_man["h_alff"],
        h_folds_lab=data_man["h_folds_lab"], h_folds_site=data_man["h_folds_site"],
        h_folds_loso=data_man["h_folds_loso"], cache_file=data_ent["cache_file"],
        unit=uid, arm=unit["arm"], arch=unit["arch"], E=unit["E"], mode=unit["mode"],
        control=unit.get("control"), alff_mode=unit.get("alff_mode"),
        seed=int(seed), fold=fold, protocol=protocol,
        # SINGLE SOURCE OF TRUTH: both of these come from the POLICY object, the same
        # one that drives training. Constructing them inline here diverged from the
        # collector's expectation (a missing `batch` key) and rejected every cell.
        epoch_policy=(policy.epoch_manifest() if policy else
                      dict(max_epochs=train_consts["max_epochs"],
                           min_epochs=train_consts["min_epochs"],
                           patience=train_consts["patience"],
                           min_delta=train_consts["min_delta"])),
        optimizer_recipe=(policy.optimizer_manifest(cfg["lr"], cfg["wd"], cfg["loss"])
                          if policy else None),
        model_state_rule=("raw = validation-best checkpoint; EMA(0.999) evaluated "
                          "alongside and reported with the delta; selection by "
                          "VALIDATION only (S15 PROTOCOL.md:186)"),
        repr_dim=int(repr_dim), status=status,
        policy_name=(policy.name if policy else None),
        policy_hash=(policy.policy_hash() if policy else None),
        effective_config=effective_cfg,
        h_labels=data_man["h_labels"], h_subject_order=data_man["h_subject_order"],
        feat_sha=sha_file(feat_path) if os.path.exists(feat_path) else None,
        ckpt_sha=sha_file(ckpt_path) if os.path.exists(ckpt_path) else None,
        result_sha=sha_file(result_path) if result_path and os.path.exists(result_path) else None,
        pred_sha=sha_file(pred_path) if pred_path and os.path.exists(pred_path) else None)

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
