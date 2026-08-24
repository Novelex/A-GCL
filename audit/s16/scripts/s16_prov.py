"""S16 provenance: run namespaces + provenance-safe resume.
Filename existence is NEVER sufficient to reuse a fold."""
import sys, os, json, hashlib, subprocess, numpy as np

S16 = "/users/3171356m/A-GCL/audit/s16/"
REPO = "/users/3171356m/A-GCL"
WORKER_VERSION = "s16_worker v4 (gate6: fusion endpoint, fold baselines, predictions)"
COLLECTOR_VERSION = "s16_collect v2 (gate4: ledger-driven refusal)"
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
def build_manifest(ns, unit, cfg, uid, fold, seed, protocol, data_man, data_ent,
                   repr_dim, feat_path, ckpt_path, status, train_consts):
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
        epoch_policy=dict(max_epochs=train_consts["max_epochs"],
                          min_epochs=train_consts["min_epochs"],
                          patience=train_consts["patience"],
                          min_delta=train_consts["min_delta"]),
        optimizer_recipe=dict(opt="AdamW", lr=cfg["lr"], wd=cfg["wd"],
                              betas=[0.9,0.999], eps=1e-8,
                              warmup_frac=train_consts["warmup_frac"],
                              cosine_floor=train_consts["cosine_floor"],
                              clip="adaptive p90 of last 200, no clip for first 50 steps",
                              label_smooth=train_consts["label_smooth"],
                              loss=cfg["loss"]),
        model_state_rule=("raw = validation-best checkpoint; EMA(0.999) evaluated "
                          "alongside and reported with the delta; selection by "
                          "VALIDATION only (S15 PROTOCOL.md:186)"),
        repr_dim=int(repr_dim), status=status,
        feat_sha=sha_file(feat_path) if os.path.exists(feat_path) else None,
        ckpt_sha=sha_file(ckpt_path) if os.path.exists(ckpt_path) else None)

MATCH_KEYS = ("schema","namespace","git_sha","worktree_clean","worker_version",
              "config_hash","h_fc","h_alff","h_folds_lab","h_folds_site",
              "h_folds_loso","cache_file","unit","arm","arch","E","mode","control",
              "alff_mode","seed","fold","protocol","epoch_policy",
              "optimizer_recipe","model_state_rule","repr_dim")

def validate_reuse(man_path, expected, feat_path, ckpt_path):
    """Returns (ok, reason). A fold is reusable ONLY if every contracted field
    matches AND the recorded artifact hashes still match the files on disk."""
    if not os.path.exists(man_path): return False, "missing manifest"
    try: m = json.load(open(man_path))
    except Exception as e: return False, f"unreadable manifest: {e!r}"
    if m.get("status") != "OK": return False, f"recorded status={m.get('status')!r}"
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
