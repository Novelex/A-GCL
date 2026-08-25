"""Pass 2, P1+P2: REAL train_fold() tests for C-RAND (frozen random encoder).

Defect D32 (P1): assert_groups_cover() skipped frozen parameters and then demanded
every group be nonempty, so BNT/WGIN C-RAND raised "group 'inp' is empty" and NO
C-RAND unit could train. The negative control was unrunnable.

Defect D33 (P2): freezing set requires_grad=False but left the encoder in TRAIN mode,
so encoder dropout kept resampling. A "fixed random encoder" that is not fixed makes
the control's representation non-reproducible.

These tests call the real train_fold() end-to-end. Testing assert_groups_cover() with
a synthetic fixture alone would not have caught either defect in situ."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s16_models as MO, s16_train as TR, s16_policy as PL

# Test-only policy: real code path, tiny budget. PROD/E2E/TEST are untouched.
TINY = PL.ExecPolicy("p1p2_test", "test", 3, 1, 2, 1e-5, 0.10, 0.05, 0.05, 16,
                     0.999, 5, 20, 90, 1, 0, 0, False)
N, R, SEED = 64, 90, 20260818
OK = []

def synth(D):
    rng = np.random.default_rng(7)
    X  = rng.normal(size=(N, R, D)).astype(np.float32)
    A  = rng.normal(size=(N, R, R)).astype(np.float32)
    FC = ((A + A.transpose(0, 2, 1)) / 2).astype(np.float32)
    for i in range(N): np.fill_diagonal(FC[i], 1.0)
    y  = np.array([0, 1] * (N // 2))
    return X, FC, y

def cfg_for(arch, frozen):
    return dict(K_or_hidden=(4 if arch == "BNT" else 96), lr=1e-3, wd=1e-4,
                loss="L-BCE", freeze_encoder=frozen, readout="roi", dropout=0.3,
                H=(96 if arch == "BNT" else 32))

def enc_mods(m, arch):
    return [m.inp, m.blocks, m.norm_f] if arch == "BNT" else [m.inp, m.convs, m.norms, m.drop]

def enc_state(m, arch):
    pre = ("inp.", "blocks.", "norm_f.") if arch == "BNT" else ("inp.", "convs.", "norms.")
    sd = m.state_dict()   # parameters AND buffers
    return {k: v.detach().clone() for k, v in sd.items() if k.startswith(pre)}

def check(cond, msg):
    OK.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

for arch in ("BNT", "WGIN"):
    print(f"\n=== {arch} C-RAND via real train_fold() ===")
    D = 93; X, FC, y = synth(D); tr = np.arange(N)
    cfg = cfg_for(arch, True)

    # Reference weights BEFORE training, from an identically-seeded build.
    ref = MO.build_model(arch, D, SEED, cfg["K_or_hidden"], freeze_encoder=True,
                         readout=cfg["readout"], p=cfg["dropout"], H=cfg["H"])
    ref_enc = enc_state(ref, arch)
    ref_head = {k: v.detach().clone() for k, v in ref.state_dict().items()
                if k.startswith("head.")}

    # P1: this raised AssertionError("group 'inp' is empty") before the fix.
    model, ema_sd, curve, info = TR.train_fold(arch, X, FC, y, tr, cfg, SEED,
                                               policy=TINY)
    check(True, "train_fold() completed — assert_groups_cover no longer blocks C-RAND")
    check(len(curve) >= 1, f"training actually ran ({len(curve)} epochs recorded)")

    # P2a: frozen encoder stays in eval mode even when the model is set to train().
    model.train()
    check(not any(m_.training for m_ in enc_mods(model, arch)),
          "encoder modules are in EVAL mode after .train() (dropout disabled)")
    check(model.head.training, "head is in TRAIN mode")

    # P2b: repeated representations are BITWISE identical.
    r1, s1 = TR.extract(model, X, FC, np.arange(N), arch == "WGIN")
    r2, s2 = TR.extract(model, X, FC, np.arange(N), arch == "WGIN")
    check(np.array_equal(r1, r2), f"representations bitwise identical (maxdiff "
                                  f"{np.abs(r1-r2).max():.3e})")
    check(np.array_equal(s1, s2), "logits bitwise identical")

    # P2c: encoder weights AND buffers unchanged by training; head changed.
    got_enc = enc_state(model, arch)
    check(set(got_enc) == set(ref_enc), f"encoder state keys match ({len(ref_enc)} tensors)")
    worst = max((float((got_enc[k] - ref_enc[k]).abs().max()) for k in ref_enc), default=0.0)
    check(worst == 0.0, f"encoder weights+buffers UNCHANGED (max |delta| {worst:.3e})")
    head_moved = any(not torch.equal(model.state_dict()[k], v) for k, v in ref_head.items())
    check(head_moved, "head parameters DID change (the head trained)")

    # P2d: an UNFROZEN model must still train its encoder — the fix is not global.
    m2, _, _, _ = TR.train_fold(arch, X, FC, y, tr, cfg_for(arch, False), SEED, policy=TINY)
    m2.train()
    check(all(m_.training for m_ in enc_mods(m2, arch)),
          "unfrozen model: encoder still in TRAIN mode (fix is scoped to C-RAND)")
    e2 = enc_state(m2, arch)
    moved = max(float((e2[k] - ref_enc[k]).abs().max()) for k in ref_enc)
    check(moved > 0.0, f"unfrozen model: encoder weights DID move (max |delta| {moved:.3e})")

print("\n=== negative controls: detection must NOT be weakened ===")
import copy
m = MO.build_model("BNT", 93, SEED, 4, H=96)
saved = copy.deepcopy(TR.GROUPS["BNT"])
try:
    TR.GROUPS["BNT"] = {"inp": ("inp.",), "enc": ("blocks.",), "head": ("head.",)}
    try: TR.assert_groups_cover(m, "BNT"); check(False, "unknown prefix (norm_f) not caught")
    except AssertionError as e: check("maps to 0 groups" in str(e), f"unknown prefix caught: {e}")
    TR.GROUPS["BNT"] = {"inp": ("inp.",), "enc": ("blocks.", "norm_f.", "inp."), "head": ("head.",)}
    try: TR.assert_groups_cover(m, "BNT"); check(False, "overlapping prefix not caught")
    except AssertionError as e: check("maps to 2 groups" in str(e), f"overlap caught: {e}")
    TR.GROUPS["BNT"] = {"inp": ("inp.",), "enc": ("blocks.", "norm_f."),
                        "head": ("head.",), "ghost": ("nothing_.",)}
    try: TR.assert_groups_cover(m, "BNT"); check(False, "empty declared group not caught")
    except AssertionError as e: check("NO parameters at all" in str(e), f"empty group caught: {e}")
finally:
    TR.GROUPS["BNT"] = saved

mh = MO.build_model("BNT", 93, SEED, 4, H=96)
for n_, q in mh.named_parameters():
    if n_.startswith("head."): q.requires_grad_(False)
try: TR.assert_groups_cover(mh, "BNT"); check(False, "frozen head not caught")
except AssertionError as e: check("must always train" in str(e), f"frozen head caught: {e}")

print(f"\n{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
