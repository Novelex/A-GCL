"""S16 IMMUTABLE EXECUTION POLICY.

One frozen object controls, simultaneously and inseparably:
  * the actual training loop (epochs, stopping, schedule, clipping, smoothing)
  * the config hash
  * the epoch manifest recorded in provenance
  * resume validation
  * the result record
  * which folds are run

WHY THIS EXISTS. `_e2e_run.py` used to mutate `s16_train` module globals AFTER
`s16_worker` had already snapshotted them into `TRAIN_CONSTS`. Training then ran for
4 epochs while provenance recorded 400 — the record was false. Module-level mutable
state cannot be made safe by ordering; the policy is passed explicitly instead."""
from dataclasses import dataclass, asdict, replace
import hashlib, json

@dataclass(frozen=True)
class ExecPolicy:
    name: str                    # "prod" | "e2e"
    namespace: str               # artifact namespace; must match `name`'s intent
    max_epochs: int
    min_epochs: int
    patience: int
    min_delta: float
    warmup_frac: float
    cosine_floor: float
    label_smooth: float          # target t = y*(1-s) + s/2   (see LABEL_SMOOTHING note)
    batch: int
    ema_decay: float
    clip_warmup_steps: int
    clip_window: int
    clip_pctl: int
    n_lab: int                   # folds taken per protocol
    n_site: int
    n_loso: int
    scientific: bool             # False => miniature/correctness only, never a result

    def as_dict(self): return asdict(self)

    def target_from(self, y):
        """t = y*(1-s) + s/2. At s=0.05 this is y*0.95 + 0.025.
        NOTE: S15 PROTOCOL.md:188 prints 't=y*0.90+0.05', which is the s=0.10 formula
        carried over from S13 and never updated when the NAMED value became 0.05
        (S15 PROTOCOL.md:184, 'label smoothing 0.05 (was 0.10)'). The named value and
        both S15/S16 implementations agree on 0.05; the printed formula is the stale
        item. Documentation corrected, training UNCHANGED."""
        return y * (1.0 - self.label_smooth) + self.label_smooth / 2.0

    def epoch_manifest(self):
        return dict(max_epochs=self.max_epochs, min_epochs=self.min_epochs,
                    patience=self.patience, min_delta=self.min_delta)

    def optimizer_manifest(self, lr, wd, loss):
        return dict(opt="AdamW", lr=lr, wd=wd, betas=[0.9, 0.999], eps=1e-8,
                    warmup_frac=self.warmup_frac, cosine_floor=self.cosine_floor,
                    clip=(f"adaptive p{self.clip_pctl} of last {self.clip_window}, "
                          f"no clip for first {self.clip_warmup_steps} steps"),
                    label_smooth=self.label_smooth, batch=self.batch, loss=loss)

    def policy_hash(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True)
                              .encode()).hexdigest()[:16]

PROD = ExecPolicy(name="prod", namespace="prod", max_epochs=400, min_epochs=80,
                  patience=50, min_delta=1e-5, warmup_frac=0.10, cosine_floor=0.05,
                  label_smooth=0.05, batch=32, ema_decay=0.999,
                  clip_warmup_steps=50, clip_window=200, clip_pctl=90,
                  n_lab=3, n_site=3, n_loso=3, scientific=True)

E2E = ExecPolicy(name="e2e", namespace="e2e", max_epochs=4, min_epochs=2,
                 patience=2, min_delta=1e-5, warmup_frac=0.10, cosine_floor=0.05,
                 label_smooth=0.05, batch=32, ema_decay=0.999,
                 clip_warmup_steps=50, clip_window=200, clip_pctl=90,
                 n_lab=1, n_site=0, n_loso=0, scientific=False)

# TEST mirrors PROD's TRAINING parameters exactly so fixtures exercise the real
# contract, but lives in its own namespace and is never scientific.
TEST = ExecPolicy(name="test", namespace="test", max_epochs=PROD.max_epochs,
                  min_epochs=PROD.min_epochs, patience=PROD.patience,
                  min_delta=PROD.min_delta, warmup_frac=PROD.warmup_frac,
                  cosine_floor=PROD.cosine_floor, label_smooth=PROD.label_smooth,
                  batch=PROD.batch, ema_decay=PROD.ema_decay,
                  clip_warmup_steps=PROD.clip_warmup_steps,
                  clip_window=PROD.clip_window, clip_pctl=PROD.clip_pctl,
                  n_lab=PROD.n_lab, n_site=PROD.n_site, n_loso=PROD.n_loso,
                  scientific=False)

POLICIES = {"prod": PROD, "e2e": E2E, "test": TEST}

def get(name):
    if name not in POLICIES: raise ValueError(f"unknown policy {name!r}")
    return POLICIES[name]
