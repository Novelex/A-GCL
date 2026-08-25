"""S16 Correction Gate 2 tests. A: A7 instrumentation + miniature execution.
B: regression proving the EVALUATED state is the DOCUMENTED state.
Local only. Produces no scientific results."""
import sys, os, json, copy, numpy as np, torch, torch.nn as nn
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_data as DAT, s16_models as MO, s16_train as TR, s16_feat as FT
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
F=[]
def ck(n,ok,d=""):
    print(("PASS " if ok else "FAIL ")+n+(" | "+d if d else ""),flush=True)
    if not ok: F.append(n)

torch.use_deterministic_algorithms(True); torch.set_num_threads(4)
d,MAN,ent = DAT.load("signed", where="gate2test")
FC,ALFF,y = d["FC"],d["ALFF"],d["y"].astype(np.int64)
tag,tr,te = DAT.folds(d,"lab")[0]
tr_enc,tr_prb = FT.honest_split(tr,y)

# ================= A. A7 INSTRUMENTATION =================
print("\n--- A. A7 instrumentation mapping ---")
X,FCu = FT.build_X("edgetri", FC, ALFF, tr_enc)
m = MO.build_model("EDGEMLP", X.shape[-1], DAT.BASE, 256)
# assert_groups_cover now returns TWO censuses (all params, trainable params):
# the frozen-parameter blind spot was defect D32.
owners_all, owners = TR.assert_groups_cover(m, "EDGEMLP")
ck("A1_every_key_exists", set(TR.GROUPS["EDGEMLP"])=={"inp","enc","head"},
   f"groups {sorted(TR.GROUPS['EDGEMLP'])}")
by={}
for k,g in owners.items(): by.setdefault(g,[]).append(k)
for g in ("inp","enc","head"):
    print(f"    {g:5s} <- {by[g]}")
ck("A2_partition_exact", len(owners)==sum(1 for _,p in m.named_parameters() if p.requires_grad),
   f"{len(owners)} trainable params, each in exactly ONE group (assert_groups_cover)")
n_tr_par = sum(p.numel() for p in m.parameters() if p.requires_grad)
per_g = {g: sum(dict(m.named_parameters())[k].numel() for k in ks) for g,ks in by.items()}
ck("A3_no_double_count", sum(per_g.values())==n_tr_par,
   f"{per_g} sums to {sum(per_g.values())} == {n_tr_par}")

# miniature batch: forward, loss, backward, clip, step, extract, serialize
ii = np.concatenate([np.where(y==0)[0][:8], np.where(y==1)[0][:8]])
b = MO.make_batch(X, FCu, ii, need_graph=False)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-3)
m.train(); opt.zero_grad()
r,lg = m(b,None)
loss = TR.loss_bce(lg, torch.tensor(y[ii],dtype=torch.float32))
loss.backward()
gn = TR.group_grad_norms(m,"EDGEMLP")
ck("A4_grad_norms_finite", all(np.isfinite(v) for v in gn.values()) and all(v>0 for v in gn.values()),
   f"{ {k:round(v,4) for k,v in gn.items()} }")
raw = float(torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1e9))
torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], raw*0.5)
opt.step()
init = {k:v.cpu().clone() for k,v in MO.build_model("EDGEMLP",X.shape[-1],DAT.BASE,256).state_dict().items()}
mv = TR.movement(init, m, "EDGEMLP")
ck("A5_movement_finite", all(np.isfinite(v) for v in mv.values()) and set(mv)=={"inp","enc","head"},
   f"{ {k:round(v,6) for k,v in mv.items()} }")
R_,S_ = TR.extract(m, X, FCu, np.arange(64), need_graph=False)
ck("A6_extract", R_.shape==(64,32) and np.isfinite(R_).all() and np.isfinite(S_).all(),
   f"repr {R_.shape} logits {S_.shape} finite")
pth="/tmp/_a7_gate2.pt"; torch.save(m.state_dict(),pth)
m2 = MO.build_model("EDGEMLP",X.shape[-1],DAT.BASE+9,256); m2.load_state_dict(torch.load(pth,weights_only=True))
R2,S2 = TR.extract(m2, X, FCu, np.arange(64), need_graph=False)
ck("A7_serialize_bitwise", np.array_equal(R_,R2) and np.array_equal(S_,S2), "save/reload bitwise")
print(f"    miniature: loss {float(loss):.6f} | raw grad-norm {raw:.4f} | clipped to {raw*0.5:.4f}")

# A7 signed input parity vs the historical float32 input
Xfc,_,_,_ = K.load_Xfc()
b32 = np.array_equal(X, Xfc.astype(np.float32))
d64 = float(np.abs(X.astype(np.float64)-Xfc).max())
ck("A8_parity_f32_bitwise", X.shape==(954,4005) and b32,
   f"A7 signed input == X_fc.astype(float32) BITWISE (what S12A5 arm C consumed, "
   f"w_wave1.py:34)")
ck("A9_f64_is_quantisation_only", d64 < 3e-8,
   f"float64 residual {d64:.3e} — QUANTISATION ONLY, not a data-parity claim; "
   f"f32 is the parity precision")

# plain and fused code paths (no scientific output: 2 epochs)
print("\n--- A. A7 plain / fused code paths (2 epochs, correctness only) ---")
cfg = dict(K_or_hidden=256, lr=3e-4, wd=1e-3, loss="L-BCE", freeze_encoder=False,
           readout="roi", dropout=0.10, H=128, max_epochs=2, min_epochs=1)
mdl, ema_sd, curve, info = TR.train_fold("EDGEMLP", X, FCu, y, tr_enc, cfg, DAT.BASE)
ck("A10_train_fold_runs", info["n_params"]>0 and np.isfinite(info["movement_max"]),
   f"params {info['n_params']:,} steps {info['total_steps']} "
   f"movement_max {info['movement_max']:.6f} clip_rate {info['clip_rate']:.3f}")
Rf,Sf = TR.extract(mdl, X, FCu, np.arange(954), need_graph=False)
s_fc,s_le = FT.scores_for_fusion(Rf, Xfc, y, tr_enc, tr_prb, te)
f1 = FT.fuse_scores(s_fc,s_le,1.0,tr_prb)
from sklearn.metrics import roc_auc_score
svm = float(roc_auc_score(y[te], s_fc[te]))
ck("A11_fused_path", abs(roc_auc_score(y[te],f1[te])-svm)<1e-12,
   f"alpha=1 AUC == svm_tr_enc {svm:.10f} exactly, on the A7 fused path")

# ================= B. EMA REGRESSION =================
print("\n--- B. evaluated state == documented state ---")
ema_model = MO.build_model("EDGEMLP", X.shape[-1], DAT.BASE, 256)
ema_model.load_state_dict(ema_sd)
same_as_raw = all(torch.equal(a.cpu(),b_.cpu()) for a,b_ in
                  zip(mdl.state_dict().values(), ema_model.state_dict().values()))
ck("B1_ema_state_differs_from_raw", not same_as_raw,
   "EMA state is genuinely distinct from the validation-best raw state")
_,S_ema = TR.extract(ema_model, X, FCu, np.arange(954), need_graph=False)
ck("B2_ema_evaluable", np.isfinite(S_ema).all() and not np.array_equal(S_ema,Sf),
   f"EMA logits finite and distinct; raw AUC {roc_auc_score(y[te],Sf[te]):.4f} "
   f"vs EMA {roc_auc_score(y[te],S_ema[te]):.4f}")
# the documented rule: raw is the VALIDATION-BEST checkpoint, selection unchanged
best_ep = info["best_epoch"]
ck("B3_raw_is_validation_best",
   abs(curve[best_ep-1]["val_auc"] - info["best_val_auc"]) < 1e-12,
   f"loaded raw state is epoch {best_ep}, the argmax of validation AUC "
   f"({info['best_val_auc']:.6f}) — selection by VALIDATION only, unchanged")
print(f"\n=== GATE 2 SUMMARY: {len(F)} failures ===")
for f in F: print("  FAIL "+f)
sys.exit(1 if F else 0)
