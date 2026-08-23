# S13 GATE 2 — NINE BLOCKING IMPLEMENTATION TESTS: **PASS**

Parameter count H=128, D=93, K=4: **293,953** (expected ~294,000; abort budget 1,030,000).

- [PASS] **param_budget** — 293,953 params (H=128,D=93,K=4); budget 1,030,000 (the EdgeMLP)
- [PASS] **1_orthonormal_buffer** — max|EE^T-I|=5.96e-08; in state_dict=True; buffer(not Parameter)=True; requires_grad=False
- [PASS] **2_soft_assignment** — max|sum_K P -1|=1.79e-07; Z_G(8, 4, 128); repr(8, 512)
- [PASS] **3_attention** — row-sum err 2.38e-07; shape (8, 4, 90, 90); layers 2; heads 4
- [PASS] **4a_profile_is_the_row** — X[s,i,:90] == FC[s,i,:] bitwise, 8 subjects x 8 ROIs
- [PASS] **4b_fc_symmetry_proof** — max|FC-FC^T| = 0.0e+00 EXACTLY 0 -> row and column profiles are the same vector, so a transposed profile is provably a no-op on this data
- [PASS] **4c_true_roi_equivariance** — permute data AND inp.weight columns -> max|Z_L(perm)-perm(Z_L)| = 1.67e-06 (<1e-4): the node axis IS the sequence axis, no [B,D,90] axis swap
- [PASS] **5_K1_equals_mean** — max|Z_G/90 - mean| = 8.34e-07; P==1 confirmed (declared factor 90)
- [PASS] **6_gradient_flow** — groups {'inp': 3.0326, 'mhsa': 6.0294, 'head': 4.925}; dead params 0; all 10 LayerNorms live; E.grad is None (True)
- [PASS] **7_overfit_one_batch** — train AUC 1.0000 (need 1.000), loss 0.00002 (need <0.01), 500 steps, dropout 0, wd 0, label smoothing OFF (smoothing floors BCE at 0.325)
- [PASS] **8_finite_determinism_reload** — finite=True; same-seed bitwise=True; save/reload bitwise=True (CPU)
- [PASS] **9_gate0_reverified_per_job** — bnt_core.load_all() asserts every Gate-0 hash and is called first in gates, in w_s13.run(), and in consolidation

Design assertions verified by inspection: FC edge weights are NEVER added to attention scores (grep `attn` in bnt_core.py — scores are Q@K^T/sqrt(d_k) only); `inp` is Linear(D,H) with H=128 >= D (asserted at construction, no compression); normalisation is LayerNorm only (no BatchNorm anywhere); there is no F.normalize and no global_add_pool.
- wall 23.1s
