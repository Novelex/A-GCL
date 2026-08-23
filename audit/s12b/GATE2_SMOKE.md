# S12B GATE 2 — FORWARD CORRECTNESS: **PASS** (all checks asserted)

- [PASS] toy_wgin mrelu=True h.float64 max_err=0.00e+00
- [PASS] toy_wgin mrelu=True h.float32 max_err=0.00e+00
- [PASS] toy_wgin mrelu=False h.float64 max_err=0.00e+00
- [PASS] toy_wgin mrelu=False h.float32 max_err=0.00e+00
- [PASS] selfloop_fork 90 self-edges, FC diag=1 -> own features counted TWICE (kept, logged, all arms)
- [PASS] armC_fcrow_bitwise 8x8 spot checks exact
- [PASS] A1_identity max_err=1.34e-05 (f32 90-term sums)
- [PASS] finiteness_all_arms 6 arms x 6 stages x 8 subjects
- [PASS] gpu_available_for_gate2 Gate 2 must run on a GPU node so the GPU determinism leg executes
- [PASS] determinism_cpu 2 runs bitwise equal (incl. BN calibration)
- [PASS] determinism_cuda 2 runs bitwise equal (incl. BN calibration)
- [PASS] determinism_cpu_vs_gpu_tol max|CPU-GPU|=2.29e-05 (<1e-4; bitwise cross-device equality not claimed)
- [PASS] save_reload_bitwise 
- [PASS] production_parity node_err=8.94e-08 pool_err=3.81e-06

Self-loop accounting (fork, kept as production): edge set includes (i,i) with FC[i,i]=1 and WGINConv adds (1+eps)*x_r with eps=0 -> own node features enter the aggregation with total coefficient 2. A1 := X + FC^T@X (diag included) matches conv.propagate to f32 tolerance.
- device: cuda; cudnn.deterministic=True
- wall 1.4s
