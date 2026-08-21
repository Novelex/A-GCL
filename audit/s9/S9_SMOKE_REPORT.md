# S9 SMOKE REPORT — PASS
date: 2026-08-20T21:29:45+01:00 | git HEAD: d671467bce16260c682ff4ebdd466af5535c1b6d | tree clean
config: 8 subjects, 1 epoch, CPU, 2 threads, Corrected-C, seed 20260818

| check | result |
|---|---|
| epoch-0 extraction BEFORE any optimizer update | PASS (asserted n_updates==0 at extraction) |
| shapes pre_norm[8,90,32] post_norm[8,90,32] h[8,32] z[8,32] labels/sids[8] | PASS (asserted) |
| all tensors finite | PASS (asserted) |
| edge_weight non-None + finite during extraction | PASS (asserted per batch) |
| checkpoints save + reload, embeddings allclose | PASS |
| atomic TEMP->validate->rename->DONE | PASS (write_unit) |
| production tree unchanged | PASS (git status clean) |
| grads finite through 1 epoch | PASS |
| runtime | 0.3 s |
Note: pre_norm captured via a forward hook on encoder.bns[1]; in eval() the trailing
dropout is identity, so bn2 output == the final node representation before F.normalize.
No production file was touched (audit-only hook).
