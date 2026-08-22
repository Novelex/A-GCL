# S12A3 RESULTS — Encoder Retention Audit (random encoders, identity input, ROI-aware readout)
Sentinels (hard-gated at consolidation): plumbing == frozen S11 float exactly (0.756484);
arm A embeddings BITWISE == S12A1 all 3 seeds; arm A AUC == S12A2 arm X per seed.
Metric: pooled OOF AUC, flatten [954, 90*emb] -> S11 harness, frozen S3C folds. No pooling.

| arm | config                    | s0     | s1     | s2     | mean   | LOSO mean |
|-----|---------------------------|--------|--------|--------|--------|-----------|
| A   | baseline norm=T pbr=T e32 | 0.6046 | 0.6224 | 0.6381 | 0.6217 | 0.6111 |
| B   | normalize_nodes=F         | 0.6509 | 0.6565 | 0.6542 | 0.6539 | 0.6305 |
| C   | post_bn_relu=F            | 0.6058 | 0.6271 | 0.6380 | 0.6236 | 0.6132 |
| D   | norm=F + pbr=F            | 0.6492 | 0.6622 | 0.6533 | **0.6549** | 0.6320 |
| E   | emb_dim=64                | 0.6523 | 0.6171 | 0.6591 | 0.6429 | 0.6394 |
| F   | emb_dim=128               | 0.6520 | 0.6497 | 0.6297 | 0.6438 | **0.6454** |

Attribution (vs baseline 0.6217): L2 node-normalize costs ~0.032 (B); post-BN ReLU ~0 alone
(C) and adds ~0.001 on top of norm=F (D~=B); width 64/128 recovers ~0.021-0.022 (E,F) and is
the best config under LOSO (F 0.6454). Gains do NOT stack beyond ~0.655 in tested arms.
All 18 embeddings finite, bitwise-deterministic (rebuild + batch-size invariance asserted);
per-config npz saved (nodes + flat + hashes) under s12a3/emb/, shas in extract_*.json.
Runtime: 18 extractions ~20 s; 37 probes on two opposing SLURM arrays (15 concurrent,
150 CPUs); adversarial 3-lens review of scripts pre-submission: 0 blockers.
