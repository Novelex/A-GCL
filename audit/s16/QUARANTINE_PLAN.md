# S16 QUARANTINE PLAN — prepared, NOT EXECUTED

Nothing has been moved or deleted. This is the inventory and the proposed destination.

## Why every one of these is invalidated

**The execution contract changed, not necessarily the input tensors.** A reused fold now
has to match a 26-field provenance contract (git SHA, clean worktree, worker version,
config hash, data/cache/split hashes, unit/arm/arch/E/mode, seed/fold/protocol, epoch and
stopping policy, optimizer/scheduler recipe, model-state rule, representation dimension,
status, and feature/checkpoint hashes). None of these artifacts carries a manifest at all,
so none can satisfy it. Independently, they were produced with `TR.MAX_EPOCHS=4` on fold
`lab0` only, under a worker that lacked `GROUPS['EDGEMLP']` and discarded EMA. This is not
a claim that every input tensor would differ numerically — it is that the contract they
were produced under no longer exists.

## Inventory — 122 artifacts

| path | type | unit | fold | sha256[:16] | mtime | bytes | probable source | proposed destination |
|---|---|---|---|---|---|---|---|---|
| `feat/abl_A1_signed_alff-raw_plain_s0__lab0.npz` | feature | abl_A1_signed_alff-raw_plain_s0 | lab0 | `770c197d455d843d` | 2026-08-24 23:11 | 40,750,615 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/abl_A1_signed_alff-raw_plain_s0__lab0.npz` |
| `feat/ctrl_A6_C-PERM_signed_plain_s0__lab0.npz` | feature | ctrl_A6_C-PERM_signed_plain_s0 | lab0 | `023e1ca94834580a` | 2026-08-24 23:03 | 14,484,652 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/ctrl_A6_C-PERM_signed_plain_s0__lab0.npz` |
| `feat/ctrl_A6_C-RAND_signed_plain_s0__lab0.npz` | feature | ctrl_A6_C-RAND_signed_plain_s0 | lab0 | `2798459296864199` | 2026-08-24 23:01 | 14,482,083 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/ctrl_A6_C-RAND_signed_plain_s0__lab0.npz` |
| `feat/ctrl_A6_C-ROI_signed_plain_s0__lab0.npz` | feature | ctrl_A6_C-ROI_signed_plain_s0 | lab0 | `e9e2ff6f35567539` | 2026-08-24 23:00 | 14,485,634 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/ctrl_A6_C-ROI_signed_plain_s0__lab0.npz` |
| `feat/ctrl_A6_C-SHUF_signed_plain_s0__lab0.npz` | feature | ctrl_A6_C-SHUF_signed_plain_s0 | lab0 | `0a7ed5107252b9f8` | 2026-08-24 23:02 | 14,483,846 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/ctrl_A6_C-SHUF_signed_plain_s0__lab0.npz` |
| `feat/main_A1_pos_zero_plain_s0__lab0.npz` | feature | main_A1_pos_zero_plain_s0 | lab0 | `2e70161b7994f121` | 2026-08-24 23:12 | 40,655,929 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/main_A1_pos_zero_plain_s0__lab0.npz` |
| `feat/main_A1_signed_fused_s0__lab0.npz` | feature | main_A1_signed_fused_s0 | lab0 | `df20ab6bb46fd8e6` | 2026-08-24 23:08 | 40,633,458 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/main_A1_signed_fused_s0__lab0.npz` |
| `feat/main_A1_signed_plain_s0__lab0.npz` | feature | main_A1_signed_plain_s0 | lab0 | `df20ab6bb46fd8e6` | 2026-08-24 23:09 | 40,633,458 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/main_A1_signed_plain_s0__lab0.npz` |
| `feat/main_A3_signed_fused_s0__lab0.npz` | feature | main_A3_signed_fused_s0 | lab0 | `a9a8da47016bcd8d` | 2026-08-24 23:04 | 40,713,286 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/main_A3_signed_fused_s0__lab0.npz` |
| `feat/main_A3_signed_plain_s0__lab0.npz` | feature | main_A3_signed_plain_s0 | lab0 | `a9a8da47016bcd8d` | 2026-08-24 23:03 | 40,713,286 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/main_A3_signed_plain_s0__lab0.npz` |
| `feat/main_A4_signed_fused_s0__lab0.npz` | feature | main_A4_signed_fused_s0 | lab0 | `6911cab83f7360b1` | 2026-08-24 23:05 | 40,683,052 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/main_A4_signed_fused_s0__lab0.npz` |
| `feat/main_A4_signed_plain_s0__lab0.npz` | feature | main_A4_signed_plain_s0 | lab0 | `6911cab83f7360b1` | 2026-08-24 23:05 | 40,683,052 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/main_A4_signed_plain_s0__lab0.npz` |
| `feat/main_A5_signed_fused_s0__lab0.npz` | feature | main_A5_signed_fused_s0 | lab0 | `b548f6dac9c7060f` | 2026-08-24 23:00 | 14,586,463 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/main_A5_signed_fused_s0__lab0.npz` |
| `feat/main_A5_signed_plain_s0__lab0.npz` | feature | main_A5_signed_plain_s0 | lab0 | `b548f6dac9c7060f` | 2026-08-24 23:01 | 14,586,463 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/main_A5_signed_plain_s0__lab0.npz` |
| `feat/main_A6_signed_fused_s0__lab0.npz` | feature | main_A6_signed_fused_s0 | lab0 | `45db0b36cfa9caf3` | 2026-08-24 23:00 | 14,485,701 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/main_A6_signed_fused_s0__lab0.npz` |
| `feat/main_A6_signed_plain_s0__lab0.npz` | feature | main_A6_signed_plain_s0 | lab0 | `45db0b36cfa9caf3` | 2026-08-24 23:01 | 14,485,701 | 4-epoch E2E (TR.MAX_EPOCHS=4, lab0 only) | `QUARANTINE/2026-08-25_pre-gate3/feat/main_A6_signed_plain_s0__lab0.npz` |
| `ckpt/abl_A1_signed_alff-raw_plain_s0__lab0.pt` | checkpoint | abl_A1_signed_alff-raw_plain_s0 | lab0 | `87993df77917991e` | 2026-08-24 23:11 | 414,194 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/abl_A1_signed_alff-raw_plain_s0__lab0.pt` |
| `ckpt/ctrl_A6_C-PERM_signed_plain_s0__lab0.pt` | checkpoint | ctrl_A6_C-PERM_signed_plain_s0 | lab0 | `ae377fb97e7f8bd0` | 2026-08-24 23:03 | 1,184,589 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/ctrl_A6_C-PERM_signed_plain_s0__lab0.pt` |
| `ckpt/ctrl_A6_C-RAND_signed_plain_s0__lab0.pt` | checkpoint | ctrl_A6_C-RAND_signed_plain_s0 | lab0 | `fc04f68324e1bafa` | 2026-08-24 23:01 | 1,184,589 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/ctrl_A6_C-RAND_signed_plain_s0__lab0.pt` |
| `ckpt/ctrl_A6_C-ROI_signed_plain_s0__lab0.pt` | checkpoint | ctrl_A6_C-ROI_signed_plain_s0 | lab0 | `c6837a9bfc8ddcb1` | 2026-08-24 23:00 | 1,184,550 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/ctrl_A6_C-ROI_signed_plain_s0__lab0.pt` |
| `ckpt/ctrl_A6_C-SHUF_signed_plain_s0__lab0.pt` | checkpoint | ctrl_A6_C-SHUF_signed_plain_s0 | lab0 | `40f61d076658b87c` | 2026-08-24 23:02 | 1,184,589 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/ctrl_A6_C-SHUF_signed_plain_s0__lab0.pt` |
| `ckpt/main_A1_pos_zero_plain_s0__lab0.pt` | checkpoint | main_A1_pos_zero_plain_s0 | lab0 | `eed1a2e041304197` | 2026-08-24 23:12 | 414,050 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/main_A1_pos_zero_plain_s0__lab0.pt` |
| `ckpt/main_A1_signed_fused_s0__lab0.pt` | checkpoint | main_A1_signed_fused_s0 | lab0 | `8717c0ed4ad9acdf` | 2026-08-24 23:08 | 414,002 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/main_A1_signed_fused_s0__lab0.pt` |
| `ckpt/main_A1_signed_plain_s0__lab0.pt` | checkpoint | main_A1_signed_plain_s0 | lab0 | `149c808befe10467` | 2026-08-24 23:09 | 414,002 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/main_A1_signed_plain_s0__lab0.pt` |
| `ckpt/main_A3_signed_fused_s0__lab0.pt` | checkpoint | main_A3_signed_fused_s0 | lab0 | `43d25991ae6386b2` | 2026-08-24 23:04 | 458,546 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/main_A3_signed_fused_s0__lab0.pt` |
| `ckpt/main_A3_signed_plain_s0__lab0.pt` | checkpoint | main_A3_signed_plain_s0 | lab0 | `31df44eac4b6ad53` | 2026-08-24 23:03 | 458,546 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/main_A3_signed_plain_s0__lab0.pt` |
| `ckpt/main_A4_signed_fused_s0__lab0.pt` | checkpoint | main_A4_signed_fused_s0 | lab0 | `030bc9af4ac88be8` | 2026-08-24 23:05 | 460,082 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/main_A4_signed_fused_s0__lab0.pt` |
| `ckpt/main_A4_signed_plain_s0__lab0.pt` | checkpoint | main_A4_signed_plain_s0 | lab0 | `8ce3850cf48186d3` | 2026-08-24 23:05 | 460,082 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/main_A4_signed_plain_s0__lab0.pt` |
| `ckpt/main_A5_signed_fused_s0__lab0.pt` | checkpoint | main_A5_signed_fused_s0 | lab0 | `6e85b63cccb062e6` | 2026-08-24 23:00 | 1,182,780 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/main_A5_signed_fused_s0__lab0.pt` |
| `ckpt/main_A5_signed_plain_s0__lab0.pt` | checkpoint | main_A5_signed_plain_s0 | lab0 | `914e71dcf91c32b3` | 2026-08-24 23:01 | 1,182,780 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/main_A5_signed_plain_s0__lab0.pt` |
| `ckpt/main_A6_signed_fused_s0__lab0.pt` | checkpoint | main_A6_signed_fused_s0 | lab0 | `840b2e4edbe77b8b` | 2026-08-24 23:00 | 1,184,316 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/main_A6_signed_fused_s0__lab0.pt` |
| `ckpt/main_A6_signed_plain_s0__lab0.pt` | checkpoint | main_A6_signed_plain_s0 | lab0 | `f3a7f610cf3a5cdb` | 2026-08-24 23:01 | 1,184,316 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/ckpt/main_A6_signed_plain_s0__lab0.pt` |
| `jobs/abl_A1_signed_alff-raw_plain_s0/HEARTBEAT` | status | abl_A1_signed_alff-raw_plain_s0 | HEARTBEAT | `7ebec39f08734d3e` | 2026-08-24 23:11 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/abl_A1_signed_alff-raw_plain_s0/HEARTBEAT` |
| `jobs/abl_A1_signed_alff-raw_plain_s0/STATUS.json` | status | abl_A1_signed_alff-raw_plain_s0 | STATUS | `96a7259e2874a55d` | 2026-08-24 23:11 | 206 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/abl_A1_signed_alff-raw_plain_s0/STATUS.json` |
| `jobs/abl_A1_signed_alff-raw_plain_s0/TALLY.json` | tally | abl_A1_signed_alff-raw_plain_s0 | TALLY | `efc1e64e035013fd` | 2026-08-24 23:11 | 94 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/abl_A1_signed_alff-raw_plain_s0/TALLY.json` |
| `jobs/abl_A1_signed_alff-raw_plain_s0/UNIT.done` | status | abl_A1_signed_alff-raw_plain_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:11 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/abl_A1_signed_alff-raw_plain_s0/UNIT.done` |
| `jobs/abl_A1_signed_alff-raw_plain_s0/fold_lab0.json` | result | abl_A1_signed_alff-raw_plain_s0 | lab0 | `9104520685189600` | 2026-08-24 23:11 | 4,882 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/abl_A1_signed_alff-raw_plain_s0/fold_lab0.json` |
| `jobs/ctrl_A6_C-PERM_signed_plain_s0/HEARTBEAT` | status | ctrl_A6_C-PERM_signed_plain_s0 | HEARTBEAT | `7ac98b3d389a9851` | 2026-08-24 23:03 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-PERM_signed_plain_s0/HEARTBEAT` |
| `jobs/ctrl_A6_C-PERM_signed_plain_s0/STATUS.json` | status | ctrl_A6_C-PERM_signed_plain_s0 | STATUS | `bcfa483a10f5f25a` | 2026-08-24 23:03 | 205 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-PERM_signed_plain_s0/STATUS.json` |
| `jobs/ctrl_A6_C-PERM_signed_plain_s0/TALLY.json` | tally | ctrl_A6_C-PERM_signed_plain_s0 | TALLY | `c3c0bfd5b11c7b2e` | 2026-08-24 23:03 | 93 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-PERM_signed_plain_s0/TALLY.json` |
| `jobs/ctrl_A6_C-PERM_signed_plain_s0/UNIT.done` | status | ctrl_A6_C-PERM_signed_plain_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:03 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-PERM_signed_plain_s0/UNIT.done` |
| `jobs/ctrl_A6_C-PERM_signed_plain_s0/fold_lab0.json` | result | ctrl_A6_C-PERM_signed_plain_s0 | lab0 | `7a67fc08b149f011` | 2026-08-24 23:03 | 4,431 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-PERM_signed_plain_s0/fold_lab0.json` |
| `jobs/ctrl_A6_C-RAND_signed_plain_s0/HEARTBEAT` | status | ctrl_A6_C-RAND_signed_plain_s0 | HEARTBEAT | `caf526d9bab47062` | 2026-08-24 23:00 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-RAND_signed_plain_s0/HEARTBEAT` |
| `jobs/ctrl_A6_C-RAND_signed_plain_s0/STATUS.json` | status | ctrl_A6_C-RAND_signed_plain_s0 | STATUS | `49643d6cb4f35db5` | 2026-08-24 23:01 | 205 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-RAND_signed_plain_s0/STATUS.json` |
| `jobs/ctrl_A6_C-RAND_signed_plain_s0/TALLY.json` | tally | ctrl_A6_C-RAND_signed_plain_s0 | TALLY | `5391f19d1a7a27ce` | 2026-08-24 23:01 | 93 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-RAND_signed_plain_s0/TALLY.json` |
| `jobs/ctrl_A6_C-RAND_signed_plain_s0/UNIT.done` | status | ctrl_A6_C-RAND_signed_plain_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:01 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-RAND_signed_plain_s0/UNIT.done` |
| `jobs/ctrl_A6_C-RAND_signed_plain_s0/fold_lab0.json` | result | ctrl_A6_C-RAND_signed_plain_s0 | lab0 | `f42194e12e6baada` | 2026-08-24 23:01 | 4,794 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-RAND_signed_plain_s0/fold_lab0.json` |
| `jobs/ctrl_A6_C-ROI_signed_plain_s0/HEARTBEAT` | status | ctrl_A6_C-ROI_signed_plain_s0 | HEARTBEAT | `caf526d9bab47062` | 2026-08-24 23:00 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-ROI_signed_plain_s0/HEARTBEAT` |
| `jobs/ctrl_A6_C-ROI_signed_plain_s0/STATUS.json` | status | ctrl_A6_C-ROI_signed_plain_s0 | STATUS | `8770e07e900581cd` | 2026-08-24 23:00 | 205 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-ROI_signed_plain_s0/STATUS.json` |
| `jobs/ctrl_A6_C-ROI_signed_plain_s0/TALLY.json` | tally | ctrl_A6_C-ROI_signed_plain_s0 | TALLY | `9da7b0ec9a00599e` | 2026-08-24 23:00 | 92 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-ROI_signed_plain_s0/TALLY.json` |
| `jobs/ctrl_A6_C-ROI_signed_plain_s0/UNIT.done` | status | ctrl_A6_C-ROI_signed_plain_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:00 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-ROI_signed_plain_s0/UNIT.done` |
| `jobs/ctrl_A6_C-ROI_signed_plain_s0/fold_lab0.json` | result | ctrl_A6_C-ROI_signed_plain_s0 | lab0 | `20d15c20f685e3d3` | 2026-08-24 23:00 | 4,925 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-ROI_signed_plain_s0/fold_lab0.json` |
| `jobs/ctrl_A6_C-SHUF_signed_plain_s0/HEARTBEAT` | status | ctrl_A6_C-SHUF_signed_plain_s0 | HEARTBEAT | `229f34a75a3432c9` | 2026-08-24 23:02 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-SHUF_signed_plain_s0/HEARTBEAT` |
| `jobs/ctrl_A6_C-SHUF_signed_plain_s0/STATUS.json` | status | ctrl_A6_C-SHUF_signed_plain_s0 | STATUS | `99742522f5e47277` | 2026-08-24 23:02 | 205 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-SHUF_signed_plain_s0/STATUS.json` |
| `jobs/ctrl_A6_C-SHUF_signed_plain_s0/TALLY.json` | tally | ctrl_A6_C-SHUF_signed_plain_s0 | TALLY | `f88495c887aa9867` | 2026-08-24 23:02 | 93 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-SHUF_signed_plain_s0/TALLY.json` |
| `jobs/ctrl_A6_C-SHUF_signed_plain_s0/UNIT.done` | status | ctrl_A6_C-SHUF_signed_plain_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:02 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-SHUF_signed_plain_s0/UNIT.done` |
| `jobs/ctrl_A6_C-SHUF_signed_plain_s0/fold_lab0.json` | result | ctrl_A6_C-SHUF_signed_plain_s0 | lab0 | `219be6c68f97ef78` | 2026-08-24 23:02 | 4,915 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/ctrl_A6_C-SHUF_signed_plain_s0/fold_lab0.json` |
| `jobs/main_A1_pos_zero_plain_s0/HEARTBEAT` | status | main_A1_pos_zero_plain_s0 | HEARTBEAT | `64d7c0a507bac4b2` | 2026-08-24 23:11 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_pos_zero_plain_s0/HEARTBEAT` |
| `jobs/main_A1_pos_zero_plain_s0/STATUS.json` | status | main_A1_pos_zero_plain_s0 | STATUS | `5cbed1a50c8354bb` | 2026-08-24 23:12 | 206 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_pos_zero_plain_s0/STATUS.json` |
| `jobs/main_A1_pos_zero_plain_s0/TALLY.json` | tally | main_A1_pos_zero_plain_s0 | TALLY | `ef1748bfe874997a` | 2026-08-24 23:12 | 88 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_pos_zero_plain_s0/TALLY.json` |
| `jobs/main_A1_pos_zero_plain_s0/UNIT.done` | status | main_A1_pos_zero_plain_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:12 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_pos_zero_plain_s0/UNIT.done` |
| `jobs/main_A1_pos_zero_plain_s0/fold_lab0.json` | result | main_A1_pos_zero_plain_s0 | lab0 | `10039c1eb4a01a9e` | 2026-08-24 23:12 | 4,908 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_pos_zero_plain_s0/fold_lab0.json` |
| `jobs/main_A1_signed_fused_s0/HEARTBEAT` | status | main_A1_signed_fused_s0 | HEARTBEAT | `ff804c6b0660f914` | 2026-08-24 23:08 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_signed_fused_s0/HEARTBEAT` |
| `jobs/main_A1_signed_fused_s0/STATUS.json` | status | main_A1_signed_fused_s0 | STATUS | `941f1424ed4ca42f` | 2026-08-24 23:08 | 205 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_signed_fused_s0/STATUS.json` |
| `jobs/main_A1_signed_fused_s0/TALLY.json` | tally | main_A1_signed_fused_s0 | TALLY | `785f8da8d3989808` | 2026-08-24 23:08 | 86 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_signed_fused_s0/TALLY.json` |
| `jobs/main_A1_signed_fused_s0/UNIT.done` | status | main_A1_signed_fused_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:08 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_signed_fused_s0/UNIT.done` |
| `jobs/main_A1_signed_fused_s0/fold_lab0.json` | result | main_A1_signed_fused_s0 | lab0 | `a0d1a4b62d517d35` | 2026-08-24 23:08 | 7,990 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_signed_fused_s0/fold_lab0.json` |
| `jobs/main_A1_signed_plain_s0/HEARTBEAT` | status | main_A1_signed_plain_s0 | HEARTBEAT | `ff804c6b0660f914` | 2026-08-24 23:08 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_signed_plain_s0/HEARTBEAT` |
| `jobs/main_A1_signed_plain_s0/STATUS.json` | status | main_A1_signed_plain_s0 | STATUS | `589afede18186b8e` | 2026-08-24 23:09 | 205 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_signed_plain_s0/STATUS.json` |
| `jobs/main_A1_signed_plain_s0/TALLY.json` | tally | main_A1_signed_plain_s0 | TALLY | `36f37c06a26614e8` | 2026-08-24 23:09 | 86 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_signed_plain_s0/TALLY.json` |
| `jobs/main_A1_signed_plain_s0/UNIT.done` | status | main_A1_signed_plain_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:09 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_signed_plain_s0/UNIT.done` |
| `jobs/main_A1_signed_plain_s0/fold_lab0.json` | result | main_A1_signed_plain_s0 | lab0 | `467bd7b37d713c94` | 2026-08-24 23:09 | 4,887 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A1_signed_plain_s0/fold_lab0.json` |
| `jobs/main_A3_signed_fused_s0/HEARTBEAT` | status | main_A3_signed_fused_s0 | HEARTBEAT | `56e6a372950addff` | 2026-08-24 23:04 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A3_signed_fused_s0/HEARTBEAT` |
| `jobs/main_A3_signed_fused_s0/STATUS.json` | status | main_A3_signed_fused_s0 | STATUS | `298180e47cdb8e65` | 2026-08-24 23:04 | 205 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A3_signed_fused_s0/STATUS.json` |
| `jobs/main_A3_signed_fused_s0/TALLY.json` | tally | main_A3_signed_fused_s0 | TALLY | `ad3b1b7452790dc4` | 2026-08-24 23:04 | 86 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A3_signed_fused_s0/TALLY.json` |
| `jobs/main_A3_signed_fused_s0/UNIT.done` | status | main_A3_signed_fused_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:04 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A3_signed_fused_s0/UNIT.done` |
| `jobs/main_A3_signed_fused_s0/fold_lab0.json` | result | main_A3_signed_fused_s0 | lab0 | `45e903b906a64b8d` | 2026-08-24 23:04 | 7,964 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A3_signed_fused_s0/fold_lab0.json` |
| `jobs/main_A3_signed_plain_s0/HEARTBEAT` | status | main_A3_signed_plain_s0 | HEARTBEAT | `7ac98b3d389a9851` | 2026-08-24 23:03 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A3_signed_plain_s0/HEARTBEAT` |
| `jobs/main_A3_signed_plain_s0/STATUS.json` | status | main_A3_signed_plain_s0 | STATUS | `7ccd1c507e378079` | 2026-08-24 23:03 | 205 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A3_signed_plain_s0/STATUS.json` |
| `jobs/main_A3_signed_plain_s0/TALLY.json` | tally | main_A3_signed_plain_s0 | TALLY | `5a193965033c33e6` | 2026-08-24 23:03 | 86 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A3_signed_plain_s0/TALLY.json` |
| `jobs/main_A3_signed_plain_s0/UNIT.done` | status | main_A3_signed_plain_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:03 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A3_signed_plain_s0/UNIT.done` |
| `jobs/main_A3_signed_plain_s0/fold_lab0.json` | result | main_A3_signed_plain_s0 | lab0 | `99de8ca69f51f585` | 2026-08-24 23:03 | 4,880 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A3_signed_plain_s0/fold_lab0.json` |
| `jobs/main_A4_signed_fused_s0/HEARTBEAT` | status | main_A4_signed_fused_s0 | HEARTBEAT | `56e6a372950addff` | 2026-08-24 23:04 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A4_signed_fused_s0/HEARTBEAT` |
| `jobs/main_A4_signed_fused_s0/STATUS.json` | status | main_A4_signed_fused_s0 | STATUS | `8a3273924ad25d6f` | 2026-08-24 23:05 | 205 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A4_signed_fused_s0/STATUS.json` |
| `jobs/main_A4_signed_fused_s0/TALLY.json` | tally | main_A4_signed_fused_s0 | TALLY | `32e785b35c9a552f` | 2026-08-24 23:05 | 86 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A4_signed_fused_s0/TALLY.json` |
| `jobs/main_A4_signed_fused_s0/UNIT.done` | status | main_A4_signed_fused_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:05 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A4_signed_fused_s0/UNIT.done` |
| `jobs/main_A4_signed_fused_s0/fold_lab0.json` | result | main_A4_signed_fused_s0 | lab0 | `6ca5f0c5bf36c259` | 2026-08-24 23:05 | 7,958 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A4_signed_fused_s0/fold_lab0.json` |
| `jobs/main_A4_signed_plain_s0/HEARTBEAT` | status | main_A4_signed_plain_s0 | HEARTBEAT | `d0df1138e7925115` | 2026-08-24 23:05 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A4_signed_plain_s0/HEARTBEAT` |
| `jobs/main_A4_signed_plain_s0/STATUS.json` | status | main_A4_signed_plain_s0 | STATUS | `5acbc9f45e6052b4` | 2026-08-24 23:05 | 205 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A4_signed_plain_s0/STATUS.json` |
| `jobs/main_A4_signed_plain_s0/TALLY.json` | tally | main_A4_signed_plain_s0 | TALLY | `2678be68b5788eab` | 2026-08-24 23:05 | 86 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A4_signed_plain_s0/TALLY.json` |
| `jobs/main_A4_signed_plain_s0/UNIT.done` | status | main_A4_signed_plain_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:05 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A4_signed_plain_s0/UNIT.done` |
| `jobs/main_A4_signed_plain_s0/fold_lab0.json` | result | main_A4_signed_plain_s0 | lab0 | `8ad0b1a7aac8f585` | 2026-08-24 23:05 | 4,872 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A4_signed_plain_s0/fold_lab0.json` |
| `jobs/main_A5_signed_fused_s0/HEARTBEAT` | status | main_A5_signed_fused_s0 | HEARTBEAT | `caf526d9bab47062` | 2026-08-24 23:00 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A5_signed_fused_s0/HEARTBEAT` |
| `jobs/main_A5_signed_fused_s0/STATUS.json` | status | main_A5_signed_fused_s0 | STATUS | `79f7bc6c4a869cdc` | 2026-08-24 23:00 | 205 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A5_signed_fused_s0/STATUS.json` |
| `jobs/main_A5_signed_fused_s0/TALLY.json` | tally | main_A5_signed_fused_s0 | TALLY | `e0d16075e6871969` | 2026-08-24 23:00 | 86 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A5_signed_fused_s0/TALLY.json` |
| `jobs/main_A5_signed_fused_s0/UNIT.done` | status | main_A5_signed_fused_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:00 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A5_signed_fused_s0/UNIT.done` |
| `jobs/main_A5_signed_fused_s0/fold_lab0.json` | result | main_A5_signed_fused_s0 | lab0 | `3b44357551a1d675` | 2026-08-24 23:00 | 7,965 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A5_signed_fused_s0/fold_lab0.json` |
| `jobs/main_A5_signed_plain_s0/HEARTBEAT` | status | main_A5_signed_plain_s0 | HEARTBEAT | `400d15450651c967` | 2026-08-24 23:01 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A5_signed_plain_s0/HEARTBEAT` |
| `jobs/main_A5_signed_plain_s0/STATUS.json` | status | main_A5_signed_plain_s0 | STATUS | `2ec9d4d3c6673830` | 2026-08-24 23:01 | 205 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A5_signed_plain_s0/STATUS.json` |
| `jobs/main_A5_signed_plain_s0/TALLY.json` | tally | main_A5_signed_plain_s0 | TALLY | `50b7b58a00714bf5` | 2026-08-24 23:01 | 86 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A5_signed_plain_s0/TALLY.json` |
| `jobs/main_A5_signed_plain_s0/UNIT.done` | status | main_A5_signed_plain_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:01 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A5_signed_plain_s0/UNIT.done` |
| `jobs/main_A5_signed_plain_s0/fold_lab0.json` | result | main_A5_signed_plain_s0 | lab0 | `f62f96124751242c` | 2026-08-24 23:01 | 4,879 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A5_signed_plain_s0/fold_lab0.json` |
| `jobs/main_A6_signed_fused_s0/HEARTBEAT` | status | main_A6_signed_fused_s0 | HEARTBEAT | `caf526d9bab47062` | 2026-08-24 23:00 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A6_signed_fused_s0/HEARTBEAT` |
| `jobs/main_A6_signed_fused_s0/STATUS.json` | status | main_A6_signed_fused_s0 | STATUS | `dfba7a588c9b52e0` | 2026-08-24 23:00 | 205 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A6_signed_fused_s0/STATUS.json` |
| `jobs/main_A6_signed_fused_s0/TALLY.json` | tally | main_A6_signed_fused_s0 | TALLY | `f1ac2a1dd6d4cd12` | 2026-08-24 23:00 | 86 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A6_signed_fused_s0/TALLY.json` |
| `jobs/main_A6_signed_fused_s0/UNIT.done` | status | main_A6_signed_fused_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:00 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A6_signed_fused_s0/UNIT.done` |
| `jobs/main_A6_signed_fused_s0/fold_lab0.json` | result | main_A6_signed_fused_s0 | lab0 | `4bd139192b83a211` | 2026-08-24 23:00 | 7,967 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A6_signed_fused_s0/fold_lab0.json` |
| `jobs/main_A6_signed_plain_s0/HEARTBEAT` | status | main_A6_signed_plain_s0 | HEARTBEAT | `caf526d9bab47062` | 2026-08-24 23:00 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A6_signed_plain_s0/HEARTBEAT` |
| `jobs/main_A6_signed_plain_s0/STATUS.json` | status | main_A6_signed_plain_s0 | STATUS | `eb104218782f91ea` | 2026-08-24 23:01 | 205 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A6_signed_plain_s0/STATUS.json` |
| `jobs/main_A6_signed_plain_s0/TALLY.json` | tally | main_A6_signed_plain_s0 | TALLY | `af12f47423690913` | 2026-08-24 23:01 | 86 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A6_signed_plain_s0/TALLY.json` |
| `jobs/main_A6_signed_plain_s0/UNIT.done` | status | main_A6_signed_plain_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:01 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A6_signed_plain_s0/UNIT.done` |
| `jobs/main_A6_signed_plain_s0/fold_lab0.json` | result | main_A6_signed_plain_s0 | lab0 | `eb1b952419fbdb5c` | 2026-08-24 23:01 | 4,883 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A6_signed_plain_s0/fold_lab0.json` |
| `jobs/main_A7_signed_fused_s0/HEARTBEAT` | status | main_A7_signed_fused_s0 | HEARTBEAT | `d717163b5eb0804a` | 2026-08-24 23:15 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A7_signed_fused_s0/HEARTBEAT` |
| `jobs/main_A7_signed_fused_s0/STATUS.json` | status | main_A7_signed_fused_s0 | STATUS | `9ebde2b2ed01cc68` | 2026-08-24 23:15 | 203 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A7_signed_fused_s0/STATUS.json` |
| `jobs/main_A7_signed_fused_s0/TALLY.json` | tally | main_A7_signed_fused_s0 | TALLY | `beb5a74d9f241ae3` | 2026-08-24 23:15 | 86 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A7_signed_fused_s0/TALLY.json` |
| `jobs/main_A7_signed_fused_s0/UNIT.done` | status | main_A7_signed_fused_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:15 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A7_signed_fused_s0/UNIT.done` |
| `jobs/main_A7_signed_fused_s0/fold_lab0.json` | result | main_A7_signed_fused_s0 | lab0 | `837c3d48a77dc555` | 2026-08-24 23:15 | 1,019 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A7_signed_fused_s0/fold_lab0.json` |
| `jobs/main_A7_signed_plain_s0/HEARTBEAT` | status | main_A7_signed_plain_s0 | HEARTBEAT | `d717163b5eb0804a` | 2026-08-24 23:15 | 19 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A7_signed_plain_s0/HEARTBEAT` |
| `jobs/main_A7_signed_plain_s0/STATUS.json` | status | main_A7_signed_plain_s0 | STATUS | `9ebde2b2ed01cc68` | 2026-08-24 23:15 | 203 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A7_signed_plain_s0/STATUS.json` |
| `jobs/main_A7_signed_plain_s0/TALLY.json` | tally | main_A7_signed_plain_s0 | TALLY | `6a3e867365d5d6e2` | 2026-08-24 23:15 | 86 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A7_signed_plain_s0/TALLY.json` |
| `jobs/main_A7_signed_plain_s0/UNIT.done` | status | main_A7_signed_plain_s0 | UNIT.done | `a4c3ed04a95a3da1` | 2026-08-24 23:15 | 4 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A7_signed_plain_s0/UNIT.done` |
| `jobs/main_A7_signed_plain_s0/fold_lab0.json` | result | main_A7_signed_plain_s0 | lab0 | `4b736d3f59a2c896` | 2026-08-24 23:15 | 1,019 | 4-epoch E2E | `QUARANTINE/2026-08-25_pre-gate3/jobs/main_A7_signed_plain_s0/fold_lab0.json` |

**Total 122 artifacts, 454.6 MB.**

## Proposed action (awaiting authorisation)
`git mv` / `mv` each path to `audit/s16/QUARANTINE/2026-08-25_pre-gate3/` preserving the relative layout. NOTHING is
deleted. The new namespaced trees `audit/s16/runs/prod/` and `audit/s16/runs/e2e/` are
already live and empty of production output, so quarantine is a tidiness step, not a
correctness one: the legacy paths are invisible to both namespaces (verified by test
A3_legacy_outside_namespaces).

## Not in scope for quarantine
`cache/*.npz` (the four E-treatment caches) and `CACHE_MANIFEST.json` are INPUTS, not
results, and are hash-verified by Gate-C on every job. They stay.
`out/C2_*.json`, `C2_PROBE.md`, `C2_PRECISION.md` are C2 outputs computed from frozen
S12A5/S13/S15 checkpoints, untouched by the C6 contract change. They stay.
