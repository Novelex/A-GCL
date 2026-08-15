# Layer 2 — Graph Construction: Data Objects, Edges, Labels, Subject IDs

Part of the full pipeline audit (see `/users/3171356m/.claude/plans/eager-drifting-teapot.md`).
This layer checks the actual `Data` objects `datasets/abideDataset.py::ABIDEDataset` builds
from Layer 1's raw sources — the objects the model actually trains on. Produced by
`layertesting/layer2/test_layer2_graph_construction.py`.

## 1. Dataset load — clean

`ABIDEDataset('data', 'ABIDE', ...)` loads the cached `data_dense_v2.pt` without error.
**N = 956 subjects**, matching every prior count this session. No crash.

## 2. Edge structure — correct, matches documented construction

| Check | Result |
|---|---|
| `edge_index` shape (per graph) | `(2, 8100)` = 90×90, exact dense M² |
| `edge_weight` shape | `(8100,)` |
| `edge_weight` range | `[-0.2032, 1.0000]` — within `[-1, 1]` as expected post max-abs normalization |
| `edge_weight` finite | ✅ True |
| Self-loop count | 90 (= num_nodes), all weight `== 1.0` exactly |

Matches the documented dense M² construction (`datasets/abideDataset.py`) and Layer 1's finding
that the diagonal (self-correlation) is exactly `1.0`, so max-abs normalization is a verified
no-op for the diagonal.

## 3. `subject_id` — correct

Range `[0, 955]`, **all 956 unique** (no collisions), and the first 455 IDs (`0`–`454`)
correspond exactly to the ASD (`label==1`) subjects — confirming the documented ASD-then-NC,
sorted-filename assignment order.

## 4. Labels (`y`) — correct

`dtype=int64`, values `{0, 1}` only. Counts: **ASD(1)=455, NC(0)=501** — matches Layer 1 and
every prior count.

## 5. Node features (`x`) — correct, and distinct from Layer 1's raw characterization

After `abideDataset.py`'s own per-subject global min-max step (applied on top of the already
z-scored old ALFF from Layer 1): shape `(956, 90, 3)`, all finite, global range exactly
`[0.0, 1.0]`. This is the **third** normalization layer old ALFF has been through by this point
(DPARSFA z-score → this pipeline's min-max) — worth remembering when interpreting anything
downstream, since old ALFF's original magnitude information was lost two steps ago and can't
be recovered.

## 6. The real finding: no code path for new ALFF exists

**`ABIDEDataset._load_class()` unconditionally loads `nf['norm_matrix']`** — old ALFF — with no
parameter, flag, or branch to select new ALFF instead. Read directly from source, not inferred.

**Practical consequence:** every GNN run this entire session (all 200-epoch jobs, the
embedding-collapse diagnostics, everything) has trained on **old ALFF only**. New ALFF has only
ever been tested through classical-ML baselines (Step 1.6), never through the actual model.
Any old-vs-new comparison at Layer 3 onward needs an explicit mechanism — this layer builds and
verifies one:

- Reconstructed the exact same ASD-then-NC, sorted-filename `file_id` ordering `_load_class()`
  uses to assign `subject_id`, so a swap-in array lines up correctly by position.
- Matched all 956 subjects between this ordering and `alff_new.npz`'s `file_ids` — **0
  missing**.
- Applied the identical global min-max normalization `abideDataset.py` applies to old ALFF, so
  the swap is apples-to-apples at this layer (only the *source* values differ, not the
  normalization convention).
- Result: swap produces a `(956, 90, 3)`, all-finite tensor, confirmed **numerically different**
  from old ALFF's `x` (not an accidental no-op).

This swap mechanism (not a full dataset/cache rebuild) is what later layers will use to run
old-ALFF and new-ALFF through the same graph structure for direct comparison.

## 7. DataLoader / batching — correct

One real batch (`batch_size=32`) pulled cleanly: `x=(2880,3)` (32×90 nodes), `edge_index=(2,
259200)` (32×8100), `edge_weight=(259200,)`, `y=(32,1)`, `subject_id=(32,)`, `batch=(2880,)`,
`num_graphs=32`. All tensors finite.

## Verdict

**Layer 2 is structurally clean — no crashes, no shape/dtype/count errors anywhere.** The one
real, load-bearing finding is not a bug but a **gap**: the dataset class only ever produces
old-ALFF graphs, so nothing about "does new ALFF change GNN behavior" has been tested yet
anywhere in this whole investigation — that's now unblocked via the verified in-memory swap.
Signal-wise, nothing at this layer explains the pipeline's chance-level result either; the
graph objects faithfully carry forward what Layer 1 already established. Layer 3 (main model,
part by part) is next.

**Update**: a genuine normalization bug (joint vs. per-band min-max) was found and fixed in
`_load_class()` — see `normalization_fix_report.md`. Fix is correct and kept, but an isolated
control test showed it is **not** the explanation for the pipeline's chance-level accuracy
(old ALFF gained as much or more from the fix as new ALFF did, contrary to the hypothesis that
it specifically hurt new ALFF). Root cause search continues at Layer 3.
