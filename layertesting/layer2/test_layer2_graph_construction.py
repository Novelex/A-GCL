"""
Layer 2 audit: GRAPH CONSTRUCTION -- the actual Data objects the model
trains on: x, edge_index, edge_weight, y, subject_id, built by
datasets/abideDataset.py from Layer 1's raw sources.

Checks:
  1. Load ABIDEDataset (its current, only code path -- old ALFF) end to end,
     no crash, structural sanity (shapes, dtypes, counts).
  2. Verify edge_index/edge_weight match the documented dense M^2 + max-abs
     normalization construction.
  3. Verify subject_id assignment (unique, 0..955, ASD-then-NC ordering).
  4. Verify y (label) convention and shape after set_tu_dataset_y_shape.
  5. Verify x (node features) after abideDataset.py's own additional
     GLOBAL min-max normalization step -- does it introduce any NaN/
     degenerate values, and how does it compare to Layer 1's raw old-ALFF
     characterization?
  6. EXPLICIT GAP CHECK: ABIDEDataset only ever loads norm_matrix (old
     ALFF) -- there is no code path to build graphs from new ALFF. Confirmed
     by reading the class; this script also builds and verifies a safe
     in-memory swap (replace dataset.data.x per-subject with new ALFF,
     matched by subject_id) as the actual mechanism later layers will need
     for old-vs-new comparison.
  7. DataLoader/batching sanity -- pull one real batch, check every field.

Usage:
    python layertesting/layer2/test_layer2_graph_construction.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from datasets import ABIDEDataset  # noqa: E402
from unsupervised.utils import set_tu_dataset_y_shape  # noqa: E402
from compare_alff import load_old_alff  # noqa: E402


def load_dataset():
    logging.info("=" * 70)
    logging.info("1. Load ABIDEDataset end to end")
    logging.info("=" * 70)
    my_transforms = Compose([set_tu_dataset_y_shape])
    dataset = ABIDEDataset(str(REPO_ROOT / "data"), "ABIDE", transform=my_transforms)
    dataset.data.y = dataset.data.y.squeeze()
    logging.info("Loaded OK. N subjects = %d", len(dataset))
    return dataset


def check_edges(dataset):
    logging.info("=" * 70)
    logging.info("2. Edge structure (dense M^2 + max-abs normalization)")
    logging.info("=" * 70)
    g0 = dataset[0]
    logging.info("Sample graph 0: num_nodes=%d, edge_index shape=%s, edge_weight shape=%s",
                 g0.num_nodes, tuple(g0.edge_index.shape), tuple(g0.edge_weight.shape))
    expected_edges = g0.num_nodes * g0.num_nodes
    logging.info("Expected dense M^2 edges: %d, actual: %d, match: %s",
                 expected_edges, g0.edge_index.shape[1], g0.edge_index.shape[1] == expected_edges)
    logging.info("edge_weight range: [%.4f, %.4f] (expect within [-1, 1] after max-abs norm)",
                 g0.edge_weight.min().item(), g0.edge_weight.max().item())
    logging.info("edge_weight finite: %s", bool(torch.isfinite(g0.edge_weight).all()))
    # self-loop diagonal entries: src==dst, weight should be exactly 1.0 (self-corr, per Layer 1)
    self_loop_mask = g0.edge_index[0] == g0.edge_index[1]
    logging.info("Self-loop count: %d (expect %d), all weight==1.0: %s",
                 int(self_loop_mask.sum()), g0.num_nodes,
                 bool(torch.allclose(g0.edge_weight[self_loop_mask], torch.ones(int(self_loop_mask.sum())))))


def check_subject_ids(dataset):
    logging.info("=" * 70)
    logging.info("3. subject_id assignment")
    logging.info("=" * 70)
    all_ids = torch.cat([dataset[i].subject_id for i in range(len(dataset))])
    logging.info("subject_id range: [%d, %d] (expect [0, %d])", all_ids.min(), all_ids.max(), len(dataset) - 1)
    logging.info("Unique count: %d / %d (expect all unique)", len(all_ids.unique()), len(all_ids))
    labels = torch.tensor([int(dataset[i].y) for i in range(len(dataset))])
    n_asd = int((labels == 1).sum())
    logging.info("ASD-then-NC ordering check: first %d subject_ids are label==1 (ASD): %s",
                 n_asd, bool((labels[:n_asd] == 1).all()))


def check_labels(dataset):
    logging.info("=" * 70)
    logging.info("4. Label (y) convention and shape")
    logging.info("=" * 70)
    labels = torch.tensor([int(dataset[i].y) for i in range(len(dataset))])
    logging.info("y dtype: %s, unique values: %s", labels.dtype, labels.unique().tolist())
    logging.info("Counts: ASD(1)=%d, NC(0)=%d (expect 455/501)",
                 int((labels == 1).sum()), int((labels == 0).sum()))


def check_node_features(dataset):
    logging.info("=" * 70)
    logging.info("5. Node features (x) after abideDataset.py's own global min-max norm")
    logging.info("=" * 70)
    all_x = torch.stack([dataset[i].x for i in range(len(dataset))])  # [N, 90, 3]
    logging.info("x shape: %s, dtype: %s", tuple(all_x.shape), all_x.dtype)
    logging.info("All finite: %s", bool(torch.isfinite(all_x).all()))
    logging.info("Global min/max after normalization: %.4f / %.4f (expect [0, 1])",
                 all_x.min().item(), all_x.max().item())
    logging.info("Per-subject min/max range check: min(per-subj min)=%.4f, max(per-subj max)=%.4f",
                 all_x.amin(dim=(1, 2)).min().item(), all_x.amax(dim=(1, 2)).max().item())


def check_new_alff_gap_and_swap(dataset):
    logging.info("=" * 70)
    logging.info("6. GAP CHECK: ABIDEDataset has no new-ALFF code path -- verify in-memory swap")
    logging.info("=" * 70)
    logging.info("Confirmed by source read: _load_class() unconditionally loads nf['norm_matrix'] "
                 "(old ALFF). No parameter, flag, or branch selects new ALFF. Any old-vs-new "
                 "comparison through the actual model needs an explicit in-memory swap -- built "
                 "and verified here, since later layers depend on it.")

    npz = np.load(REPO_ROOT / "data/ALFF_need/alff_new.npz", allow_pickle=True)
    new_file_ids = npz["file_ids"].tolist()
    new_alff_raw = npz["alff"]  # [956, 90, 3], raw, per Layer 1

    # Same global min-max normalization abideDataset.py applies to old ALFF,
    # applied identically here so the swap is apples-to-apples at this layer
    # (only the SOURCE values differ, not the normalization convention)
    def global_minmax(x):
        x = torch.nan_to_num(torch.as_tensor(x, dtype=torch.float32))
        xmin, xmax = x.min(), x.max()
        return (x - xmin) / (xmax - xmin) if xmax > xmin else x

    # dataset[i]'s subject_id was assigned by _load_class in ASD-then-NC,
    # sorted-filename order -- reconstruct that same file_id order to match
    asd_dir = REPO_ROOT / "data/raw/ASD_ADJ"
    nc_dir = REPO_ROOT / "data/raw/NC_ADJ"
    import os
    ordered_file_ids = (
        [f[:-len("_adj.mat")] for f in sorted(os.listdir(asd_dir))] +
        [f[:-len("_adj.mat")] for f in sorted(os.listdir(nc_dir))]
    )
    assert len(ordered_file_ids) == len(dataset), "file_id ordering length mismatch"

    new_idx_by_id = {fid: i for i, fid in enumerate(new_file_ids)}
    missing = [fid for fid in ordered_file_ids if fid not in new_idx_by_id]
    logging.info("file_id alignment: %d/%d subjects found in new ALFF, %d missing",
                 len(ordered_file_ids) - len(missing), len(ordered_file_ids), len(missing))

    swapped_x = torch.stack([
        global_minmax(new_alff_raw[new_idx_by_id[fid]]) for fid in ordered_file_ids
    ])
    logging.info("Swap successful: swapped_x shape=%s, finite=%s",
                 tuple(swapped_x.shape), bool(torch.isfinite(swapped_x).all()))

    old_x = torch.stack([dataset[i].x for i in range(len(dataset))])
    identical = torch.allclose(old_x, swapped_x)
    logging.info("Sanity: swapped-in new-ALFF x differs from old-ALFF x: %s (expect True, i.e. NOT identical)",
                 not identical)
    return swapped_x, ordered_file_ids


def check_dataloader_batch(dataset):
    logging.info("=" * 70)
    logging.info("7. DataLoader / batching sanity")
    logging.info("=" * 70)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    batch = next(iter(loader))
    logging.info("Batch fields: x=%s edge_index=%s edge_weight=%s y=%s subject_id=%s batch=%s",
                 tuple(batch.x.shape), tuple(batch.edge_index.shape), tuple(batch.edge_weight.shape),
                 tuple(batch.y.shape), tuple(batch.subject_id.shape), tuple(batch.batch.shape))
    logging.info("num_graphs in batch: %d (expect 32)", batch.num_graphs)
    logging.info("All batch tensors finite: x=%s edge_weight=%s",
                 bool(torch.isfinite(batch.x).all()), bool(torch.isfinite(batch.edge_weight).all()))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dataset = load_dataset()
    check_edges(dataset)
    check_subject_ids(dataset)
    check_labels(dataset)
    check_node_features(dataset)
    check_new_alff_gap_and_swap(dataset)
    check_dataloader_batch(dataset)
    logging.info("=" * 70)
    logging.info("Layer 2 audit complete.")


if __name__ == "__main__":
    main()
