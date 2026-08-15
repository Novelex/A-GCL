"""
Layer 3 follow-up: pooling/oversmoothing ablation.

Layer 6 found the trained encoder's embeddings are dominated by connectivity
magnitude (node-strength R^2~0.65-0.67) rather than diagnosis, traced to
WGINConv scaling messages by edge_weight combined with global_add_pool
SUMMING (not averaging) node embeddings, with nothing normalizing the
result. A direct supervised check (same TUEncoder architecture, trained
end-to-end with real cross-entropy on DX_GROUP, no contrastive objective
at all) confirmed this is architectural, not a self-supervision problem:
even with real labels, the same architecture stayed at chance (~49%),
while a plain linear model on the raw FC edges alone (no GNN) reached
65-69%. The signal exists and survives to the raw edges; it's specifically
lost inside the encoder's message-passing + pooling.

This tests three single-variable ablations against that known chance-level
baseline (2-layer GIN, sum-pool, no post-pool normalization: acc~0.4917,
auc~0.49-0.50, both ALFF sources), each changing exactly one thing:

  mean_pool -- global_mean_pool instead of global_add_pool (2 GIN layers
               unchanged). Directly removes the magnitude-sum sensitivity.
  l2norm    -- unchanged sum-pool, but L2-normalize the pooled graph
               embedding h before the classifier head. Discards magnitude
               regardless of pooling method.
  1layer    -- 1 GIN layer instead of 2 (sum-pool unchanged). The graph is
               already fully connected (every node reaches every other node
               in one hop), so a second layer means every node has
               effectively seen the whole graph twice -- tests whether that
               second hop is where oversmoothing finishes the job.

Same supervised, real-labels training harness as before (5-fold
StratifiedKFold, held-out validation split per fold for early stopping,
no test-based epoch selection) -- the fastest, cleanest way to isolate
representational capacity from the self-supervised objective's own issues
(already separately investigated in Layers 4 and 6).

Usage:
    python layertesting/layer3/test_pooling_ablation.py --variant mean_pool --source old
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.transforms import Compose
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "layertesting" / "layer3"))

from datasets import ABIDEDataset  # noqa: E402
from unsupervised.convs.wgin_conv import WGINConv  # noqa: E402
from unsupervised.utils import set_tu_dataset_y_shape  # noqa: E402
from agcl_ABIDE_queue import setup_seed  # noqa: E402
from test_layer3_model_architecture import build_new_alff_x  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                     datefmt="%H:%M:%S")

SEED = 123
N_SPLITS = 5
PATIENCE = 15
EPOCHS = 100

VARIANT_CONFIG = {
    "mean_pool": dict(num_gc_layers=2, pool_type="mean", post_pool_l2norm=False),
    "l2norm":    dict(num_gc_layers=2, pool_type="sum",  post_pool_l2norm=True),
    "1layer":    dict(num_gc_layers=1, pool_type="sum",  post_pool_l2norm=False),
}
# known baseline, already established (2-layer GIN, sum-pool, no post-pool norm):
BASELINE = dict(old=dict(acc=0.4917, auc=0.4941), new=dict(acc=0.4917, auc=0.5008))


class ConfigurableEncoder(torch.nn.Module):
    """Same conv/bn construction as unsupervised/encoder/tu_encoder.py's
    TUEncoder, but with pool_type and post_pool_l2norm made configurable for
    this ablation -- everything else (WGINConv, per-node F.normalize before
    pooling, dropout/relu schedule) identical to the real encoder."""

    def __init__(self, num_dataset_features, emb_dim=32, num_gc_layers=2, drop_ratio=0.3,
                 pool_type="sum", post_pool_l2norm=False, normalize_nodes=True,
                 message_relu=True, post_bn_relu=True):
        super().__init__()
        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.pool_type = pool_type
        self.post_pool_l2norm = post_pool_l2norm
        self.normalize_nodes = normalize_nodes
        self.post_bn_relu = post_bn_relu
        self.out_graph_dim = emb_dim

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            else:
                nn = Sequential(Linear(num_dataset_features, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            self.convs.append(WGINConv(nn, message_relu=message_relu))
            self.bns.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batch, x, edge_index, edge_weight):
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            if i == self.num_gc_layers - 1:
                x = F.dropout(x, self.drop_ratio, training=self.training)
            elif self.post_bn_relu:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            else:
                x = F.dropout(x, self.drop_ratio, training=self.training)
        if self.normalize_nodes:
            x = F.normalize(x, dim=1)
        if self.pool_type == "sum":
            h = global_add_pool(x, batch)
        elif self.pool_type == "mean":
            h = global_mean_pool(x, batch)
        else:
            raise ValueError(self.pool_type)
        if self.post_pool_l2norm:
            h = F.normalize(h, dim=1)
        return h, x


class SupervisedGNN(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Linear(encoder.out_graph_dim, 1)

    def forward(self, batch, x, edge_index, edge_weight):
        h, _ = self.encoder(batch, x, edge_index, edge_weight)
        return self.classifier(h).squeeze(-1)


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.batch, batch.x, batch.edge_index, batch.edge_weight)
            all_logits.append(logits.cpu())
            all_y.append(batch.y.cpu())
    logits = torch.cat(all_logits)
    y = torch.cat(all_y).numpy().ravel()
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    return accuracy_score(y, preds), roc_auc_score(y, probs)


def train_one_fold(dataset, train_idx, val_idx, test_idx, device, variant_cfg, fold_num, tag):
    setup_seed(SEED)
    encoder = ConfigurableEncoder(num_dataset_features=3, emb_dim=32, drop_ratio=0.3,
                                   normalize_nodes=True, message_relu=True, post_bn_relu=True,
                                   **variant_cfg)
    model = SupervisedGNN(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(dataset[train_idx], batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=128)
    test_loader = DataLoader(dataset[test_idx], batch_size=128)

    best_val_acc = -1.0
    best_state = None
    patience_ctr = 0
    best_epoch = 0
    epoch = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.batch, batch.x, batch.edge_index, batch.edge_weight)
            loss = criterion(logits, batch.y.float().view(-1))
            loss.backward()
            optimizer.step()

        val_acc, val_auc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    model.load_state_dict(best_state)
    test_acc, test_auc = evaluate(model, test_loader, device)
    logging.info("  [%s fold %d/%d] stopped at epoch %d (best val_acc=%.4f @ epoch %d) | "
                 "test acc=%.4f auc=%.4f", tag, fold_num, N_SPLITS, epoch, best_val_acc,
                 best_epoch, test_acc, test_auc)
    return test_acc, test_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANT_CONFIG.keys()))
    parser.add_argument("--source", required=True, choices=["old", "new"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{args.variant}/{args.source}"
    logging.info("Device: %s | variant=%s config=%s | source=%s ALFF",
                 device, args.variant, VARIANT_CONFIG[args.variant], args.source)

    my_transforms = Compose([set_tu_dataset_y_shape])
    dataset = ABIDEDataset(str(REPO_ROOT / "data"), "ABIDE", transform=my_transforms)
    dataset.data.y = dataset.data.y.squeeze()
    y = dataset.data.y.numpy().ravel()
    logging.info("Loaded dataset: N=%d, ASD=%d, NC=%d", len(dataset), int((y == 1).sum()), int((y == 0).sum()))

    if args.source == "new":
        dataset.data.x = build_new_alff_x(dataset)

    variant_cfg = VARIANT_CONFIG[args.variant]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    accs, aucs = [], []
    for fold_num, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=0.15, stratify=y[trainval_idx], random_state=SEED)
        acc, auc = train_one_fold(dataset, train_idx, val_idx, test_idx, device, variant_cfg, fold_num, tag)
        accs.append(acc)
        aucs.append(auc)

    acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
    auc_mean, auc_std = float(np.mean(aucs)), float(np.std(aucs))
    base = BASELINE[args.source]
    logging.info("=" * 78)
    logging.info("[%s] acc=%.4f +- %.4f | auc=%.4f +- %.4f", tag, acc_mean, acc_std, auc_mean, auc_std)
    logging.info("[%s] delta vs. known chance-level baseline: acc %+.4f, auc %+.4f",
                 tag, acc_mean - base["acc"], auc_mean - base["auc"])

    out_path = REPO_ROOT / f"layertesting/layer3/pooling_ablation_{args.variant}_{args.source}.npz"
    np.savez(out_path, acc_mean=acc_mean, acc_std=acc_std, auc_mean=auc_mean, auc_std=auc_std)
    logging.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
