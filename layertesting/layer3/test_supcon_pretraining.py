"""
Layer 3 follow-up: Supervised Contrastive (SupCon, Khosla et al. 2020)
representation-learning test.

Motivation, in sequence: (1) the unsupervised contrastive pipeline never
develops class separation because its loss has no label information at all
(Layers 3/4/6); (2) plain end-to-end supervised classification on the SAME
architecture also stayed at chance -- but that check used early stopping on
a noisy per-epoch validation accuracy, and several folds "converged" within
the first 1-8 epochs, which is genuinely ambiguous between "the architecture
can't learn this" and "the optimization never got a fair, well-monitored
shot" (a concern the user raised directly). This test removes that ambiguity
two ways: no early-stopping-on-a-noisy-metric (fixed epoch budget instead,
downstream evaluation only happens once, after pretraining is done), and the
SupCon training LOSS itself is logged every epoch so convergence can be
checked directly, not inferred from a single downstream number.

Method: SupCon loss (single-view variant -- positives are OTHER SAME-CLASS
subjects in the same batch, not an augmented view of the same subject; no
ViewLearner, no adversarial masking) directly supervises the SAME encoder
architecture used throughout (2-layer GIN, sum-pool -- the pipeline's actual
default, not one of the pooling-ablation variants, to isolate "does adding
label supervision to the loss alone fix it" as a single variable). After
pretraining, embeddings are frozen and evaluated with the same downstream
protocol used everywhere else this session (StandardScaler + LinearSVC +
GridSearchCV, proper nested 5-fold CV, encoder trained only on each fold's
training subjects so nothing about the test fold ever leaks into
pretraining).

Usage:
    python layertesting/layer3/test_supcon_pretraining.py --epochs 100 --source old
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
from torch_geometric.transforms import Compose
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "layertesting" / "layer3"))

from datasets import ABIDEDataset  # noqa: E402
from unsupervised.utils import set_tu_dataset_y_shape  # noqa: E402
from agcl_ABIDE_queue import setup_seed  # noqa: E402
from test_layer3_model_architecture import build_new_alff_x  # noqa: E402
from test_pooling_ablation import ConfigurableEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                     datefmt="%H:%M:%S")

SEED = 123
N_SPLITS = 5
BATCH_SIZE = 64
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Khosla et al. 2020, single-view variant: positives are other
    same-class samples in the batch (not an augmented view of self)."""
    device = z.device
    B = z.size(0)
    sim = torch.matmul(z, z.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1, 1)
    same_label = torch.eq(labels, labels.T).float().to(device)
    self_mask = torch.eye(B, device=device)
    positive_mask = same_label - self_mask
    logits_mask = 1 - self_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = positive_mask.sum(dim=1)
    valid = pos_count > 0
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()


def get_embeddings(encoder, proj_head, dataset, idx, device):
    encoder.eval()
    proj_head.eval()
    loader = DataLoader(dataset[idx], batch_size=128)
    zs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h, _ = encoder(batch.batch, batch.x, batch.edge_index, batch.edge_weight)
            z = F.normalize(proj_head(h), dim=1)
            zs.append(z.cpu())
            ys.append(batch.y.cpu())
    return torch.cat(zs, dim=0).numpy(), torch.cat(ys, dim=0).numpy().ravel()


def train_one_fold(dataset, train_idx, test_idx, device, epochs, fold_num, source_name):
    setup_seed(SEED)
    encoder = ConfigurableEncoder(num_dataset_features=3, emb_dim=32, num_gc_layers=2, drop_ratio=0.3,
                                   pool_type="sum", post_pool_l2norm=False, normalize_nodes=True,
                                   message_relu=True, post_bn_relu=True).to(device)
    proj_head = Sequential(Linear(32, 32), ReLU(inplace=True), Linear(32, 32)).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()), lr=0.0005)

    train_loader = DataLoader(dataset[train_idx], batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    loss_first, loss_last = None, None
    for epoch in range(1, epochs + 1):
        encoder.train()
        proj_head.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            h, _ = encoder(batch.batch, batch.x, batch.edge_index, batch.edge_weight)
            z = F.normalize(proj_head(h), dim=1)
            loss = supcon_loss(z, batch.y.view(-1), temperature=0.1)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= n_batches
        if epoch == 1:
            loss_first = epoch_loss
        loss_last = epoch_loss
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            logging.info("  [%s fold %d SupCon] epoch %3d: loss=%.4f", source_name, fold_num, epoch, epoch_loss)

    z_train, y_train = get_embeddings(encoder, proj_head, dataset, train_idx, device)
    z_test, y_test = get_embeddings(encoder, proj_head, dataset, test_idx, device)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(dual=False, max_iter=10000))])
    clf = GridSearchCV(pipe, {"clf__C": C_GRID}, cv=5, scoring="accuracy", n_jobs=4)
    clf.fit(z_train, y_train)
    pred = clf.predict(z_test)
    dec = clf.decision_function(z_test)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, dec)

    logging.info("  [%s fold %d/%d] SupCon loss %.4f -> %.4f | downstream: acc=%.4f auc=%.4f best_C=%s",
                 source_name, fold_num, N_SPLITS, loss_first, loss_last, acc, auc, clf.best_params_["clf__C"])
    return acc, auc, loss_first, loss_last


def run_source(source_name, dataset, y, device, epochs):
    logging.info("=" * 78)
    logging.info("SOURCE: %s ALFF (SupCon pretraining, same 2-layer/sum-pool architecture)", source_name)
    logging.info("=" * 78)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    accs, aucs, losses_first, losses_last = [], [], [], []
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        acc, auc, lf, ll = train_one_fold(dataset, train_idx, test_idx, device, epochs, fold_num, source_name)
        accs.append(acc)
        aucs.append(auc)
        losses_first.append(lf)
        losses_last.append(ll)

    logging.info("[%s] SupCon downstream: acc=%.4f +- %.4f | auc=%.4f +- %.4f | "
                 "mean loss %.4f -> %.4f", source_name, np.mean(accs), np.std(accs),
                 np.mean(aucs), np.std(aucs), np.mean(losses_first), np.mean(losses_last))
    return dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                loss_first_mean=float(np.mean(losses_first)), loss_last_mean=float(np.mean(losses_last)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    my_transforms = Compose([set_tu_dataset_y_shape])
    dataset = ABIDEDataset(str(REPO_ROOT / "data"), "ABIDE", transform=my_transforms)
    dataset.data.y = dataset.data.y.squeeze()
    y = dataset.data.y.numpy().ravel()
    logging.info("Loaded dataset: N=%d, ASD=%d, NC=%d", len(dataset), int((y == 1).sum()), int((y == 0).sum()))

    old_x = dataset.data.x.clone()
    new_x = build_new_alff_x(dataset)

    all_results = {}

    dataset.data.x = old_x
    all_results["old"] = run_source("OLD", dataset, y, device, args.epochs)

    dataset.data.x = new_x
    all_results["new"] = run_source("NEW", dataset, y, device, args.epochs)

    out_path = REPO_ROOT / "layertesting/layer3/supcon_results.npz"
    np.savez(out_path, old_results=np.array(all_results["old"], dtype=object),
              new_results=np.array(all_results["new"], dtype=object))
    logging.info("Saved: %s", out_path)

    logging.info("=" * 78)
    logging.info("SUMMARY: SupCon pretraining vs. everything else tested so far")
    logging.info("=" * 78)
    logging.info("Unsupervised contrastive pipeline (real 200-epoch runs): ~50%% (chance)")
    logging.info("Plain end-to-end supervised, same arch (early-stopped, possibly premature): ~49%%")
    logging.info("L1-LogReg on raw FC edges directly (no GNN): acc=0.6506 auc=0.6939")
    for src in ["old", "new"]:
        r = all_results[src]
        logging.info("SupCon pretraining, %s ALFF: acc=%.4f +- %.4f | auc=%.4f +- %.4f | loss %.4f -> %.4f",
                     src, r["acc_mean"], r["acc_std"], r["auc_mean"], r["auc_std"],
                     r["loss_first_mean"], r["loss_last_mean"])


if __name__ == "__main__":
    main()
