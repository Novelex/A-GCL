"""
Layer 6 audit: LOSS FUNCTIONS, decomposed.

Layers 3-4 established that training works by its own lights (embeddings
decollapse, gradients flow) but never develops ASD-vs-NC class separation.
The InfoNCE contrastive loss has no way to "know" about diagnosis -- it only
rewards "recognize this subject's augmented view among a batch of others".
That is satisfiable by learning to separate subjects along ANY axis of
individual difference. ABIDE has a famous, well-documented confound sitting
right there: 19 different acquisition sites in this cohort, with real
scanner/protocol differences -- exactly the kind of thing self-supervised
contrastive training could latch onto instead of diagnosis.

Checks:
  6a. Does the contrastive loss succeed at its OWN task? Batch retrieval
      accuracy (is subject i's true augmented-view embedding the closest
      match to its original, among the other subjects in the same batch?)
      and the mean positive-vs-negative similarity gap. Untrained vs
      trained, both ALFF sources. Confirms training is not merely inert --
      it is solving self-recognition well, just not usefully.
  6b. Loss decomposition over training -- contrastive calc_loss, memory-bank
      cr_loss, and the regularizer (mean keep-prob) logged every epoch (not
      just at checkpoints) to see which term actually dominates.
  6c. THE KEY TEST -- what axis is actually being learned, if not
      diagnosis? Same quick-probe methodology used throughout, applied to
      SITE_ID (19 classes, data/subject_tr.csv) instead of DX_GROUP, on the
      trained z embeddings. Also a Ridge-regression probe against
      node-strength (sum |edge_weight|, excluding self-loops) -- a second,
      classic connectomics nuisance covariate, cheap to add alongside site.

Usage:
    python layertesting/layer6/test_layer6_loss_functions.py --epochs 30
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "layertesting" / "layer3"))

from datasets import ABIDEDataset  # noqa: E402
from unsupervised.encoder import TUEncoder  # noqa: E402
from unsupervised.learning import GInfoMinMax  # noqa: E402
from unsupervised.utils import set_tu_dataset_y_shape  # noqa: E402
from unsupervised.view_learner import (ViewLearner, symmetrize_edge_logits, compute_reverse_index,  # noqa: E402
                                        sample_symmetric_logistic_noise)
from agcl_ABIDE_queue import MemoryBank_Q, calc_regloss, setup_seed  # noqa: E402
from test_layer3_model_architecture import build_new_alff_x  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                     datefmt="%H:%M:%S")

NUM_NODES = 90
EDGES_PER_GRAPH = NUM_NODES * NUM_NODES


def ordered_file_ids():
    asd_dir = REPO_ROOT / "data/raw/ASD_ADJ"
    nc_dir = REPO_ROOT / "data/raw/NC_ADJ"
    return ([f[:-len("_adj.mat")] for f in sorted(os.listdir(asd_dir))] +
            [f[:-len("_adj.mat")] for f in sorted(os.listdir(nc_dir))])


def load_site_labels(file_ids) -> tuple[np.ndarray, list]:
    site_by_fid = {}
    with open(REPO_ROOT / "data/subject_tr.csv") as f:
        for row in csv.DictReader(f):
            site_by_fid[row["FILE_ID"]] = row["SITE_ID"]
    sites = [site_by_fid[fid] for fid in file_ids]
    uniq = sorted(set(sites))
    site_to_idx = {s: i for i, s in enumerate(uniq)}
    return np.array([site_to_idx[s] for s in sites]), uniq


def classification_probe(embeddings: torch.Tensor, labels: np.ndarray, name: str) -> dict:
    emb = embeddings.numpy()
    labels = np.asarray(labels).ravel()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    accs = []
    for train_idx, test_idx in skf.split(emb, labels):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(dual=False, max_iter=5000))])
        pipe.fit(emb[train_idx], labels[train_idx])
        accs.append(accuracy_score(labels[test_idx], pipe.predict(emb[test_idx])))
    n_classes = len(np.unique(labels))
    result = dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)), n_classes=n_classes)
    logging.info("[%-28s] probe acc=%.4f +- %.4f (chance~%.4f, n_classes=%d)",
                 name, result["acc_mean"], result["acc_std"], 1.0 / n_classes, n_classes)
    return result


def regression_probe(embeddings: torch.Tensor, target: np.ndarray, name: str) -> dict:
    emb = embeddings.numpy()
    target = np.asarray(target).ravel()
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    r2s = []
    for train_idx, test_idx in kf.split(emb):
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
        pipe.fit(emb[train_idx], target[train_idx])
        r2s.append(r2_score(target[test_idx], pipe.predict(emb[test_idx])))
    result = dict(r2_mean=float(np.mean(r2s)), r2_std=float(np.std(r2s)))
    logging.info("[%-28s] probe R^2=%.4f +- %.4f (0 = no better than predicting the mean)",
                 name, result["r2_mean"], result["r2_std"])
    return result


def retrieval_diagnostics(model, view_learner, loader, device, tag):
    model.eval()
    view_learner.eval()
    n_correct, n_total = 0, 0
    pos_sims, neg_sims = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z, _ = model(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)
            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)
            rev_idx = compute_reverse_index(batch.edge_index)
            sym_logits = symmetrize_edge_logits(batch.edge_index, edge_logits, rev_idx)
            noise = sample_symmetric_logistic_noise(batch.edge_index, rev_idx, bias=1e-4, device=device)
            edge_mask = torch.sigmoid((noise + sym_logits) / 1.0)
            aug_edge_weight = batch.edge_weight * edge_mask
            z_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, aug_edge_weight)

            zn, zan = F.normalize(z, dim=1), F.normalize(z_aug, dim=1)
            sim = zn @ zan.T
            B = sim.size(0)
            top1 = sim.argmax(dim=1)
            n_correct += (top1 == torch.arange(B, device=device)).sum().item()
            n_total += B

            pos_sims.append(sim.diagonal().cpu())
            neg_mask = ~torch.eye(B, dtype=torch.bool, device=device)
            neg_sims.append(sim[neg_mask].view(B, B - 1).mean(dim=1).cpu())

    pos_sims = torch.cat(pos_sims)
    neg_sims = torch.cat(neg_sims)
    top1_acc = n_correct / n_total
    gap = (pos_sims - neg_sims).mean().item()
    logging.info("[%s] batch retrieval top1 acc=%.4f (chance~0.03125, B=32) | mean pos sim=%.4f | "
                 "mean neg sim=%.4f | gap=%.4f", tag, top1_acc, pos_sims.mean().item(),
                 neg_sims.mean().item(), gap)
    return dict(top1_acc=top1_acc, pos_sim_mean=pos_sims.mean().item(),
                neg_sim_mean=neg_sims.mean().item(), gap=gap)


def full_z_pass(model, full_loader, device):
    model.eval()
    diag_mask = torch.eye(NUM_NODES, dtype=torch.bool).reshape(-1)
    zs, labels, subject_ids, node_strength = [], [], [], []
    with torch.no_grad():
        for data in full_loader:
            data = data.to(device)
            z, _ = model(data.batch, data.x, data.edge_index, None, data.edge_weight)
            zs.append(z.cpu())
            labels.append(data.y.cpu().numpy())
            subject_ids.append(data.subject_id.cpu().numpy())
            n_edges = data.edge_index.size(1)
            B = n_edges // EDGES_PER_GRAPH
            ew = data.edge_weight.reshape(B, EDGES_PER_GRAPH)
            node_strength.append(ew[:, ~diag_mask].abs().sum(dim=1).cpu())
    return dict(z=torch.cat(zs, dim=0), labels=np.concatenate(labels, axis=0).ravel(),
                subject_id=np.concatenate(subject_ids, axis=0).ravel(),
                node_strength=torch.cat(node_strength, dim=0).numpy())


def run_source(source_name, dataset, device, epochs, site_by_subject_id):
    logging.info("=" * 78)
    logging.info("SOURCE: %s ALFF", source_name)
    logging.info("=" * 78)

    full_loader = DataLoader(dataset, batch_size=128)
    fixed_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    setup_seed(123)
    memory_bank = MemoryBank_Q(max_length=256, feature_dim=32, device=device)

    model = GInfoMinMax(
        TUEncoder(num_dataset_features=3, emb_dim=32, num_gc_layers=2, drop_ratio=0.3,
                  pooling_type="standard", normalize_nodes=True, message_relu=True, post_bn_relu=True),
        32).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    view_learner = ViewLearner(TUEncoder(num_dataset_features=3, emb_dim=32, num_gc_layers=2, drop_ratio=0.3,
                                          pooling_type="standard", normalize_nodes=True, message_relu=True,
                                          post_bn_relu=True), mlp_edge_model_dim=64).to(device)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=0.0005)

    results = {}

    def checkpoint(tag):
        results[f"{tag}_retrieval"] = retrieval_diagnostics(model, view_learner, fixed_loader, device, f"{source_name} {tag}")
        zp = full_z_pass(model, full_loader, device)
        site_labels = site_by_subject_id[zp["subject_id"]]
        logging.info("--- %s, %s ALFF: probes on trained z ---", tag, source_name)
        results[f"{tag}_dx_probe"] = classification_probe(zp["z"], zp["labels"], f"{source_name} {tag} DX")
        results[f"{tag}_site_probe"] = classification_probe(zp["z"], site_labels, f"{source_name} {tag} SITE")
        results[f"{tag}_strength_probe"] = regression_probe(zp["z"], zp["node_strength"], f"{source_name} {tag} node-strength")

    checkpoint("untrained")

    for epoch in range(1, epochs + 1):
        loss_view_sum, cr_view_sum, reg_sum = 0.0, 0.0, 0.0
        loss_model_sum, cr_model_sum = 0.0, 0.0
        n_batches = 0
        for batch in dataloader:
            batch = batch.to(device)

            view_learner.train()
            view_learner.zero_grad()
            model.eval()

            x, _ = model(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)
            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)
            rev_idx = compute_reverse_index(batch.edge_index)
            sym_logits = symmetrize_edge_logits(batch.edge_index, edge_logits, rev_idx)
            mu = torch.sigmoid(sym_logits)
            noise = sample_symmetric_logistic_noise(batch.edge_index, rev_idx, bias=1e-4, device=device)
            edge_mask = torch.sigmoid((noise + sym_logits) / 1.0)
            aug_edge_weight = batch.edge_weight * edge_mask
            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, aug_edge_weight)

            keep_prob = mu.mean()
            reg = keep_prob

            valid_memory, valid_memory_ids = memory_bank.get_valid_memory()
            if valid_memory.size(0) == 0:
                cr_loss = x.sum() * 0.0
            else:
                cr_loss = calc_regloss(x, x_aug, valid_memory, valid_memory_ids, batch.subject_id, temperature=0.1)
            batch_loss = model.calc_loss(x, x_aug, temperature=0.2, sym=True)
            view_loss = batch_loss + 1.0 * (2.0 * reg) + 0.4 * cr_loss
            loss_view_sum += batch_loss.item()
            cr_view_sum += cr_loss.item()
            reg_sum += reg.item()
            (-view_loss).backward()
            view_optimizer.step()

            model.train()
            view_learner.eval()
            model.zero_grad()

            x, _ = model(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)
            with torch.no_grad():
                edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None, batch.edge_weight)
                rev_idx = compute_reverse_index(batch.edge_index)
                sym_logits = symmetrize_edge_logits(batch.edge_index, edge_logits, rev_idx)
                noise = sample_symmetric_logistic_noise(batch.edge_index, rev_idx, bias=1e-4, device=device)
                edge_mask = torch.sigmoid((noise + sym_logits) / 1.0)
            aug_edge_weight = batch.edge_weight * edge_mask
            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, aug_edge_weight)

            valid_memory, valid_memory_ids = memory_bank.get_valid_memory()
            if valid_memory.size(0) == 0:
                cr_loss = x.sum() * 0.0
            else:
                cr_loss = calc_regloss(x, x_aug, valid_memory, valid_memory_ids, batch.subject_id, temperature=0.1)
            batch_loss_m = model.calc_loss(x, x_aug, temperature=0.2, sym=True)
            model_loss = batch_loss_m + 0.4 * cr_loss
            loss_model_sum += batch_loss_m.item()
            cr_model_sum += cr_loss.item()
            model_loss.backward()
            model_optimizer.step()

            memory_bank.push(x_aug.detach(), batch.subject_id)
            n_batches += 1

        logging.info(
            "%s ALFF epoch %2d: view[calc_loss=%.4f cr_loss=%.4f reg(keep_prob)=%.4f] | "
            "model[calc_loss=%.4f cr_loss=%.4f]",
            source_name, epoch, loss_view_sum / n_batches, cr_view_sum / n_batches, reg_sum / n_batches,
            loss_model_sum / n_batches, cr_model_sum / n_batches)

    checkpoint(f"trained_ep{epochs}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    my_transforms = Compose([set_tu_dataset_y_shape])
    dataset = ABIDEDataset(str(REPO_ROOT / "data"), "ABIDE", transform=my_transforms)
    dataset.data.y = dataset.data.y.squeeze()
    logging.info("Loaded dataset: N=%d", len(dataset))

    file_ids = ordered_file_ids()
    site_by_subject_id, site_names = load_site_labels(file_ids)
    logging.info("Loaded SITE_ID for %d subjects, %d unique sites", len(site_by_subject_id), len(site_names))

    old_x = dataset.data.x.clone()
    new_x = build_new_alff_x(dataset)

    all_results = {}

    dataset.data.x = old_x
    all_results["old"] = run_source("OLD", dataset, device, args.epochs, site_by_subject_id)

    dataset.data.x = new_x
    all_results["new"] = run_source("NEW", dataset, device, args.epochs, site_by_subject_id)

    out_path = REPO_ROOT / "layertesting/layer6/layer6_results.npz"
    np.savez(out_path,
              old_results=np.array(all_results["old"], dtype=object),
              new_results=np.array(all_results["new"], dtype=object))
    logging.info("Saved: %s", out_path)

    logging.info("=" * 78)
    logging.info("SUMMARY: trained-z probe accuracy, DX vs SITE vs node-strength")
    logging.info("=" * 78)
    for src in ["old", "new"]:
        tag = f"trained_ep{args.epochs}"
        dx = all_results[src][f"{tag}_dx_probe"]
        site = all_results[src][f"{tag}_site_probe"]
        strength = all_results[src][f"{tag}_strength_probe"]
        retr = all_results[src][f"{tag}_retrieval"]
        logging.info("%-4s DX acc=%.4f (chance %.4f) | SITE acc=%.4f (chance %.4f) | "
                     "node-strength R^2=%.4f | retrieval top1=%.4f",
                     src, dx["acc_mean"], 0.5, site["acc_mean"], 1.0 / site["n_classes"],
                     strength["r2_mean"], retr["top1_acc"])


if __name__ == "__main__":
    main()
