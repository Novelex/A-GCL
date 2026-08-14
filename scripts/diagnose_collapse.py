"""
One-off diagnostic: agcl_ABIDE_queue.py's corrected/budget 200-epoch run showed
Model Loss never decreasing, while a trivial linear-SVM baseline on the raw FC
matrices gets ~64% accuracy -- meaning the data has signal the encoder isn't
learning to extract. This script mirrors the real training loop (same model
construction, same profile) but replaces the expensive periodic GridSearchCV
eval with cheap embedding-collapse diagnostics, and logs calc_loss/cr_loss
separately instead of only their sum, so we can see WHERE training is stuck.

Not part of the pipeline -- a throwaway investigation script.
"""
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose

from datasets import ABIDEDataset
from unsupervised.encoder import TUEncoder
from unsupervised.learning import GInfoMinMax
from unsupervised.utils import set_tu_dataset_y_shape
from unsupervised.view_learner import (ViewLearner, symmetrize_edge_logits, compute_reverse_index,
                                       sample_symmetric_logistic_noise)
from agcl_ABIDE_queue import MemoryBank_Q, calc_regloss, setup_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


def embedding_diagnostics(model, loader, device, tag):
    embeddings, labels = model.get_embeddings(loader, device, representation='z')
    labels = labels.ravel()
    emb = torch.from_numpy(embeddings)
    norms = emb.norm(dim=1)
    emb_n = F.normalize(emb, dim=1)
    sim = emb_n @ emb_n.T
    n = sim.size(0)
    off_diag_mean = (sim.sum() - sim.diagonal().sum()) / (n * n - n)

    asd_mean = emb_n[labels == 1].mean(dim=0)
    nc_mean = emb_n[labels == 0].mean(dim=0)
    class_mean_cos_sim = F.cosine_similarity(asd_mean.unsqueeze(0), nc_mean.unsqueeze(0)).item()

    # cheap, single-threaded linear probe -- trend only, NOT a substitute for
    # the real GridSearchCV/kf_embedding_evaluation protocol
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    accs = []
    for train_idx, test_idx in skf.split(embeddings, labels):
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC(dual=False, max_iter=5000))])
        pipe.fit(embeddings[train_idx], labels[train_idx])
        accs.append(accuracy_score(labels[test_idx], pipe.predict(embeddings[test_idx])))

    logging.info(
        "[%s] embedding norm mean=%.4f std=%.4f | mean pairwise cos-sim=%.4f | "
        "class-mean cos-sim (ASD vs NC)=%.4f | quick-probe acc=%.4f +- %.4f",
        tag, norms.mean().item(), norms.std().item(), off_diag_mean.item(),
        class_mean_cos_sim, np.mean(accs), np.std(accs))


def run(epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(123)

    my_transforms = Compose([set_tu_dataset_y_shape])
    dataset = ABIDEDataset('data', 'ABIDE', transform=my_transforms)
    dataset.data.y = dataset.data.y.squeeze()
    full_loader = DataLoader(dataset, batch_size=128)

    memory_bank = MemoryBank_Q(max_length=256, feature_dim=32, device=device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    model = GInfoMinMax(
        TUEncoder(num_dataset_features=3, emb_dim=32, num_gc_layers=2, drop_ratio=0.3,
                  pooling_type='standard', normalize_nodes=True, message_relu=True, post_bn_relu=True),
        32).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    view_learner = ViewLearner(TUEncoder(num_dataset_features=3, emb_dim=32, num_gc_layers=2, drop_ratio=0.3,
                                         pooling_type='standard', normalize_nodes=True, message_relu=True,
                                         post_bn_relu=True), mlp_edge_model_dim=64).to(device)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=0.0005)

    model.eval()
    embedding_diagnostics(model, full_loader, device, "epoch 0 (untrained)")

    for epoch in range(1, epochs + 1):
        calc_loss_view_sum = 0.0
        cr_loss_view_sum = 0.0
        calc_loss_model_sum = 0.0
        cr_loss_model_sum = 0.0
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
            calc_loss_view_sum += batch_loss.item()
            cr_loss_view_sum += cr_loss.item()
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
            calc_loss_model_sum += batch_loss_m.item()
            cr_loss_model_sum += cr_loss.item()
            model_loss.backward()
            model_optimizer.step()

            memory_bank.push(x_aug.detach(), batch.subject_id)
            n_batches += 1

        logging.info(
            "Epoch %d: view calc_loss=%.4f cr_loss=%.4f | model calc_loss=%.4f cr_loss=%.4f",
            epoch, calc_loss_view_sum / n_batches, cr_loss_view_sum / n_batches,
            calc_loss_model_sum / n_batches, cr_loss_model_sum / n_batches)

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            embedding_diagnostics(model, full_loader, device, f"epoch {epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    run(args.epochs)
