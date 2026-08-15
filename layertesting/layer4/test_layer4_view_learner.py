"""
Layer 4 audit: VIEW LEARNER, part by part.

Motivated by Layer 3's finding: joint adversarial training spreads subject
embeddings apart from each other (pairwise cos-sim collapses from ~0.999
toward 0.06-0.33 over 30 epochs) but ASD-vs-NC class-mean cosine similarity
barely moves (~0.98-0.99 throughout) -- the encoder never learns anything
ASD/NC-relevant, even though it is clearly learning *something*. This layer
checks whether the VIEW LEARNER is complicit: in adversarial min-max
training it is rewarded for making the augmented view hard to distinguish
from the original, which it could satisfy for free by masking out exactly
the edges that vary most between classes -- destroying the signal before
the main encoder ever sees it.

Checks:
  4a. View learner's own encoder -- same collapse diagnostics as Layer 3
      (pairwise cos-sim, class-mean cos-sim), untrained vs trained.
  4b. THE KEY TEST: per-edge correlation between (i) how diagnostically
      discriminative an edge's actual (as-used) weight is between ASD and
      NC -- a Welch-t-statistic computed directly on data.edge_weight, the
      exact tensor the view learner operates on, not a separately re-derived
      raw-PCC array -- and (ii) the view learner's mean keep-probability
      (mu) at that same edge position, averaged across all 956 subjects. A
      negative correlation means the view learner preferentially masks out
      the more diagnostic edges. Computed for both the corrected (symmetric)
      and paper_exact (unsymmetrized, sample_ordered_concrete_mask) mask
      paths, untrained and trained, both ALFF sources.
  4c. Mask-sampling correctness: mu/edge_mask range [0,1] and finite; the
      corrected path is exactly symmetric ((i,j) mu == (j,i) mu) by
      construction; the paper_exact path is NOT -- verified numerically,
      not just asserted from the code, to confirm the documented
      distinction actually manifests.

Usage:
    python layertesting/layer4/test_layer4_view_learner.py --epochs 30
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.transforms import Compose
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "layertesting" / "layer3"))

from datasets import ABIDEDataset  # noqa: E402
from unsupervised.encoder import TUEncoder  # noqa: E402
from unsupervised.learning import GInfoMinMax  # noqa: E402
from unsupervised.utils import set_tu_dataset_y_shape  # noqa: E402
from unsupervised.view_learner import (ViewLearner, symmetrize_edge_logits, compute_reverse_index,  # noqa: E402
                                        sample_symmetric_logistic_noise, sample_ordered_concrete_mask)
from agcl_ABIDE_queue import MemoryBank_Q, calc_regloss, setup_seed  # noqa: E402
from test_layer3_model_architecture import per_band_minmax, build_new_alff_x, stage_diagnostics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                     datefmt="%H:%M:%S")

NUM_NODES = 90
EDGES_PER_GRAPH = NUM_NODES * NUM_NODES


def view_learner_full_pass(view_learner, full_loader, device):
    """One pass over the full dataset: view learner's own node embeddings
    (subject-mean-pooled, for 4a), and per-edge mu/edge_mask for both mask
    paths + the raw edge_weight used to build them (for 4b/4c)."""
    view_learner.eval()
    encoder = view_learner.encoder

    node_emb_subj, mu_corr_all, mu_paper_all, edge_mask_corr_all = [], [], [], []
    edge_weight_all, labels = [], []
    sym_check_batch = None  # captured once, from the first batch's first graph

    with torch.no_grad():
        for data in full_loader:
            data = data.to(device)
            n_edges = data.edge_index.size(1)
            assert n_edges % EDGES_PER_GRAPH == 0
            B = n_edges // EDGES_PER_GRAPH

            _, node_emb = encoder(data.batch, data.x, data.edge_index, None, data.edge_weight)
            node_emb_subj.append(global_mean_pool(node_emb, data.batch).cpu())

            edge_logits = view_learner(data.batch, data.x, data.edge_index, None, data.edge_weight)
            rev_idx = compute_reverse_index(data.edge_index)
            sym_logits = symmetrize_edge_logits(data.edge_index, edge_logits, rev_idx)
            mu_corr = torch.sigmoid(sym_logits)
            noise = sample_symmetric_logistic_noise(data.edge_index, rev_idx, bias=1e-4, device=device)
            edge_mask_corr = torch.sigmoid((noise + sym_logits) / 1.0)

            mu_paper, _ = sample_ordered_concrete_mask(edge_logits)

            if sym_check_batch is None:
                local_rev_idx = rev_idx[:EDGES_PER_GRAPH].cpu()
                sym_check_batch = dict(
                    mu_corr_g0=mu_corr[:EDGES_PER_GRAPH].cpu(),
                    mu_paper_g0=mu_paper[:EDGES_PER_GRAPH].cpu(),
                    local_rev_idx=local_rev_idx,
                )

            mu_corr_all.append(mu_corr.reshape(B, EDGES_PER_GRAPH).cpu())
            mu_paper_all.append(mu_paper.reshape(B, EDGES_PER_GRAPH).cpu())
            edge_mask_corr_all.append(edge_mask_corr.reshape(B, EDGES_PER_GRAPH).cpu())
            edge_weight_all.append(data.edge_weight.reshape(B, EDGES_PER_GRAPH).cpu())
            labels.append(data.y.cpu().numpy())

    return dict(
        node_emb_subj=torch.cat(node_emb_subj, dim=0),
        mu_corr=torch.cat(mu_corr_all, dim=0),
        mu_paper=torch.cat(mu_paper_all, dim=0),
        edge_mask_corr=torch.cat(edge_mask_corr_all, dim=0),
        edge_weight=torch.cat(edge_weight_all, dim=0),
        labels=np.concatenate(labels, axis=0).ravel(),
        sym_check=sym_check_batch,
    )


def diagnostic_relevance(edge_weight: torch.Tensor, y: np.ndarray) -> torch.Tensor:
    """abs(Welch t-statistic) per edge position between ASD (y==1) and NC
    (y==0), computed on the exact edge_weight tensor the view learner sees.
    Higher = that edge's weight differs more between the two classes."""
    y_t = torch.as_tensor(y)
    asd, nc = edge_weight[y_t == 1], edge_weight[y_t == 0]
    mean_asd, mean_nc = asd.mean(dim=0), nc.mean(dim=0)
    var_asd, var_nc = asd.var(dim=0, unbiased=True), nc.var(dim=0, unbiased=True)
    se = torch.sqrt(var_asd / asd.shape[0] + var_nc / nc.shape[0])
    t_stat = torch.where(se > 0, (mean_asd - mean_nc) / se, torch.zeros_like(se))
    return t_stat.abs()


def report_edge_correlation(rel: torch.Tensor, mu: torch.Tensor, mask_name: str, tag: str):
    diag_mask = torch.eye(NUM_NODES, dtype=torch.bool).reshape(-1)  # exclude self-loops
    a = rel[~diag_mask].numpy()
    b = mu[~diag_mask].numpy()
    r_pearson, _ = pearsonr(a, b)
    r_spearman, _ = spearmanr(a, b)
    logging.info("[%s | %s] corr(diagnostic relevance, mean keep-prob mu): pearson=%+.4f spearman=%+.4f",
                 tag, mask_name, r_pearson, r_spearman)
    return dict(pearson=float(r_pearson), spearman=float(r_spearman))


def report_mask_correctness(pass_result: dict, tag: str):
    mu_corr, mu_paper = pass_result["mu_corr"], pass_result["mu_paper"]
    edge_mask_corr = pass_result["edge_mask_corr"]
    for name, t in [("mu_corr", mu_corr), ("mu_paper", mu_paper), ("edge_mask_corr", edge_mask_corr)]:
        logging.info("[%s] %s: range=[%.4f, %.4f], finite=%s",
                     tag, name, t.min().item(), t.max().item(), bool(torch.isfinite(t).all()))

    sc = pass_result["sym_check"]
    rev = sc["local_rev_idx"]
    corr_sym = torch.allclose(sc["mu_corr_g0"], sc["mu_corr_g0"][rev], atol=1e-6)
    paper_diff = (sc["mu_paper_g0"] - sc["mu_paper_g0"][rev]).abs()
    frac_differ = (paper_diff > 1e-4).float().mean().item()
    logging.info("[%s] corrected mu symmetric (i,j)==(j,i): %s (expect True)", tag, corr_sym)
    logging.info("[%s] paper_exact mu symmetric: %s | fraction of edges where (i,j)!=(j,i): %.4f (expect >0, i.e. NOT symmetric)",
                 tag, torch.allclose(sc["mu_paper_g0"], sc["mu_paper_g0"][rev], atol=1e-6), frac_differ)


def run_source(source_name, dataset, device, epochs):
    logging.info("=" * 78)
    logging.info("SOURCE: %s ALFF", source_name)
    logging.info("=" * 78)

    full_loader = DataLoader(dataset, batch_size=128)
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
        pr = view_learner_full_pass(view_learner, full_loader, device)
        logging.info("--- %s, %s ALFF ---", tag, source_name)
        results[f"{tag}_4a"] = stage_diagnostics(pr["node_emb_subj"], pr["labels"], f"{source_name} {tag} view-encoder")
        rel = diagnostic_relevance(pr["edge_weight"], pr["labels"])
        mean_mu_corr = pr["mu_corr"].mean(dim=0)
        mean_mu_paper = pr["mu_paper"].mean(dim=0)
        results[f"{tag}_4b_corrected"] = report_edge_correlation(rel, mean_mu_corr, "corrected", f"{source_name} {tag}")
        results[f"{tag}_4b_paper"] = report_edge_correlation(rel, mean_mu_paper, "paper_exact", f"{source_name} {tag}")
        report_mask_correctness(pr, f"{source_name} {tag}")

    checkpoint("untrained")

    for epoch in range(1, epochs + 1):
        loss_view_sum, loss_model_sum, n_batches = 0.0, 0.0, 0
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
            model_loss.backward()
            model_optimizer.step()

            memory_bank.push(x_aug.detach(), batch.subject_id)
            n_batches += 1

        if epoch % 5 == 0 or epoch == epochs:
            logging.info("%s ALFF epoch %d: view calc_loss=%.4f | model calc_loss=%.4f",
                         source_name, epoch, loss_view_sum / n_batches, loss_model_sum / n_batches)

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

    old_x = dataset.data.x.clone()
    new_x = build_new_alff_x(dataset)

    all_results = {}

    dataset.data.x = old_x
    all_results["old"] = run_source("OLD", dataset, device, args.epochs)

    dataset.data.x = new_x
    all_results["new"] = run_source("NEW", dataset, device, args.epochs)

    out_path = REPO_ROOT / "layertesting/layer4/layer4_results.npz"
    np.savez(out_path,
              old_results=np.array(all_results["old"], dtype=object),
              new_results=np.array(all_results["new"], dtype=object))
    logging.info("Saved: %s", out_path)

    logging.info("=" * 78)
    logging.info("SUMMARY: edge-level correlation, diagnostic relevance vs. mean keep-prob mu")
    logging.info("=" * 78)
    for src in ["old", "new"]:
        for tag in ["untrained", f"trained_ep{args.epochs}"]:
            for path in ["corrected", "paper"]:
                r = all_results[src][f"{tag}_4b_{path}"]
                logging.info("%-4s %-16s %-10s | pearson=%+.4f spearman=%+.4f",
                             src, tag, path, r["pearson"], r["spearman"])


if __name__ == "__main__":
    main()
