"""
Layer 3 audit: MAIN MODEL, part by part -- localize where the near-total
embedding collapse (cos-sim 0.9997, untrained encoder, per
scripts/diagnose_collapse.py) actually originates inside
TUEncoder/GInfoMinMax, rather than only observing that it exists.

Stages captured, matching TUEncoder.forward() exactly (replicated step by
step here, not monkey-patched, since num_gc_layers=2 for every real run this
session):

  3a. Node embeddings after GIN layer 1 (WGINConv -> BatchNorm -> ReLU+dropout)
  3b. Node embeddings after GIN layer 2 (last layer, no ReLU per the code)
  3c. Node embeddings after the pre-pool F.normalize(x, dim=1) step -- a
      documented deviation from the paper (tu_encoder.py's own comment:
      "paper sums GIN outputs directly, no F.normalize before pooling").
      Hypothesis under test: L2-normalizing every node to unit length before
      SUM-pooling ~90 of them turns ordinary oversmoothing (expected on a
      dense, fully-connected 90x90 graph after 2 message-passing layers)
      into near-total graph-level collapse.
  3d. Graph embedding h -- pooled TWO ways as a direct ablation of that
      hypothesis: h_normalized (real code path, sum-pool of L2-normalized
      nodes) vs. h_raw (sum-pool WITHOUT normalizing first, paper-literal).
  3e. Projected z = proj_head(h), both h_raw and h_normalized carried through.

At each stage: mean pairwise cosine similarity, ASD-vs-NC class-mean cosine
similarity, and a quick 5-fold LinearSVC probe accuracy -- the same
diagnostic trio as scripts/diagnose_collapse.py's embedding_diagnostics().
Node-level stages are mean-pooled into one vector per subject purely for
this measurement (NOT part of the model's own sum-pooling).

Two conditions: untrained (fresh init, eval mode) and after real training
(same loop/hyperparams as scripts/diagnose_collapse.py, already proven to
run end to end).

Two ALFF sources: old (real ABIDEDataset path -- current per-band-min-max
fix, layertesting/layer2, applied automatically) and new (swapped in here
via a fresh per-band-min-max function matching the CURRENT pipeline state --
NOT layer2's original global_minmax swap helper, which predates the fix and
would make old vs. new not apples-to-apples).

Usage:
    python layertesting/layer3/test_layer3_model_architecture.py --epochs 30
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.transforms import Compose
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from datasets import ABIDEDataset  # noqa: E402
from unsupervised.encoder import TUEncoder  # noqa: E402
from unsupervised.learning import GInfoMinMax  # noqa: E402
from unsupervised.utils import set_tu_dataset_y_shape  # noqa: E402
from unsupervised.view_learner import (ViewLearner, symmetrize_edge_logits, compute_reverse_index,  # noqa: E402
                                        sample_symmetric_logistic_noise)
from agcl_ABIDE_queue import MemoryBank_Q, calc_regloss, setup_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                     datefmt="%H:%M:%S")

STAGE_ORDER = ["gin1_subj", "gin2_subj", "normalized_subj", "h_raw", "h_normalized", "z_raw", "z_normalized"]


def per_band_minmax(raw: np.ndarray) -> np.ndarray:
    """raw: [N, 90, 3]. Per-subject, per-band min-max to [0,1] -- matches the
    current (fixed) datasets/abideDataset.py normalization exactly."""
    x = torch.as_tensor(raw, dtype=torch.float64)
    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    span = x_max - x_min
    scaled = torch.where(span > 0, (x - x_min) / span, x)
    return scaled.numpy()


def build_new_alff_x(dataset) -> torch.Tensor:
    """Swap-in tensor for new ALFF, per-band min-max normalized, in the exact
    flattened node ordering dataset.data.x uses (ASD-then-NC, sorted
    filename order, per Layer 2's verified swap mechanism)."""
    npz = np.load(REPO_ROOT / "data/ALFF_need/alff_new.npz", allow_pickle=True)
    new_file_ids = npz["file_ids"].tolist()
    new_fixed = per_band_minmax(npz["alff"])  # [N, 90, 3]

    asd_dir = REPO_ROOT / "data/raw/ASD_ADJ"
    nc_dir = REPO_ROOT / "data/raw/NC_ADJ"
    ordered_file_ids = (
        [f[:-len("_adj.mat")] for f in sorted(os.listdir(asd_dir))] +
        [f[:-len("_adj.mat")] for f in sorted(os.listdir(nc_dir))]
    )
    assert len(ordered_file_ids) == len(dataset), "file_id ordering length mismatch"

    idx_by_id = {fid: i for i, fid in enumerate(new_file_ids)}
    missing = [fid for fid in ordered_file_ids if fid not in idx_by_id]
    assert not missing, f"{len(missing)} subjects missing from new ALFF"

    per_subject = torch.stack([torch.from_numpy(new_fixed[idx_by_id[fid]]) for fid in ordered_file_ids])
    flat = per_subject.reshape(-1, 3).float()
    assert flat.shape == dataset.data.x.shape, (flat.shape, dataset.data.x.shape)
    return flat


def stage_diagnostics(vecs: torch.Tensor, labels: np.ndarray, tag: str) -> dict:
    emb = vecs.detach().cpu()
    labels = np.asarray(labels).ravel()
    norms = emb.norm(dim=1)
    emb_n = F.normalize(emb, dim=1)
    sim = emb_n @ emb_n.T
    n = sim.size(0)
    off_diag_mean = ((sim.sum() - sim.diagonal().sum()) / (n * n - n)).item()

    asd_mean = emb_n[labels == 1].mean(dim=0)
    nc_mean = emb_n[labels == 0].mean(dim=0)
    class_mean_cos_sim = F.cosine_similarity(asd_mean.unsqueeze(0), nc_mean.unsqueeze(0)).item()

    embeddings_np = emb.numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    accs = []
    for train_idx, test_idx in skf.split(embeddings_np, labels):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(dual=False, max_iter=5000))])
        pipe.fit(embeddings_np[train_idx], labels[train_idx])
        accs.append(accuracy_score(labels[test_idx], pipe.predict(embeddings_np[test_idx])))

    result = dict(tag=tag, norm_mean=norms.mean().item(), norm_std=norms.std().item(),
                  pairwise_cos=off_diag_mean, class_mean_cos=class_mean_cos_sim,
                  probe_acc_mean=float(np.mean(accs)), probe_acc_std=float(np.std(accs)))
    logging.info(
        "[%-40s] norm=%.4f+-%.4f | pairwise cos=%.4f | class-mean cos=%.4f | probe acc=%.4f+-%.4f",
        tag, result["norm_mean"], result["norm_std"], result["pairwise_cos"],
        result["class_mean_cos"], result["probe_acc_mean"], result["probe_acc_std"])
    return result


def encoder_stages(encoder, x, edge_index, edge_weight, batch_vec):
    """Replicates TUEncoder.forward() step by step (same ops, same order),
    returning every named intermediate stage instead of only the final
    pooled output -- so collapse can be localized to a specific stage."""
    assert encoder.num_gc_layers == 2, "stage names below assume exactly 2 GIN layers, matching every real run this session"

    x = encoder.convs[0](x, edge_index, edge_weight)
    x = encoder.bns[0](x)
    x = F.dropout(F.relu(x), encoder.drop_ratio, training=encoder.training) if encoder.post_bn_relu \
        else F.dropout(x, encoder.drop_ratio, training=encoder.training)
    gin1_node = x

    x = encoder.convs[1](x, edge_index, edge_weight)
    x = encoder.bns[1](x)
    x = F.dropout(x, encoder.drop_ratio, training=encoder.training)  # last layer: no ReLU
    gin2_node = x

    normalized_node = F.normalize(x, dim=1)  # tu_encoder.py's pre-pool deviation from the paper

    h_normalized = global_add_pool(normalized_node, batch_vec)  # real code path
    h_raw = global_add_pool(gin2_node, batch_vec)  # ablation: skip pre-pool F.normalize

    return {
        "gin1_subj": global_mean_pool(gin1_node, batch_vec),
        "gin2_subj": global_mean_pool(gin2_node, batch_vec),
        "normalized_subj": global_mean_pool(normalized_node, batch_vec),
        "h_raw": h_raw,
        "h_normalized": h_normalized,
    }


def full_pass(model, full_loader, device):
    model.eval()
    encoder = model.encoder
    accum = {k: [] for k in ["gin1_subj", "gin2_subj", "normalized_subj", "h_raw", "h_normalized",
                              "z_raw", "z_normalized"]}
    labels = []
    with torch.no_grad():
        for data in full_loader:
            data = data.to(device)
            stages = encoder_stages(encoder, data.x, data.edge_index, data.edge_weight, data.batch)
            for k in ["gin1_subj", "gin2_subj", "normalized_subj", "h_raw", "h_normalized"]:
                accum[k].append(stages[k].cpu())
            accum["z_raw"].append(model.proj_head(stages["h_raw"]).cpu())
            accum["z_normalized"].append(model.proj_head(stages["h_normalized"]).cpu())
            labels.append(data.y.cpu().numpy())
    out = {k: torch.cat(v, dim=0) for k, v in accum.items()}
    labels = np.concatenate(labels, axis=0).ravel()
    return out, labels


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

    stages, labels = full_pass(model, full_loader, device)
    logging.info("--- Stage diagnostics: UNTRAINED, %s ALFF ---", source_name)
    for key in STAGE_ORDER:
        results[f"untrained_{key}"] = stage_diagnostics(stages[key], labels, f"{source_name} untrained {key}")

    for epoch in range(1, epochs + 1):
        loss_view_sum = 0.0
        loss_model_sum = 0.0
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

    stages, labels = full_pass(model, full_loader, device)
    logging.info("--- Stage diagnostics: TRAINED (%d epochs), %s ALFF ---", epochs, source_name)
    for key in STAGE_ORDER:
        results[f"trained_{key}"] = stage_diagnostics(stages[key], labels, f"{source_name} trained(ep{epochs}) {key}")

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
    logging.info("New ALFF swap built: shape=%s, differs from old: %s",
                 tuple(new_x.shape), not torch.allclose(old_x, new_x))

    all_results = {}

    dataset.data.x = old_x
    all_results["old"] = run_source("OLD", dataset, device, args.epochs)

    dataset.data.x = new_x
    all_results["new"] = run_source("NEW", dataset, device, args.epochs)

    out_path = REPO_ROOT / "layertesting/layer3/layer3_results.npz"
    np.savez(out_path,
              old_results=np.array(all_results["old"], dtype=object),
              new_results=np.array(all_results["new"], dtype=object))
    logging.info("Saved: %s", out_path)

    logging.info("=" * 78)
    logging.info("SUMMARY: pairwise cosine similarity (collapse indicator) by stage")
    logging.info("=" * 78)
    logging.info("%-10s %-18s | %-10s | %-10s", "condition", "stage", "old cos", "new cos")
    for cond in ["untrained", "trained"]:
        for stage in STAGE_ORDER:
            o = all_results["old"][f"{cond}_{stage}"]["pairwise_cos"]
            n = all_results["new"][f"{cond}_{stage}"]["pairwise_cos"]
            logging.info("%-10s %-18s | %10.4f | %10.4f", cond, stage, o, n)


if __name__ == "__main__":
    main()
