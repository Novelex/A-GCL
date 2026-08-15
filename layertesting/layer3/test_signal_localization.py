"""
Layer 3 follow-up: exactly WHERE inside the encoder does the raw-FC
diagnostic signal disappear?

Three different training objectives (unsupervised contrastive, plain
supervised classification, SupCon) all converged to chance on this
architecture, while a plain linear model on the raw FC edges alone gets
65-69%. That rules out the loss function as the cause -- something about
the forward pass itself destroys the signal, independent of how the
network is trained. This probes the UNTRAINED (random-init, eval mode)
network at every individual sub-operation inside the first two GIN layers,
not just at the layer boundaries Layer 3 already checked, using the exact
same quick-probe methodology (StandardScaler + LinearSVC, 5-fold CV) that
got 65-69% on the raw edges directly. Untrained is the right condition to
test: it isolates the STRUCTURAL bottleneck from any training-dynamics
confound, and every trained variant tried so far (3 different objectives)
landed at the same chance level anyway, so the untrained analysis is
representative of the ceiling any of them could reach.

Stages, in the exact order WGINConv actually computes them (using its own
propagate() and lin() directly -- not reimplemented, to guarantee this
matches the real forward pass exactly):

  raw_edges       -- the 4005 unique raw edge weights themselves (known
                      ceiling: 65-69% via L1-LogReg)
  agg1_pre_mlp    -- GIN layer 1, immediately after message+aggregate
                      (sum_j relu(x_j)*edge_weight(i,j)) + self term,
                      BEFORE the shared MLP (self.lin). [90,3] per subject.
  gin1_post_mlp   -- after self.lin (Linear->ReLU->Linear, 3->32->32)
  gin1_post_bn    -- after BatchNorm + ReLU + dropout (off in eval mode)
  agg2_pre_mlp    -- GIN layer 2, message+aggregate, BEFORE its MLP
  gin2_post_mlp   -- after layer 2's self.lin
  gin2_post_bn    -- after BatchNorm (last layer: no ReLU)
  h               -- after global_add_pool (the real graph embedding)

Node-level stages are mean-pooled into one vector per subject purely for
this measurement, same convention as every prior layer.

Usage:
    python layertesting/layer3/test_signal_localization.py
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
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
NUM_NODES = 90
EDGES_PER_GRAPH = NUM_NODES * NUM_NODES
EMB_DIM = 32


def probe(X: np.ndarray, y: np.ndarray, name: str) -> dict:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    accs, aucs = [], []
    for train_idx, test_idx in skf.split(X, y):
        # dual='auto': raw_edges has 4005 features vs ~764 training samples
        # (features >> samples -> dual=True is the fast formulation there);
        # every other stage has only 3-32 features (samples >> features ->
        # dual=False, matching every prior probe in this session). Hardcoding
        # dual=False across all 8 stages (as every earlier <=270-dim probe
        # did) made the 4005-dim raw_edges stage pathologically slow.
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(dual="auto", max_iter=10000))])
        clf = GridSearchCV(pipe, {"clf__C": C_GRID}, cv=5, scoring="accuracy", n_jobs=4)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        dec = clf.decision_function(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        aucs.append(roc_auc_score(y[test_idx], dec))
    result = dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                  auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)))
    logging.info("[%-16s] acc=%.4f +- %.4f | auc=%.4f +- %.4f",
                 name, result["acc_mean"], result["acc_std"], result["auc_mean"], result["auc_std"])
    return result


def build_untrained_convs():
    setup_seed(SEED)
    nn1 = Sequential(Linear(3, EMB_DIM), ReLU(), Linear(EMB_DIM, EMB_DIM))
    conv1 = WGINConv(nn1, message_relu=True)
    bn1 = torch.nn.BatchNorm1d(EMB_DIM)
    nn2 = Sequential(Linear(EMB_DIM, EMB_DIM), ReLU(), Linear(EMB_DIM, EMB_DIM))
    conv2 = WGINConv(nn2, message_relu=True)
    bn2 = torch.nn.BatchNorm1d(EMB_DIM)
    for m in (conv1, bn1, conv2, bn2):
        m.eval()
    return conv1, bn1, conv2, bn2


def run_stages(dataset, source_name, y):
    conv1, bn1, conv2, bn2 = build_untrained_convs()
    full_loader = DataLoader(dataset, batch_size=128)

    accum = {k: [] for k in ["raw_edges", "agg1_pre_mlp", "gin1_post_mlp", "gin1_post_bn",
                              "agg2_pre_mlp", "gin2_post_mlp", "gin2_post_bn", "h"]}
    diag_mask = torch.eye(NUM_NODES, dtype=torch.bool).reshape(-1)

    with torch.no_grad():
        for data in full_loader:
            n_edges = data.edge_index.size(1)
            B = n_edges // EDGES_PER_GRAPH
            ew = data.edge_weight.reshape(B, EDGES_PER_GRAPH)
            accum["raw_edges"].append(ew[:, ~diag_mask])

            x = data.x
            ei = data.edge_index
            w = data.edge_weight

            agg1 = conv1.propagate(ei, x=(x, x), edge_weight=w, size=None)
            agg1 = agg1 + (1 + conv1.eps) * x
            accum["agg1_pre_mlp"].append(global_mean_pool(agg1, data.batch))

            post_mlp1 = conv1.lin(agg1)
            accum["gin1_post_mlp"].append(global_mean_pool(post_mlp1, data.batch))

            post_bn1 = bn1(post_mlp1)
            post_bn1_relu = F.relu(post_bn1)  # not last layer -> ReLU applies; dropout off (eval)
            accum["gin1_post_bn"].append(global_mean_pool(post_bn1_relu, data.batch))

            agg2 = conv2.propagate(ei, x=(post_bn1_relu, post_bn1_relu), edge_weight=w, size=None)
            agg2 = agg2 + (1 + conv2.eps) * post_bn1_relu
            accum["agg2_pre_mlp"].append(global_mean_pool(agg2, data.batch))

            post_mlp2 = conv2.lin(agg2)
            accum["gin2_post_mlp"].append(global_mean_pool(post_mlp2, data.batch))

            post_bn2 = bn2(post_mlp2)  # last layer -> no ReLU, dropout off (eval)
            accum["gin2_post_bn"].append(global_mean_pool(post_bn2, data.batch))

            x_normalized = F.normalize(post_bn2, dim=1)
            h = global_add_pool(x_normalized, data.batch)
            accum["h"].append(h)

    logging.info("=" * 78)
    logging.info("SOURCE: %s ALFF -- signal localization, stage by stage", source_name)
    logging.info("=" * 78)
    results = {}
    for stage in ["raw_edges", "agg1_pre_mlp", "gin1_post_mlp", "gin1_post_bn",
                  "agg2_pre_mlp", "gin2_post_mlp", "gin2_post_bn", "h"]:
        X = torch.cat(accum[stage], dim=0).numpy()
        results[stage] = probe(X, y, stage)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["old", "new"])
    args = parser.parse_args()

    my_transforms = Compose([set_tu_dataset_y_shape])
    dataset = ABIDEDataset(str(REPO_ROOT / "data"), "ABIDE", transform=my_transforms)
    dataset.data.y = dataset.data.y.squeeze()
    y = dataset.data.y.numpy().ravel()
    logging.info("Loaded dataset: N=%d, ASD=%d, NC=%d", len(dataset), int((y == 1).sum()), int((y == 0).sum()))

    if args.source == "new":
        dataset.data.x = build_new_alff_x(dataset)

    result = run_stages(dataset, args.source.upper(), y)

    out_path = REPO_ROOT / f"layertesting/layer3/signal_localization_results_{args.source}.npz"
    np.savez(out_path, result=np.array(result, dtype=object))
    logging.info("Saved: %s", out_path)

    logging.info("=" * 78)
    logging.info("SUMMARY: accuracy by stage, %s ALFF", args.source)
    logging.info("=" * 78)
    logging.info("%-16s | %-10s", "Stage", "Accuracy")
    for stage in ["raw_edges", "agg1_pre_mlp", "gin1_post_mlp", "gin1_post_bn",
                  "agg2_pre_mlp", "gin2_post_mlp", "gin2_post_bn", "h"]:
        logging.info("%-16s | %.4f", stage, result[stage]["acc_mean"])


if __name__ == "__main__":
    main()
