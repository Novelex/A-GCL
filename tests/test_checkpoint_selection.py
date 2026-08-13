"""
Tests for the "no test-based epoch selection" guarantee: during training,
kf_embedding_evaluation(..., include_test=False) must never compute a real
test score, and the checkpoint dict written by the training scripts must
carry everything needed to resume evaluation from exactly one selected epoch.
"""
from unittest.mock import patch

import numpy as np
import torch
import pytest
from sklearn.svm import LinearSVC
from torch_geometric.data import Data

from unsupervised.embedding_evaluation import EmbeddingEvaluation, create_fixed_splits
from unsupervised.encoder.tu_encoder import TUEncoder
from unsupervised.learning.ginfominmax import GInfoMinMax


class _TUEvaluatorStub:
    eval_metric = 'accuracy'

    def eval(self, input_dict):
        y_true = np.ravel(input_dict['y_true'])
        y_pred = np.ravel(input_dict['y_pred'])
        return {'accuracy': float((y_true == y_pred).mean())}


class _GraphDataset:
    """Small synthetic dataset with the same per-item interface ABIDEDataset
    exposes (x, edge_index, edge_weight, y), enough to drive a real forward
    pass through TUEncoder/GInfoMinMax without needing real ABIDE data."""
    def __init__(self, n_per_class=20, num_nodes=8, num_features=3):
        self._items = []
        idx = torch.arange(num_nodes)
        src = idx.repeat_interleave(num_nodes)
        dst = idx.repeat(num_nodes)
        edge_index = torch.stack([src, dst], dim=0)
        for label in (0, 1):
            for _ in range(n_per_class):
                x = torch.rand(num_nodes, num_features)
                edge_weight = torch.rand(edge_index.size(1))
                self._items.append(Data(x=x, edge_index=edge_index.clone(),
                                         edge_weight=edge_weight, y=torch.tensor([label])))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def make_ee(model_device='cpu'):
    evaluator = _TUEvaluatorStub()
    return EmbeddingEvaluation(
        LinearSVC(dual=False, fit_intercept=True, max_iter=2000),
        evaluator, task_type='classification', num_tasks=1,
        device=torch.device(model_device), param_search=False,
    )


def make_model():
    encoder = TUEncoder(num_dataset_features=3, emb_dim=8, num_gc_layers=2)
    return GInfoMinMax(encoder, proj_hidden_dim=8)


def test_include_test_false_never_yields_real_test_scores():
    dataset = _GraphDataset(n_per_class=20)
    splits = create_fixed_splits(dataset, n_splits=5, seed=123)
    model = make_model()
    ee = make_ee()

    train_score, val_score, test_score = ee.kf_embedding_evaluation(
        model, dataset, splits, representation='z', include_test=False)

    assert np.isfinite(train_score[0])
    assert np.isfinite(val_score[0])
    # every test-related summary statistic must be NaN -- nothing usable
    assert all(np.isnan(v) for v in test_score)


def test_fit_on_train_val_uses_combined_80_percent_not_train_only():
    """The final evaluation must refit the classifier on train+val (80% of
    the outer split) before scoring test, not just the 64% train fold --
    verified by capturing the actual row count passed to
    ee_binary_classification (i.e. what classifier.fit() sees), not just
    checking that the outputs happen to be finite."""
    dataset = _GraphDataset(n_per_class=20)
    splits = create_fixed_splits(dataset, n_splits=5, seed=123)
    model = make_model()

    def fit_sizes(fit_on_train_val):
        ee = make_ee()
        sizes = []
        with patch.object(ee, 'ee_binary_classification', wraps=ee.ee_binary_classification) as spy:
            ee.kf_embedding_evaluation(
                model, dataset, splits, representation='z',
                include_test=True, fit_on_train_val=fit_on_train_val)
            for call in spy.call_args_list:
                fit_emb = call.args[0]
                sizes.append(fit_emb.shape[0])
        return sizes

    sizes_train_only = fit_sizes(fit_on_train_val=False)
    sizes_train_val = fit_sizes(fit_on_train_val=True)

    assert len(sizes_train_only) == len(splits)
    assert len(sizes_train_val) == len(splits)

    for split, n_train_only, n_train_val in zip(splits, sizes_train_only, sizes_train_val):
        expected_train_only = len(split["train"])
        expected_train_val = len(split["train"]) + len(split["val"])
        assert n_train_only == expected_train_only
        assert n_train_val == expected_train_val
        assert n_train_val > n_train_only


def test_include_test_true_yields_real_test_scores():
    dataset = _GraphDataset(n_per_class=20)
    splits = create_fixed_splits(dataset, n_splits=5, seed=123)
    model = make_model()
    ee = make_ee()

    _, _, test_score = ee.kf_embedding_evaluation(
        model, dataset, splits, representation='z', include_test=True)

    assert np.isfinite(test_score[0])  # acc_mean is a real number now


def test_checkpoint_dict_round_trips_model_state(tmp_path):
    model = make_model()
    view_learner_state = {'dummy': torch.tensor([1.0])}
    checkpoint_path = tmp_path / "ckpt.pt"

    torch.save({
        'epoch': 25,
        'model_state_dict': model.state_dict(),
        'view_state_dict': view_learner_state,
        'validation_score': [0.61, 0.02],
    }, checkpoint_path)

    loaded = torch.load(checkpoint_path, weights_only=False)
    assert loaded['epoch'] == 25
    assert 'model_state_dict' in loaded
    assert 'validation_score' in loaded

    fresh_model = make_model()
    fresh_model.load_state_dict(loaded['model_state_dict'])
    for p1, p2 in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(p1, p2)
