"""
Tests for create_fixed_splits() -- stratified, reproducible, non-overlapping
train/val/test indices generated once and reused for every evaluation.
"""
import numpy as np
import torch
import pytest

from unsupervised.embedding_evaluation import create_fixed_splits


class _Item:
    def __init__(self, y):
        self.y = torch.tensor([y])


class _MockDataset:
    """Minimal stand-in for ABIDEDataset -- only needs __getitem__/__len__
    with a .y attribute, exactly what create_fixed_splits reads."""
    def __init__(self, labels):
        self._items = [_Item(y) for y in labels]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def make_dataset(n_asd=455, n_nc=501):
    labels = [1] * n_asd + [0] * n_nc
    return _MockDataset(labels)


def test_same_seed_gives_identical_indices():
    dataset = make_dataset()
    splits_a = create_fixed_splits(dataset, n_splits=5, seed=123)
    splits_b = create_fixed_splits(dataset, n_splits=5, seed=123)

    assert len(splits_a) == len(splits_b) == 5
    for a, b in zip(splits_a, splits_b):
        assert np.array_equal(a["train"], b["train"])
        assert np.array_equal(a["val"], b["val"])
        assert np.array_equal(a["test"], b["test"])


def test_different_seed_gives_different_indices():
    dataset = make_dataset()
    splits_a = create_fixed_splits(dataset, n_splits=5, seed=123)
    splits_b = create_fixed_splits(dataset, n_splits=5, seed=456)
    assert not np.array_equal(splits_a[0]["test"], splits_b[0]["test"])


def test_no_overlap_between_train_val_test():
    dataset = make_dataset()
    for split in create_fixed_splits(dataset, n_splits=5, seed=123):
        train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)


def test_every_subject_appears_in_exactly_one_outer_test_fold():
    dataset = make_dataset()
    splits = create_fixed_splits(dataset, n_splits=5, seed=123)
    all_test_indices = np.concatenate([s["test"] for s in splits])
    assert len(all_test_indices) == len(dataset)
    assert len(set(all_test_indices.tolist())) == len(dataset)


def test_class_proportions_preserved_across_folds():
    dataset = make_dataset(n_asd=455, n_nc=501)
    labels = np.array([int(dataset[i].y.item()) for i in range(len(dataset))])
    overall_asd_ratio = labels.mean()

    for split in create_fixed_splits(dataset, n_splits=5, seed=123):
        for part in ("train", "val", "test"):
            idx = split[part]
            fold_ratio = labels[idx].mean()
            assert abs(fold_ratio - overall_asd_ratio) < 0.05
