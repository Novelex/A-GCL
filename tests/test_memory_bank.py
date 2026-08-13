"""
Tests for MemoryBank_Q (the FIFO negative-sample queue) and calc_regloss's
same-subject exclusion, from agcl_ABIDE_queue.py.
"""
import torch
import pytest

from agcl_ABIDE_queue import MemoryBank_Q, calc_regloss


def make_bank(max_length=256, feature_dim=32, device='cpu'):
    return MemoryBank_Q(max_length=max_length, feature_dim=feature_dim, device=device)


def test_fresh_queue_has_zero_valid_negatives():
    bank = make_bank()
    memory, subject_ids = bank.get_valid_memory()
    assert memory.size(0) == 0
    assert subject_ids.size(0) == 0


def test_queue_contains_only_detached_tensors_no_gradients():
    bank = make_bank(max_length=64, feature_dim=8)
    features = torch.randn(32, 8, requires_grad=True)
    subject_ids = torch.arange(32)
    bank.push(features, subject_ids)

    assert bank.memory.requires_grad is False
    memory, _ = bank.get_valid_memory()
    assert memory.requires_grad is False
    # pushed values should be detached copies, not the same graph-attached tensor
    assert memory.grad_fn is None


def test_after_one_batch_32_valid_entries():
    bank = make_bank(max_length=256, feature_dim=8)
    bank.push(torch.randn(32, 8), torch.arange(32))
    memory, _ = bank.get_valid_memory()
    assert memory.size(0) == 32
    assert bank.valid.sum().item() == 32


def test_after_eight_batches_queue_is_full():
    bank = make_bank(max_length=256, feature_dim=8)
    for b in range(8):
        bank.push(torch.randn(32, 8), torch.arange(b * 32, (b + 1) * 32))
    memory, _ = bank.get_valid_memory()
    assert memory.size(0) == 256
    assert bank.valid.all()


def test_ninth_batch_overwrites_oldest_and_size_never_exceeds_max():
    bank = make_bank(max_length=256, feature_dim=8)
    for b in range(9):
        bank.push(torch.full((32, 8), float(b)), torch.arange(b * 32, (b + 1) * 32))
    memory, subject_ids = bank.get_valid_memory()
    assert memory.size(0) == 256  # never exceeds max_length
    # the very first batch (subject_ids 0..31, value 0.0) must be gone
    assert 0 not in subject_ids.tolist()
    # the most recent batch (subject_ids 256..287, value 8.0) must be present
    assert 256 in subject_ids.tolist()


def test_pointer_wraps_correctly():
    bank = make_bank(max_length=100, feature_dim=4)
    bank.push(torch.zeros(80, 4), torch.arange(80))
    assert bank.current_index == 80
    bank.push(torch.ones(40, 4), torch.arange(80, 120))
    # 80 + 40 = 120 > 100, wraps: current_index = 40 - (100-80) = 20
    assert bank.current_index == 20
    memory, subject_ids = bank.get_valid_memory()
    assert memory.size(0) == 100  # fully overwritten, all valid


def test_current_batch_not_visible_before_its_own_push():
    """Documents the intended call order (bug 5's fix): a batch's own
    embeddings must not be visible via get_valid_memory() until AFTER
    that batch's loss has already been computed and push() called."""
    bank = make_bank(max_length=256, feature_dim=8)
    this_batch_subject_ids = torch.arange(32)

    memory_before, subject_ids_before = bank.get_valid_memory()
    assert not set(this_batch_subject_ids.tolist()) & set(subject_ids_before.tolist())

    bank.push(torch.randn(32, 8), this_batch_subject_ids)
    _, subject_ids_after = bank.get_valid_memory()
    assert set(this_batch_subject_ids.tolist()).issubset(set(subject_ids_after.tolist()))


def test_matching_subject_ids_excluded_from_calc_regloss_negatives():
    torch.manual_seed(0)
    z = torch.randn(4, 8)
    aug = torch.randn(4, 8)
    anchor_subject_ids = torch.tensor([1, 2, 3, 4])

    # memory contains subject 1's own stale embedding plus 3 unrelated ones
    memory = torch.randn(4, 8)
    memory_subject_ids = torch.tensor([1, 5, 6, 7])

    # should not raise, and should not error out even though subject 1's
    # only "negative" is itself excluded (still has 3 valid negatives)
    loss = calc_regloss(z, aug, memory, memory_subject_ids, anchor_subject_ids)
    assert torch.isfinite(loss)


def test_calc_regloss_handles_zero_valid_negatives_for_all_anchors():
    """If literally every memory slot belongs to the same subject as every
    anchor, there are no valid negatives at all -- must return a finite,
    zero-ish loss rather than NaN/-inf."""
    z = torch.randn(2, 8)
    aug = torch.randn(2, 8)
    anchor_subject_ids = torch.tensor([1, 1])
    memory = torch.randn(2, 8)
    memory_subject_ids = torch.tensor([1, 1])

    loss = calc_regloss(z, aug, memory, memory_subject_ids, anchor_subject_ids)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_calc_regloss_handles_mixed_valid_and_zero_negative_anchors():
    """Mixed case: subject 1 has zero valid negatives (every memory slot is
    subject 1), subject 2 has valid negatives (neither memory slot is
    subject 2). has_negative.any() is True overall, so the old code did not
    early-return -- but subject 1's row still had an all -inf denominator,
    producing +inf/NaN that leaked into the batch mean. The loss must stay
    finite, computed only from subject 2's row."""
    torch.manual_seed(0)
    z = torch.randn(2, 8)
    aug = torch.randn(2, 8)
    anchor_subject_ids = torch.tensor([1, 2])
    memory = torch.randn(2, 8)
    memory_subject_ids = torch.tensor([1, 1])

    loss = calc_regloss(z, aug, memory, memory_subject_ids, anchor_subject_ids)
    assert torch.isfinite(loss)

    # cross-check: computing the loss from subject 2's row alone (the only
    # anchor with any valid negative) must match exactly
    z2, aug2 = z[1:2], aug[1:2]
    anchor2 = anchor_subject_ids[1:2]
    loss_subject2_only = calc_regloss(z2, aug2, memory, memory_subject_ids, anchor2)
    assert torch.allclose(loss, loss_subject2_only)


def test_calc_regloss_mixed_negatives_gradients_are_finite():
    """Filtering happens BEFORE logsumexp now, so no -inf/NaN intermediate
    value should ever be created for the invalid-negative anchor -- confirmed
    by checking the gradients flowing back into z and aug are finite too,
    not just the forward loss value."""
    torch.manual_seed(0)
    z = torch.randn(2, 8, requires_grad=True)
    aug = torch.randn(2, 8, requires_grad=True)
    anchor_subject_ids = torch.tensor([1, 2])
    memory = torch.randn(2, 8)
    memory_subject_ids = torch.tensor([1, 1])

    loss = calc_regloss(z, aug, memory, memory_subject_ids, anchor_subject_ids)
    loss.backward()

    assert z.grad is not None
    assert aug.grad is not None
    assert torch.isfinite(z.grad).all()
    assert torch.isfinite(aug.grad).all()
