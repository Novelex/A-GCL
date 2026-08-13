"""
Tests for the paper_exact profile's memory components: PaperMemoryBank_Q
(zero-initialized FIFO queue, no validity mask) and calc_regloss_paper (no
same-subject exclusion), from agcl_ABIDE_queue.py. These are additive,
side-by-side alternatives to MemoryBank_Q/calc_regloss (see
test_memory_bank.py) -- not a replacement.
"""
import torch
import pytest

from agcl_ABIDE_queue import MemoryBank_Q, PaperMemoryBank_Q, calc_regloss, calc_regloss_paper


def make_paper_bank(max_length=256, feature_dim=32, device='cpu'):
    return PaperMemoryBank_Q(max_length=max_length, feature_dim=feature_dim, device=device)


def test_fresh_paper_queue_is_full_of_zeros_not_empty():
    """Unlike MemoryBank_Q (0 valid entries until pushed to), the paper-literal
    queue reports its full max_length immediately -- the zero rows are real
    negatives from the start, matching the printed paper's pseudocode."""
    bank = make_paper_bank(max_length=256, feature_dim=8)
    memory = bank.get_memory()
    assert memory.size(0) == 256
    assert torch.equal(memory, torch.zeros(256, 8))


def test_paper_bank_has_no_validity_or_subject_tracking():
    bank = make_paper_bank()
    assert not hasattr(bank, 'valid')
    assert not hasattr(bank, 'subject_ids')


def test_paper_push_requires_no_subject_ids():
    bank = make_paper_bank(max_length=64, feature_dim=8)
    bank.push(torch.randn(32, 8))
    memory = bank.get_memory()
    assert memory.size(0) == 64  # still reports full length, half real half zero


def test_paper_push_overwrites_fifo_like_corrected_bank():
    bank = make_paper_bank(max_length=100, feature_dim=4)
    bank.push(torch.zeros(80, 4))
    assert bank.current_index == 80
    bank.push(torch.ones(40, 4))
    # 80 + 40 = 120 > 100, wraps: current_index = 40 - (100-80) = 20
    assert bank.current_index == 20
    memory = bank.get_memory()
    assert memory.size(0) == 100


def test_paper_queue_contains_only_detached_tensors_no_gradients():
    bank = make_paper_bank(max_length=64, feature_dim=8)
    features = torch.randn(32, 8, requires_grad=True)
    bank.push(features)
    assert bank.memory.requires_grad is False
    assert bank.get_memory().grad_fn is None


def test_calc_regloss_paper_counts_zero_rows_as_negatives():
    """A cold-start queue (all zeros) must still produce a finite loss --
    the zero rows are phantom negatives by design in this profile, not
    filtered out the way MemoryBank_Q's valid mask would filter them."""
    torch.manual_seed(0)
    z = torch.randn(4, 8)
    aug = torch.randn(4, 8)
    memory = torch.zeros(256, 8)

    loss = calc_regloss_paper(z, aug, memory)
    assert torch.isfinite(loss)


def test_calc_regloss_paper_does_not_exclude_same_subject():
    """calc_regloss_paper takes no subject ids at all -- a subject's own
    past embedding sitting in the queue counts as a negative like any other
    row, unlike calc_regloss's explicit same-subject exclusion."""
    torch.manual_seed(0)
    z = torch.randn(2, 8)
    aug = torch.randn(2, 8)
    # memory is literally z's own rows -- calc_regloss with matching subject
    # ids would exclude these; calc_regloss_paper has no such mechanism
    memory = z.clone()

    loss = calc_regloss_paper(z, aug, memory)
    assert torch.isfinite(loss)


def test_calc_regloss_paper_matches_calc_regloss_when_nothing_would_be_excluded():
    """When no subject overlaps memory (so calc_regloss's exclusion mask is
    a no-op), calc_regloss_paper and calc_regloss must agree exactly --
    they differ only in whether same-subject exclusion and cold-start
    zero-filtering happen, not in the underlying InfoNCE math."""
    torch.manual_seed(0)
    z = torch.randn(4, 8)
    aug = torch.randn(4, 8)
    memory = torch.randn(6, 8)
    anchor_subject_ids = torch.tensor([1, 2, 3, 4])
    memory_subject_ids = torch.tensor([10, 11, 12, 13, 14, 15])

    loss_corrected = calc_regloss(z, aug, memory, memory_subject_ids, anchor_subject_ids, temperature=0.1)
    loss_paper = calc_regloss_paper(z, aug, memory, temperature=0.1)
    assert torch.allclose(loss_corrected, loss_paper)


def test_paper_profile_bank_and_corrected_bank_diverge_at_cold_start():
    """Documents the intended divergence in behavior at epoch start: the
    corrected bank has zero usable negatives until batches are pushed
    (bug 6's fix), while the paper bank always has max_length negatives,
    including phantom all-zero ones."""
    corrected = MemoryBank_Q(max_length=64, feature_dim=8, device='cpu')
    paper = PaperMemoryBank_Q(max_length=64, feature_dim=8, device='cpu')

    valid_memory, _ = corrected.get_valid_memory()
    paper_memory = paper.get_memory()

    assert valid_memory.size(0) == 0
    assert paper_memory.size(0) == 64
