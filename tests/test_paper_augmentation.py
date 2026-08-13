"""
Tests for the paper_exact profile's mask sampling: sample_ordered_concrete_mask,
the literal per-directed-edge Binary-Concrete relaxation with independent
noise per direction -- no reverse-edge symmetrization, unlike
sample_symmetric_logistic_noise (see test_augmentation.py). Additive,
side-by-side alternative, not a replacement.
"""
import torch
import pytest

from unsupervised.view_learner import compute_reverse_index, sample_ordered_concrete_mask


def fully_connected_edge_index(num_nodes):
    idx = torch.arange(num_nodes)
    src = idx.repeat_interleave(num_nodes)
    dst = idx.repeat(num_nodes)
    return torch.stack([src, dst], dim=0)


def test_mu_is_deterministic_sigmoid_of_logits():
    torch.manual_seed(0)
    logits = torch.randn(20, 1)
    mu, _ = sample_ordered_concrete_mask(logits, uniform_noise=torch.full((20,), 0.5))
    assert torch.allclose(mu, torch.sigmoid(logits.squeeze(-1)))


def test_shapes_are_squeezed_to_1d():
    logits = torch.randn(15, 1)
    mu, edge_mask = sample_ordered_concrete_mask(logits)
    assert mu.shape == (15,)
    assert edge_mask.shape == (15,)


def test_mask_can_be_asymmetric_across_reverse_pairs():
    """The defining difference from the symmetric/corrected mask: (i,j) and
    (j,i) may sample to different keep/drop outcomes, since no symmetrization
    or shared-noise mechanism is applied."""
    edge_index = fully_connected_edge_index(4)
    rev_idx = compute_reverse_index(edge_index)
    num_edges = edge_index.size(1)

    torch.manual_seed(0)
    logits = torch.randn(num_edges, 1) * 3.0  # large magnitude, easy to separate
    uniform_noise = torch.rand(num_edges)  # independent draw per directed edge

    _, edge_mask = sample_ordered_concrete_mask(logits, uniform_noise=uniform_noise)

    non_self_loop = edge_index[0] != edge_index[1]
    max_asym = (edge_mask - edge_mask[rev_idx])[non_self_loop].abs().max().item()
    assert max_asym > 1e-6  # unlike the symmetric mask, this is NOT ~0


def test_same_uniform_noise_gives_reproducible_mask():
    logits = torch.randn(10, 1)
    noise = torch.rand(10)
    _, mask_a = sample_ordered_concrete_mask(logits, uniform_noise=noise.clone())
    _, mask_b = sample_ordered_concrete_mask(logits, uniform_noise=noise.clone())
    assert torch.equal(mask_a, mask_b)


def test_uniform_noise_is_clamped_away_from_zero_and_one():
    """uniform_noise=0 or 1 would send the logistic-noise term to +/-inf via
    log(0); the function must clamp to [eps, 1-eps] instead of propagating
    that into a NaN/inf mask."""
    logits = torch.zeros(4, 1)
    noise = torch.tensor([0.0, 1.0, 0.5, 1.0])
    mu, edge_mask = sample_ordered_concrete_mask(logits, uniform_noise=noise, eps=1e-4)
    assert torch.isfinite(edge_mask).all()
    assert torch.isfinite(mu).all()


def test_default_noise_is_sampled_when_not_provided():
    # logits generated unseeded, deliberately outside the seeded region below
    # -- seeding here first would let torch.randn(50, 1) consume part of the
    # stream, so the second manual_seed(0) would NOT realign with the first
    # sample_ordered_concrete_mask call's draw. Seed immediately before each
    # call instead.
    logits = torch.randn(50, 1)
    torch.manual_seed(0)
    _, mask_a = sample_ordered_concrete_mask(logits)
    torch.manual_seed(0)
    _, mask_b = sample_ordered_concrete_mask(logits)
    # same seed, same draw -- reproducible even without an explicit uniform_noise
    assert torch.equal(mask_a, mask_b)


def test_lower_temperature_sharpens_mask_toward_mu_extremes():
    """As temperature shrinks, sigmoid((logits+noise)/T) becomes closer to a
    hard step at zero -- pushing samples toward 0/1 for logits far from the
    boundary, compared to a higher temperature's smoother relaxation."""
    logits = torch.full((200,), 2.0).unsqueeze(-1)
    torch.manual_seed(0)
    noise = torch.rand(200)

    _, mask_high_temp = sample_ordered_concrete_mask(logits, temperature=2.0, uniform_noise=noise.clone())
    _, mask_low_temp = sample_ordered_concrete_mask(logits, temperature=0.1, uniform_noise=noise.clone())

    # low-temperature samples should be closer to {0, 1} on average
    dist_high = (mask_high_temp - mask_high_temp.round()).abs().mean()
    dist_low = (mask_low_temp - mask_low_temp.round()).abs().mean()
    assert dist_low < dist_high
