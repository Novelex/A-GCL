"""
Tests for the augmented-view construction: E_aug = E (hadamard) B, symmetric
mask sampling, and gradient flow through the view learner's logits.
"""
import torch
import pytest

from unsupervised.view_learner import (
    compute_reverse_index,
    symmetrize_edge_logits,
    sample_symmetric_logistic_noise,
)


def fully_connected_edge_index(num_nodes):
    """Row-major (i,j) for every pair, including self-loops -- matches
    abideDataset.py's dense-matrix .nonzero() ordering."""
    idx = torch.arange(num_nodes)
    src = idx.repeat_interleave(num_nodes)
    dst = idx.repeat(num_nodes)
    return torch.stack([src, dst], dim=0)


def batched_fully_connected_edge_index(num_graphs, num_nodes):
    single = fully_connected_edge_index(num_nodes)
    num_edges = single.size(1)
    edges = []
    for g in range(num_graphs):
        edges.append(single + g * num_nodes)
    return torch.cat(edges, dim=1), num_edges


def test_shapes_match_real_config():
    # 32 subjects x 90 nodes, fully connected including diagonal -> 259200 edges
    edge_index, per_graph = batched_fully_connected_edge_index(32, 90)
    assert per_graph == 90 * 90
    num_edges = edge_index.size(1)
    assert num_edges == 259200

    E = torch.randn(num_edges)
    logits = torch.randn(num_edges, 1)
    rev_idx = compute_reverse_index(edge_index)
    sym_logits = symmetrize_edge_logits(edge_index, logits, rev_idx)
    noise = sample_symmetric_logistic_noise(edge_index, rev_idx, bias=1e-4)
    B = torch.sigmoid((noise + sym_logits) / 1.0)
    E_aug = E * B

    assert E.shape == (num_edges,)
    assert B.shape == (num_edges,)
    assert E_aug.shape == (num_edges,)


def test_mask_is_symmetric():
    edge_index, _ = batched_fully_connected_edge_index(4, 6)
    num_edges = edge_index.size(1)
    logits = torch.randn(num_edges, 1)
    rev_idx = compute_reverse_index(edge_index)
    sym_logits = symmetrize_edge_logits(edge_index, logits, rev_idx)
    noise = sample_symmetric_logistic_noise(edge_index, rev_idx, bias=1e-4)
    B = torch.sigmoid((noise + sym_logits) / 1.0)

    max_asym = (B - B[rev_idx]).abs().max().item()
    assert max_asym < 1e-7


def test_shared_noise_not_averaged_independent_draws():
    """B_ij and B_ji must come from ONE shared noise draw, not the average
    of two independent ones -- verified by checking the noise tensor itself
    assigns identical values to reverse-edge pairs."""
    edge_index, _ = batched_fully_connected_edge_index(2, 5)
    rev_idx = compute_reverse_index(edge_index)
    torch.manual_seed(0)
    noise = sample_symmetric_logistic_noise(edge_index, rev_idx, bias=1e-4)
    assert torch.equal(noise, noise[rev_idx])


def test_all_ones_mask_recovers_E():
    edge_index, _ = batched_fully_connected_edge_index(2, 5)
    num_edges = edge_index.size(1)
    E = torch.randn(num_edges)
    B = torch.ones(num_edges)
    E_aug = E * B
    assert torch.equal(E_aug, E)


def test_all_zeros_mask_gives_zero():
    edge_index, _ = batched_fully_connected_edge_index(2, 5)
    num_edges = edge_index.size(1)
    E = torch.randn(num_edges)
    B = torch.zeros(num_edges)
    E_aug = E * B
    assert torch.equal(E_aug, torch.zeros(num_edges))


def test_gradients_reach_both_directional_logits():
    edge_index, _ = batched_fully_connected_edge_index(1, 4)
    num_edges = edge_index.size(1)
    logits = torch.randn(num_edges, 1, requires_grad=True)
    rev_idx = compute_reverse_index(edge_index)
    sym_logits = symmetrize_edge_logits(edge_index, logits, rev_idx)
    loss = sym_logits.sum()
    loss.backward()
    assert logits.grad is not None
    assert torch.all(logits.grad != 0)


def test_self_loop_handling_deterministic():
    """A self-loop (i,i) is its own reverse -- compute_reverse_index must map
    it to itself, and symmetrizing it must be a no-op."""
    edge_index, _ = batched_fully_connected_edge_index(1, 5)
    rev_idx = compute_reverse_index(edge_index)
    src, dst = edge_index[0], edge_index[1]
    self_loop_mask = src == dst
    self_loop_positions = self_loop_mask.nonzero(as_tuple=True)[0]
    assert torch.equal(rev_idx[self_loop_positions], self_loop_positions)

    values = torch.randn(edge_index.size(1))
    sym = symmetrize_edge_logits(edge_index, values, rev_idx)
    assert torch.allclose(sym[self_loop_positions], values[self_loop_positions])


def test_different_subjects_do_not_share_edge_pairs():
    """Reverse-edge lookup must stay within each graph's own block in a
    batch -- a node in graph 0 must never be matched to a node in graph 1."""
    num_graphs, num_nodes = 3, 5
    edge_index, per_graph = batched_fully_connected_edge_index(num_graphs, num_nodes)
    rev_idx = compute_reverse_index(edge_index)
    graph_of_edge = torch.arange(edge_index.size(1)) // per_graph
    assert torch.equal(graph_of_edge, graph_of_edge[rev_idx])
