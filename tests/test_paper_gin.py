"""
Tests for the paper-literal GIN ablation flags (message_relu, normalize_nodes,
post_bn_relu -- see docs/changes.md "deliberately left unfixed") on WGINConv
and TUEncoder. These flags already exist as ordinary constructor args; the
paper_exact profile just forces them to their paper-literal values via
PAPER_EXACT_OVERRIDES (message_relu=False, normalize_nodes=False,
post_bn_relu=False). Nothing here changes the corrected defaults (all True).
"""
import torch
import pytest

from unsupervised.convs.wgin_conv import WGINConv
from unsupervised.encoder.tu_encoder import TUEncoder


def test_paper_self_signal_is_counted_twice():
    """With a self-loop present (as abideDataset.py's dense M^2 construction
    with a nonzero fc diagonal produces) and eps=0, a node's own signal
    enters the WGIN update twice: once through the self-loop message
    (weighted by the diagonal edge weight) and once through the root term
    (1+eps)*x_r. With message_relu=False and edge_weight=1 on the single
    self-loop, propagate()+root should sum to exactly 2*x -- the paper's
    literal formula has no mechanism to avoid this double-count."""
    conv = WGINConv(torch.nn.Identity(), message_relu=False)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    edge_index = torch.tensor([[0], [0]])
    edge_weight = torch.tensor([1.0])

    out = conv(x, edge_index, edge_weight)
    assert torch.allclose(out, 2.0 * x)


def test_message_relu_false_does_not_clip_negative_messages():
    """paper-literal: the weighted neighbour representation is used
    directly, with no ReLU inside message() -- a negative edge weight (a
    real, meaningful negative correlation in FC data) must be allowed to
    produce a negative message, not get clipped to zero."""
    conv = WGINConv(torch.nn.Identity(), message_relu=False)
    x_j = torch.tensor([[1.0, -2.0]])
    edge_weight = torch.tensor([-1.0])  # negative correlation
    message = conv.message(x_j, edge_weight)
    assert (message < 0).any()


def test_message_relu_true_clips_negative_activations_before_weighting():
    """corrected/current code: ReLU is applied to x_j before the edge-weight
    multiply -- a negative edge weight can still flip sign, but the
    activation itself is clipped to >= 0 first."""
    conv = WGINConv(torch.nn.Identity(), message_relu=True)
    x_j = torch.tensor([[1.0, -2.0]])
    edge_weight = torch.tensor([1.0])
    message = conv.message(x_j, edge_weight)
    assert torch.equal(message, torch.tensor([[1.0, 0.0]]))


def test_normalize_nodes_false_skips_l2_normalization():
    """paper-literal: GIN outputs are summed/pooled directly, no
    F.normalize before pooling -- so node embeddings need not have unit
    norm, unlike the corrected default."""
    torch.manual_seed(0)
    encoder = TUEncoder(num_dataset_features=3, emb_dim=8, num_gc_layers=2, normalize_nodes=False)
    encoder.eval()
    x = torch.rand(5, 3)
    idx = torch.arange(5)
    src, dst = idx.repeat_interleave(5), idx.repeat(5)
    edge_index = torch.stack([src, dst], dim=0)
    edge_weight = torch.rand(edge_index.size(1))

    batch = torch.zeros(5, dtype=torch.long)
    _, node_emb = encoder(batch, x, edge_index, None, edge_weight)

    norms = node_emb.norm(dim=1)
    assert not torch.allclose(norms, torch.ones_like(norms))


def test_normalize_nodes_true_produces_unit_norm_node_embeddings():
    torch.manual_seed(0)
    encoder = TUEncoder(num_dataset_features=3, emb_dim=8, num_gc_layers=2, normalize_nodes=True)
    encoder.eval()
    x = torch.rand(5, 3)
    idx = torch.arange(5)
    src, dst = idx.repeat_interleave(5), idx.repeat(5)
    edge_index = torch.stack([src, dst], dim=0)
    edge_weight = torch.rand(edge_index.size(1))

    batch = torch.zeros(5, dtype=torch.long)
    _, node_emb = encoder(batch, x, edge_index, None, edge_weight)

    norms = node_emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
