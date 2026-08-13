"""
Tests for GInfoMinMax.encode()/get_embeddings() -- the fix that makes
downstream evaluation actually use the projected representation z (or h
as an explicit ablation) instead of silently bypassing the projection head.
"""
import torch
import pytest
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from unsupervised.encoder.tu_encoder import TUEncoder
from unsupervised.learning.ginfominmax import GInfoMinMax


def make_loader(num_graphs=6, num_nodes=10, num_features=3):
    graphs = []
    for _ in range(num_graphs):
        x = torch.rand(num_nodes, num_features)
        idx = torch.arange(num_nodes)
        src = idx.repeat_interleave(num_nodes)
        dst = idx.repeat(num_nodes)
        edge_index = torch.stack([src, dst], dim=0)
        edge_weight = torch.rand(edge_index.size(1))
        y = torch.tensor([0])
        graphs.append(Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y))
    return DataLoader(graphs, batch_size=3)


def make_model(emb_dim=16, proj_hidden_dim=16):
    encoder = TUEncoder(num_dataset_features=3, emb_dim=emb_dim, num_gc_layers=2)
    return GInfoMinMax(encoder, proj_hidden_dim=proj_hidden_dim)


def test_h_and_z_shapes():
    model = make_model(emb_dim=16, proj_hidden_dim=16)
    loader = make_loader(num_graphs=6)
    device = torch.device('cpu')

    h_emb, h_y = model.get_embeddings(loader, device, representation='h')
    z_emb, z_y = model.get_embeddings(loader, device, representation='z')

    assert h_emb.shape == (6, 16)
    assert z_emb.shape == (6, 16)
    assert h_y.shape == (6,)
    assert z_y.shape == (6,)


def test_h_and_z_not_numerically_identical():
    model = make_model()
    loader = make_loader(num_graphs=4)
    device = torch.device('cpu')

    h_emb, _ = model.get_embeddings(loader, device, representation='h')
    z_emb, _ = model.get_embeddings(loader, device, representation='z')

    assert not (h_emb == z_emb).all()


def test_projection_head_parameters_affect_z_not_h():
    model = make_model()
    loader = make_loader(num_graphs=4)
    device = torch.device('cpu')

    h_before, _ = model.get_embeddings(loader, device, representation='h')
    z_before, _ = model.get_embeddings(loader, device, representation='z')

    # perturb only the projection head's weights
    with torch.no_grad():
        for p in model.proj_head.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    h_after, _ = model.get_embeddings(loader, device, representation='h')
    z_after, _ = model.get_embeddings(loader, device, representation='z')

    assert (h_before == h_after).all()  # encoder untouched -> h unchanged
    assert not (z_before == z_after).all()  # proj_head changed -> z must change


def test_invalid_representation_raises():
    model = make_model()
    loader = make_loader(num_graphs=2)
    with pytest.raises(AssertionError):
        model.get_embeddings(loader, torch.device('cpu'), representation='bogus')
