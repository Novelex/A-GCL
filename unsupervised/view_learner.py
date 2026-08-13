import torch
from torch.nn import Sequential, Linear, ReLU


def compute_reverse_index(edge_index):
	"""For each directed edge (i,j), find the position of its reverse (j,i)
	in edge_index. Self-loops map to themselves."""
	src, dst = edge_index[0], edge_index[1]
	num_nodes = int(edge_index.max()) + 1

	key = src * num_nodes + dst
	rev_key = dst * num_nodes + src

	sorted_key, sort_idx = torch.sort(key)
	pos = torch.searchsorted(sorted_key, rev_key).clamp(max=sorted_key.numel() - 1)
	rev_idx = sort_idx[pos]
	rev_idx = torch.where(key[rev_idx] == rev_key, rev_idx, torch.arange(edge_index.size(1), device=edge_index.device))
	return rev_idx


def symmetrize_edge_logits(edge_index, values, rev_idx=None):
	"""Average each directed edge (i,j) with its reverse (j,i) so that
	values derived from them (e.g. keep-probability) are symmetric,
	regardless of the order edges appear in edge_index."""
	values = values.squeeze(-1) if values.dim() > 1 else values
	if rev_idx is None:
		rev_idx = compute_reverse_index(edge_index)
	return (values + values[rev_idx]) / 2


def sample_symmetric_logistic_noise(edge_index, rev_idx=None, bias=1e-4, device=None):
	"""Draw one Logistic-Concrete noise value per undirected edge pair and
	broadcast it to both directions, instead of drawing independently per
	directed edge (which would make (i,j) and (j,i) sample differently)."""
	if rev_idx is None:
		rev_idx = compute_reverse_index(edge_index)
	num_edges = edge_index.size(1)
	position = torch.arange(num_edges, device=edge_index.device)
	canonical = torch.minimum(position, rev_idx)

	eps = (bias - (1 - bias)) * torch.rand(num_edges, device=device) + (1 - bias)
	eps = eps[canonical]
	noise = torch.log(eps) - torch.log(1 - eps)
	return noise


def sample_ordered_concrete_mask(
	edge_logits,
	temperature=1.0,
	uniform_noise=None,
	eps=1e-4,
):
	"""paper_exact profile: literal Binary-Concrete relaxation applied directly
	per directed edge, with independent noise per direction -- no symmetrization,
	no shared noise across (i,j)/(j,i). Kept alongside, not in place of, the
	corrected profile's symmetric mask (symmetrize_edge_logits +
	sample_symmetric_logistic_noise)."""
	logits = edge_logits.squeeze(-1)

	if uniform_noise is None:
		uniform_noise = torch.empty_like(logits).uniform_(eps, 1.0 - eps)
	else:
		uniform_noise = uniform_noise.to(
			device=logits.device,
			dtype=logits.dtype,
		)
		uniform_noise = uniform_noise.clamp(eps, 1.0 - eps)

	logistic_noise = (
		torch.log(uniform_noise)
		- torch.log1p(-uniform_noise)
	)

	# Deterministic Bernoulli keep probability
	mu = torch.sigmoid(logits)

	# Relaxed sampled mask B
	edge_mask = torch.sigmoid(
		(logits + logistic_noise) / temperature
	)

	return mu, edge_mask


class ViewLearner(torch.nn.Module):
	def __init__(self, encoder, mlp_edge_model_dim=64):
		super(ViewLearner, self).__init__()

		self.encoder = encoder
		self.input_dim = self.encoder.out_node_dim

		self.mlp_edge_model = Sequential(
			Linear(self.input_dim * 2, mlp_edge_model_dim),
			ReLU(),
			Linear(mlp_edge_model_dim, 1)
		)
		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, batch, x, edge_index, edge_attr, edge_weight):

		_, node_emb = self.encoder(batch, x, edge_index, edge_attr, edge_weight)

		src, dst = edge_index[0], edge_index[1]
		emb_src = node_emb[src]
		emb_dst = node_emb[dst]

		edge_emb = torch.cat([emb_src, emb_dst], 1)
		edge_logits = self.mlp_edge_model(edge_emb)

		return edge_logits