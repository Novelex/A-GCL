import torch
from torch.nn import Sequential, Linear, ReLU


def symmetrize_edge_logits(edge_index, values):
	"""Average each directed edge (i,j) with its reverse (j,i) so that
	values derived from them (keep-probability, sampled mask) are symmetric,
	regardless of the order edges appear in edge_index."""
	values = values.squeeze(-1) if values.dim() > 1 else values
	src, dst = edge_index[0], edge_index[1]
	num_nodes = int(edge_index.max()) + 1

	key = src * num_nodes + dst
	rev_key = dst * num_nodes + src

	sorted_key, sort_idx = torch.sort(key)
	pos = torch.searchsorted(sorted_key, rev_key).clamp(max=sorted_key.numel() - 1)
	rev_idx = sort_idx[pos]

	matched = key[rev_idx] == rev_key
	sym_values = values.clone()
	sym_values[matched] = (values[matched] + values[rev_idx[matched]]) / 2
	return sym_values


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