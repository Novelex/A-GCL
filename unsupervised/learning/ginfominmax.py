import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU


class GInfoMinMax(torch.nn.Module):
	def __init__(self, encoder, proj_hidden_dim=300):
		super(GInfoMinMax, self).__init__()

		self.encoder = encoder
		self.input_proj_dim = self.encoder.out_graph_dim

		self.proj_head = Sequential(Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
									   Linear(proj_hidden_dim, proj_hidden_dim))

		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def encode(self, batch, x, edge_index, edge_attr=None, edge_weight=None):
		h, node_emb = self.encoder(batch, x, edge_index, edge_attr, edge_weight)
		z = self.proj_head(h)
		return h, z, node_emb

	def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):

		_, z, node_emb = self.encode(batch, x, edge_index, edge_attr, edge_weight)

		# z shape -> Batch x proj_hidden_dim = 32*32
		return z, node_emb

	def get_embeddings(self, loader, device, representation='z', is_rand_label=False):
		assert representation in ('z', 'h')
		ret = []
		y = []
		self.eval()
		with torch.no_grad():
			for data in loader:
				if isinstance(data, list):
					data = data[0].to(device)
				data = data.to(device)
				batch, x, edge_index = data.batch, data.x, data.edge_index
				edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

				if x is None:
					x = torch.ones((batch.shape[0], 1)).to(device)

				h, z, _ = self.encode(batch, x, edge_index, None, edge_weight)
				selected = z if representation == 'z' else h

				ret.append(selected.cpu().numpy())
				if is_rand_label:
					y.append(data.rand_label.cpu().numpy())
				else:
					y.append(data.y.cpu().numpy())
		ret = np.concatenate(ret, 0)
		y = np.concatenate(y, 0)
		return ret, y

	@staticmethod
	def calc_loss( x, x_aug, temperature=0.2, sym=True):
		# x and x_aug shape -> Batch x proj_hidden_dim
		batch_size, _ = x.size()
		x_abs = x.norm(dim=1)
		x_aug_abs = x_aug.norm(dim=1)

		sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
		sim_matrix = torch.exp(sim_matrix / temperature)
		pos_sim = sim_matrix[range(batch_size), range(batch_size)]
		m1 = torch.einsum('ik,jk->ij', x, x_aug)
		m2 = torch.einsum('i,j->ij', x_abs, x_aug_abs)
		if sym:

			loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
			loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

			loss_0 = - torch.log(loss_0).mean()
			loss_1 = - torch.log(loss_1).mean()
			loss = (loss_0 + loss_1)/2.0
		else:
			loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
			loss_1 = - torch.log(loss_1).mean()
			return loss_1

		return loss


