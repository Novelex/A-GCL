import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from os import listdir
import numpy as np
import os.path as osp
import scipy.io as sio
import numpy as np
import scipy.sparse as sp


class ABIDEDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        self.num_tasks = 1
        self.task_type = 'classification'
        self.eval_metric = 'accuracy'

        super(ABIDEDataset, self).__init__(root,transform, pre_transform)
        path = osp.join(self.processed_dir, 'data.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root,'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return  'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def _load_class(self, adj_folder, nf_folder, label):
        path_adj = osp.join(self.raw_dir, adj_folder)
        path_nf = osp.join(self.raw_dir, nf_folder)

        adj_files = os.listdir(path_adj)

        data_list = []
        for file in adj_files:
            nf_file = file.replace('_adj.mat', '_nf.mat')
            if not osp.isfile(osp.join(path_nf, nf_file)):
                continue

            nf = sio.loadmat(osp.join(path_nf, nf_file))
            x = nf['norm_matrix']
            x = np.nan_to_num(x)
            x = torch.Tensor(x)
            # paper Sec. 2.1: node features min-max normalized to [0, 1]
            # across all 3 channels
            x_min, x_max = x.min(), x.max()
            if x_max > x_min:
                x = (x - x_min) / (x_max - x_min)

            adj = sio.loadmat(osp.join(path_adj, file))
            edge_index = adj['cropped_matrix']

            edge_index = np.nan_to_num(edge_index)
            edge_index_temp = sp.coo_matrix(edge_index)
            edge_weight = torch.Tensor(edge_index_temp.data)
            # paper Sec. 2.1: edge weights normalized to [-1, 1] by
            # dividing by the max absolute value
            edge_weight_max = edge_weight.abs().max()
            if edge_weight_max > 0:
                edge_weight = edge_weight / edge_weight_max

            edge_index = torch.Tensor(edge_index)
            edge_index = edge_index.nonzero(as_tuple=False).t().contiguous()

            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                        y=label)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        print(f"{adj_folder}: loaded {len(data_list)} / {len(adj_files)} adj files")
        assert len(data_list) == len(adj_files), (
            f"{len(adj_files) - len(data_list)} subjects in {adj_folder} have no "
            f"matching file in {nf_folder}"
        )

        return data_list

    def process(self):
        data_list_ASD = self._load_class('ASD_ADJ', 'ASD_NF', label=1)
        data_list_TC = self._load_class('NC_ADJ', 'NC_NF', label=0)

        data_list = data_list_ASD + data_list_TC

        torch.save(self.collate(data_list),
                osp.join(self.processed_dir, 'data.pt'))

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))