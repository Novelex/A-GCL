import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from os import listdir
import numpy as np
import os.path as osp
import scipy.io as sio


class ABIDEDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        self.num_tasks = 1
        self.task_type = 'classification'
        self.eval_metric = 'accuracy'

        super(ABIDEDataset, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

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
        # v2 -> v3: node-feature normalization changed from joint (all 3
        # bands combined) to per-band min-max (layertesting/layer2) -- an
        # old v2 cache, built with the joint normalization, must not be
        # silently reloaded
        return 'data_dense_v3.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def _load_class(self, adj_folder, nf_folder, label):
        path_adj = osp.join(self.raw_dir, adj_folder)
        path_nf = osp.join(self.raw_dir, nf_folder)

        adj_files = sorted(os.listdir(path_adj))

        data_list = []
        for file in adj_files:
            nf_file = file.replace('_adj.mat', '_nf.mat')
            if not osp.isfile(osp.join(path_nf, nf_file)):
                continue

            nf = sio.loadmat(osp.join(path_nf, nf_file))
            x = nf['norm_matrix']
            x = np.nan_to_num(x)
            x = torch.Tensor(x)
            # per-band min-max to [0, 1], NOT joint/global across all 3 channels
            # combined -- verified on real data (layertesting/layer2) that a
            # joint min-max lets whichever band has the single largest raw
            # value (consistently slow-5) claim the full [0,1] range while the
            # other two bands get compressed into roughly half that range or
            # less, every subject, systematically. Harmless for old ALFF
            # (already z-scored per band upstream, so all 3 bands enter this
            # step on equal footing already) but destructive for new ALFF
            # (raw magnitude, no such pre-equalization). Per-band normalization
            # gives every band its own full [0,1] range regardless of its raw
            # units, fixing this for both sources.
            x_min = x.min(dim=0, keepdim=True).values
            x_max = x.max(dim=0, keepdim=True).values
            span = x_max - x_min
            x = torch.where(span > 0, (x - x_min) / span, x)

            adj = sio.loadmat(osp.join(path_adj, file))
            fc = adj['cropped_matrix']
            fc = np.nan_to_num(fc).astype(np.float32)

            if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
                raise ValueError(f"{file}: FC matrix must be square, got {fc.shape}")

            # paper Sec. 2.1: edge weights normalized to [-1, 1] by dividing
            # by the max absolute value
            max_abs = np.abs(fc).max()
            if max_abs > 0:
                fc = fc / max_abs

            # dense M x M construction, not .nonzero() -- the paper initializes
            # A with all 1's (a complete graph including the diagonal), so an
            # edge must not silently disappear just because its FC weight
            # happens to be exactly 0.0. row-major order matches what
            # .nonzero() produced anyway for our real (zero-free) data, so
            # this is a verified no-op here and a correctness generalization
            # for any future data that does contain exact zeros.
            num_nodes = fc.shape[0]
            nodes = torch.arange(num_nodes, dtype=torch.long)
            src = nodes.repeat_interleave(num_nodes)
            dst = nodes.repeat(num_nodes)
            edge_index = torch.stack([src, dst], dim=0)
            edge_weight = torch.from_numpy(fc.reshape(-1)).float()

            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                        y=label, num_nodes=num_nodes)

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

        # stable, globally-unique per-subject id, used by the memory bank to
        # exclude a subject's own (past-epoch) embedding from its own negatives
        for i, data in enumerate(data_list):
            data.subject_id = torch.tensor([i], dtype=torch.long)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))