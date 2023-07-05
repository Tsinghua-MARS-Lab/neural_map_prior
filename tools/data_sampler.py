from copy import deepcopy

import torch
from torch.utils.data import DistributedSampler as _DistributedSampler


class GeoDistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 drop_last=True,
                 seed=0):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last)
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        gpu2sample_id = deepcopy(self.dataset.gpu2sample_id)

        if self.shuffle:
            for gpu_id, sample_id_list in gpu2sample_id.items():
                sample_id_list = torch.tensor(sample_id_list)
                g = torch.Generator()
                g.manual_seed(self.epoch + self.seed)
                gpu_indices = torch.randperm(len(sample_id_list), generator=g)
                sample_id_list = sample_id_list[gpu_indices].tolist()
                gpu2sample_id[gpu_id] = sample_id_list
            print('GeoDistributedSampler' * 3, 'shuffle' * 3)
        else:
            print('GeoDistributedSampler' * 3, 'no shuffle' * 3)

        indices = gpu2sample_id[self.rank]
        assert len(indices) == self.num_samples, (len(indices), self.num_samples)

        return iter(indices)
