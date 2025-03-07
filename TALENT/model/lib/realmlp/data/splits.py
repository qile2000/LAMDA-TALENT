import math
from typing import Tuple, List

import torch

from TALENT.model.lib.realmlp import utils
from TALENT.model.lib.realmlp.data.data import DictDataset
from TALENT.model.lib.realmlp.torch_utils import seeded_randperm


# splits should not reference tasks, since tasks should only be loaded in the respective processes in the DevicePool,
# while splits are loaded earlier

class Split:
    def __init__(self, ds: DictDataset, idxs: Tuple[torch.Tensor, torch.Tensor]):
        """
        :param ds: The dataset that is split into parts
        :param idxs: Tuple of Tensors containing indices of the different parts of ds
        """
        self.ds = ds
        self.idxs = idxs

    def get_sub_ds(self, i):
        return self.ds.get_sub_dataset(self.idxs[i])

    def get_sub_idxs(self, i):
        return self.idxs[i]


class Splitter:
    def get_idxs(self, ds: DictDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def split_ds(self, ds: DictDataset) -> Split:
        idxs = self.get_idxs(ds)
        return Split(ds, idxs)


class RandomSplitter(Splitter):
    def __init__(self, seed, first_fraction=0.8):
        self.seed = seed
        self.first_fraction = first_fraction

    def get_idxs(self, ds: DictDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        # use ceil such that e.g. in the case of 1 sample, the sample ends up in the training set.
        split_idx = int(math.ceil(self.first_fraction * ds.n_samples))
        perm = seeded_randperm(ds.n_samples, ds.device, self.seed)
        return perm[:split_idx], perm[split_idx:]


class IndexSplitter(Splitter):
    def __init__(self, index):
        self.index = index

    def get_idxs(self, ds: DictDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        idxs = torch.arange(ds.n_samples, device=ds.device, dtype=torch.long)
        return idxs[:self.index], idxs[self.index:]


class AllNothingSplitter(Splitter):
    def get_idxs(self, ds: DictDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        all = torch.arange(ds.n_samples, device=ds.device, dtype=torch.long)
        nothing = torch.zeros(0, device=ds.device, dtype=torch.long)
        return all, nothing

    def split_ds(self, ds: DictDataset) -> Split:
        idxs = self.get_idxs(ds)
        return Split(ds, idxs)


class MultiSplitter:
    def get_idxs(self, ds: DictDataset) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError()

    def split_ds(self, ds: DictDataset) -> List[Split]:
        idxs_list = self.get_idxs(ds)
        return [Split(ds, idxs) for idxs in idxs_list]


class KFoldSplitter(MultiSplitter):
    def __init__(self, k: int, seed: int, stratified=False):
        self.k = k
        self.seed = seed
        self.stratified = stratified

    def get_idxs(self, ds: DictDataset) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        idxs = seeded_randperm(ds.n_samples, device=ds.device, seed=self.seed)
        if self.stratified:
            # do it with random shuffling such that elements of the same class are still shuffled
            perm = torch.argsort(ds.tensors['y'][idxs, 0])
            idxs = idxs[perm]
        fold_len = (ds.n_samples // self.k) * self.k
        fold_idxs = [idxs[start:fold_len:self.k] for start in range(self.k)]
        rest_idxs = idxs[fold_len:]
        idxs_list = []
        for i in range(self.k):
            idxs_1 = torch.cat([fold_idxs[j] for j in range(self.k) if j != i] + [rest_idxs], dim=-1)
            idxs_list.append((idxs_1, fold_idxs[i]))
        return idxs_list


class SplitInfo:
    def __init__(self, splitter: Splitter, split_type: str, id: int, alg_seed: int):
        self.splitter = splitter
        self.split_type = split_type  # one of "random", "default"
        self.id = id
        self.alg_seed = alg_seed

    def get_sub_seed(self, split_idx: int, is_cv: bool):
        return utils.combine_seeds(self.alg_seed, 2 * split_idx + int(is_cv))
        # return self.alg_seed + 5000 * int(is_cv) + 10000 * split_idx

    def get_sub_splits(self, ds: DictDataset, n_splits: int, is_cv: bool) -> List[Split]:
        if not is_cv:
            split = AllNothingSplitter().split_ds(ds)
            return [split] * n_splits

        if n_splits <= 1:
            return [RandomSplitter(seed=self.alg_seed, first_fraction=0.75).split_ds(ds)]
        else:
            is_classification = ds.tensor_infos['y'].get_cat_sizes()[0].item() > 0
            return KFoldSplitter(n_splits, seed=self.alg_seed, stratified=is_classification).split_ds(ds)
