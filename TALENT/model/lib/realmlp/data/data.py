import math
from typing import Optional, Union, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from TALENT.model.lib.realmlp import utils
from TALENT.model.lib.realmlp.torch_utils import seeded_randperm, batch_randperm


class TaskType:
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


# todo: add info which values might be missing?
# todo: use np arrays instead of torch.Tensor? need to convert back a lot of .item()...
class TensorInfo:
    def __init__(self, feat_shape: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
                 cat_sizes: Optional[Union[List, np.ndarray, torch.Tensor]] = None):
        self.feat_shape = feat_shape
        self.cat_sizes = cat_sizes
        if isinstance(self.feat_shape, torch.Tensor):
            self.feat_shape = self.feat_shape.detach().cpu().numpy()

    def get_feat_shape(self) -> np.ndarray:
        if self.feat_shape is None and self.cat_sizes is not None:
            self.feat_shape = np.asarray(self.cat_sizes).shape
        return np.asarray(self.feat_shape)
        # return torch.as_tensor(self.feat_shape)

    def get_cat_sizes(self) -> torch.Tensor:
        if self.cat_sizes is None and self.feat_shape is not None:
            self.cat_sizes = torch.zeros(*self.feat_shape, dtype=torch.long)
        return torch.as_tensor(self.cat_sizes)

    def get_n_features(self) -> int:
        return np.prod(self.get_feat_shape())

    def get_cat_size_product(self) -> int:
        return torch.prod(self.get_cat_sizes()).item()

    def is_empty(self) -> bool:
        return self.get_n_features() == 0

    def is_cont(self) -> bool:
        return self.cat_sizes is None or len(self.cat_sizes) == 0 or self.cat_sizes[
            0] == 0  # todo: might not work for multi-dimensional tensors

    def is_cat(self) -> bool:
        return not self.is_cont()

    def to_dict(self) -> Dict:
        return {'feat_shape': self.feat_shape, 'cat_sizes': self.cat_sizes}

    @staticmethod
    def from_dict(data: Dict) -> 'TensorInfo':
        return TensorInfo(data['feat_shape'], data['cat_sizes'])

    @staticmethod
    def concat(tensor_infos: List['TensorInfo']) -> 'TensorInfo':
        """
        Create the TensorInfo that corresponds to concatenating the tensors.
        :param tensor_infos:
        :return:
        """
        assert len(tensor_infos) > 0
        if tensor_infos[0].is_cat():
            return TensorInfo(cat_sizes=torch.cat([ti.get_cat_sizes() for ti in tensor_infos], dim=0))
        else:
            return TensorInfo(feat_shape=sum([ti.get_feat_shape() for ti in tensor_infos]))


class DictDataset:
    # todo: add conversion methods to/from pandas dataframe?
    #  also to/from numpy/torch tensors?
    def __init__(self, tensors: Optional[Dict[str, torch.Tensor]], tensor_infos: Dict[str, TensorInfo],
                 device: Optional[Union[str, torch.device]] = None,
                 n_samples: Optional[int] = None):
        """
        :param tensors: Can be None, but then device and n_samples must be specified.
        :param tensor_infos: Information (shape, category sizes) for each tensor.
        :param device: Device that tensors is on. If tensors is specified, this will be computed automatically.
        :param n_samples: Number of samples. If tensors is specified, this will be computed automatically.
        """
        self.device = device if device is not None else next(iter(tensors.values())).device
        self.n_samples = n_samples if n_samples is not None else next(iter(tensors.values())).shape[0]
        self.tensors = None if tensors is None else {key: t.to(device) for key, t in tensors.items()}
        self.tensor_infos = tensor_infos

    def split_xy(self) -> Tuple['DictDataset', 'DictDataset']:
        y_keys = [key for key in self.tensors if key.startswith('y')]
        x_keys = [key for key in self.tensors if key not in y_keys]
        return self[x_keys], self[y_keys]

    def without_labels(self) -> 'DictDataset':
        return self.split_xy()[0]

    def to_df(self) -> pd.DataFrame:
        tensor_dfs = []
        for key in self.tensors:
            val_np = self.tensors[key].detach().cpu().numpy()
            col_names = [f'{key}_{i}' for i in range(val_np.shape[1])]
            df = pd.DataFrame(val_np, columns=col_names)
            if self.tensor_infos[key].is_cat():
                df = df.astype('category')
            tensor_dfs.append(df)

        return pd.concat(tensor_dfs, axis=1)

    def get_batch(self, idxs) -> Dict[str, torch.Tensor]:
        return {key: t[idxs, :] for key, t in self.tensors.items()}

    def get_sub_dataset(self, idxs) -> 'DictDataset':
        return DictDataset(self.get_batch(idxs), self.tensor_infos, device=self.device)

    def get_shuffled(self, seed) -> 'DictDataset':
        return self.get_sub_dataset(seeded_randperm(self.n_samples, self.device, seed))

    def get_size_gb(self) -> float:
        """
        :return: RAM usage in Gigabytes
        """
        return self.n_samples * sum([ti.get_n_features() * (8 if ti.is_cat() else 4)
                                     for ti in self.tensor_infos.values()]) / (1024 ** 3)

    @staticmethod
    def join(*datasets):
        return DictDataset(utils.join_dicts(*[ds.tensors for ds in datasets]),
                           utils.join_dicts(*[ds.tensor_infos for ds in datasets]))

    def to(self, device):
        return DictDataset(self.tensors, self.tensor_infos, device=device)

    def __getitem__(self, key):
        if isinstance(key, list):
            return DictDataset({k: self.tensors[k] for k in key}, {k: self.tensor_infos[k] for k in key},
                               device=self.device, n_samples=self.n_samples)
        return DictDataset({key: self.tensors[key]}, {key: self.tensor_infos[key]},
                           device=self.device, n_samples=self.n_samples)

    def get_n_classes(self):
        """
        :return: Returns the number of classes, given by the category size of the first feature of the y tensor.
        This only makes sense if there is a y tensor, and it does not check if y has more than one feature.
        """
        return self.tensor_infos['y'].get_cat_sizes()[0].item()


class ParallelDictDataLoader:
    def __init__(self, ds: DictDataset, idxs: torch.Tensor, batch_size: int, shuffle: bool = False,
                 adjust_bs: bool = False, drop_last: bool = False,
                 output_device: Optional[Union[str, torch.device]] = None):
        """
        :param dataset: A TaskData instance
        :param batch_size: default batch size, might be automatically adjusted
        :param shuffle: whether the dataset should be shuffled before each epoch
        :param adjust_bs: whether the batch_size may be lowered
        so that the batches are of more equal size while keeping the number of batches the same
        :param drop_last: whether the last batch should be omitted if it is smaller than the other ones
        :param output_device: The device that the returned data should be on
        (if None, take the device where the data already is)
        """
        self.ds = ds
        self.idxs = idxs.to(ds.device)
        self.n_parallel = idxs.shape[0]
        self.n_samples = idxs.shape[1]
        self.output_device = ds.device if output_device is None else output_device
        self.adjust_bs = adjust_bs
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.specified_batch_size = batch_size
        self.batch_size = min(batch_size, self.n_samples)

        if self.drop_last:
            self.n_batches = math.floor(self.n_samples / self.batch_size)
            if adjust_bs:
                self.batch_size = math.floor(self.n_samples / self.n_batches)
            self.sep_idxs = [self.batch_size * i for i in range(self.n_batches + 1)]
        else:
            self.n_batches = math.ceil(self.n_samples / self.batch_size)
            if adjust_bs:
                self.batch_size = math.ceil(self.n_samples / self.n_batches)
            self.sep_idxs = [self.batch_size * i for i in range(self.n_batches)] + [self.n_samples]

    def get_num_samples(self):
        return self.n_samples

    def get_num_iterated_samples(self):
        if self.drop_last:
            return self.n_batches * self.batch_size
        return self.get_num_samples()

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            perms = batch_randperm(self.n_parallel, self.n_samples, device=self.ds.device)
            for start, stop in zip(self.sep_idxs[:-1], self.sep_idxs[1:]):
                batches = self.ds.get_batch(idxs=self.idxs.gather(1, perms[:, start:stop]))
                yield {key: t.to(self.output_device) for key, t in batches.items()}
        else:
            for start, stop in zip(self.sep_idxs[:-1], self.sep_idxs[1:]):
                batches = self.ds.get_batch(idxs=self.idxs[:, start:stop])
                yield {key: t.to(self.output_device) for key, t in batches.items()}


class ValDictDataLoader:
    def __init__(self, ds: DictDataset, val_idxs: torch.Tensor, val_batch_size=256):
        """
        Create a Prediction Dataloader from Dataset and validation indices
        """
        ds_x, ds_y = ds.split_xy()
        self.val_x_dl = ParallelDictDataLoader(ds_x, val_idxs, batch_size=val_batch_size)
        self.val_idxs = val_idxs
        self.val_y = ds_y.get_batch(val_idxs).get('y', None)
        self.n_samples = val_idxs.shape[1]

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        return self.val_x_dl.__iter__()
