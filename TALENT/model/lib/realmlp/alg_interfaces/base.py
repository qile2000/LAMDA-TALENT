from typing import Optional, List

import numpy as np
import torch


class SplitIdxs:
    """
    Represents multiple train-validation-test splits for AlgInterface.
    """
    def __init__(self, train_idxs: torch.Tensor, val_idxs: Optional[torch.Tensor], test_idxs: Optional[torch.Tensor],
                 split_seed: int, sub_split_seeds: List[int], split_id: int):
        # train_idxs: n_trainval_splits x n_train_idxs
        # val_idxs: n_trainval_splits x n_val_idxs (optional)
        # trainval_idxs: n_train_idxs + n_val_idxs
        # test_idxs: n_test_idxs (optional)
        # alg_seeds: n_trainval_splits
        # split_id is useful for logging
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.split_seed = split_seed
        self.sub_split_seeds = sub_split_seeds
        self.split_id = split_id
        self.n_trainval_splits = train_idxs.shape[0]
        self.n_train = train_idxs.shape[-1]
        self.n_val = 0 if val_idxs is None else val_idxs.shape[-1]
        self.n_test = 0 if test_idxs is None else test_idxs.shape[-1]
        if len(self.sub_split_seeds) != self.n_trainval_splits:
            raise ValueError('len(self.alg_seeds) != self.n_trainval_splits')
        if val_idxs is not None and val_idxs.shape[0] != self.n_trainval_splits:
            raise ValueError('val_idxs.shape[0] != self.n_trainval_splits')

    def get_sub_split_idxs(self, i: int) -> 'SubSplitIdxs':
        return SubSplitIdxs(self.train_idxs[i], self.val_idxs[i] if self.val_idxs is not None else None,
                            self.test_idxs, self.sub_split_seeds[i])

    def get_sub_split_idxs_alt(self, i: int) -> 'SplitIdxs':
        return SplitIdxs(self.train_idxs[i:i+1], self.val_idxs[i:i+1] if self.val_idxs is not None else None,
                            self.test_idxs, self.split_seed, self.sub_split_seeds[i:i+1], split_id=self.split_id)


class SubSplitIdxs:
    """
    Represents a single trainval-test split with multiple train-val splits
    """
    def __init__(self, train_idxs: torch.Tensor, val_idxs: Optional[torch.Tensor], test_idxs: Optional[torch.Tensor],
                 alg_seed: int):
        # train_idxs: n_train_idxs
        # val_idxs: n_val_idxs (optional)
        # test_idxs: n_test_idxs (optional)
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.alg_seed = alg_seed
        self.n_train = train_idxs.shape[-1]
        self.n_val = 0 if val_idxs is None else val_idxs.shape[-1]
        self.n_test = 0 if test_idxs is None else test_idxs.shape[-1]


class InterfaceResources:
    """
    Simple class representing resources that a method is allowed to use (number of threads and GPUs).
    """
    def __init__(self, n_threads: int, gpu_devices: List[str]):
        self.n_threads = n_threads
        self.gpu_devices = gpu_devices


class RequiredResources:
    """
    Represents estimated/requested resources by a method.
    """
    def __init__(self, time_s: float, n_threads: float, cpu_ram_gb: float, n_gpus: int = 0, gpu_usage: float = 1.0,
                 gpu_ram_gb: float = 0.0, n_explicit_physical_cores: int = 0):
        self.n_threads = n_threads
        self.cpu_ram_gb = cpu_ram_gb
        self.n_gpus = n_gpus
        self.gpu_usage = gpu_usage
        self.gpu_ram_gb = gpu_ram_gb
        self.time_s = time_s
        # for liquidSVM, want to have contiguous core indices
        self.n_explicit_physical_cores = n_explicit_physical_cores

    def get_resource_vector(self, fixed_resource_vector: np.ndarray):
        own_resources = np.asarray([self.n_threads, self.cpu_ram_gb, self.gpu_usage, self.gpu_ram_gb])
        if self.should_add_fixed_resources():
            # do not use fixed cpu ram since that is also measured for GPU usage
            own_resources += fixed_resource_vector
        multiplier = np.asarray([1.0, 1.0, self.n_gpus, self.n_gpus])
        return multiplier * own_resources

    def should_add_fixed_resources(self) -> bool:
        return self.n_gpus > 0

    @staticmethod
    def combine_sequential(resources_list: List['RequiredResources']):
        return RequiredResources(time_s=sum([r.time_s for r in resources_list]),
                                 n_threads=max([r.n_threads for r in resources_list]),
                                 cpu_ram_gb=max([r.cpu_ram_gb for r in resources_list]),
                                 n_gpus=max([r.n_gpus for r in resources_list]),
                                 gpu_usage=max([r.gpu_usage for r in resources_list]),
                                 gpu_ram_gb=max([r.gpu_ram_gb for r in resources_list]),
                                 n_explicit_physical_cores=max([r.n_explicit_physical_cores for r in resources_list]),
                                 )
