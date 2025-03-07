import os
import os.path
import heapq
import glob
import gzip
import shutil
import timeit
from pathlib import Path
from typing import List, Tuple, Any, Dict, Union, Optional, Callable

import dill
import copy
import uuid
import multiprocessing
import time
import json

import msgpack
import msgpack_numpy as m
m.patch()

import yaml
from torch import multiprocessing as mp

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from sklearn.base import check_is_fitted
import numpy as np


def select_from_config(config: Dict, keys: List):
    selected = {}
    for key in keys:
        if key in config:
            selected[key] = config[key]
    return selected


def adapt_config(config, **kwargs):
    new_config = copy.deepcopy(config)
    for key, value in kwargs.items():
        new_config[key] = value
    return new_config


def existsDir(directory):
    if directory != '':
        if not os.path.exists(directory):
            return False
    return True


def existsFile(file_path):
    return os.path.isfile(file_path)


def ensureDir(file_path):
    directory = os.path.dirname(file_path)
    if directory != '':
        if not os.path.exists(directory):
            os.makedirs(directory)


def matchFiles(file_matcher):
    return glob.glob(file_matcher)


def newDirname(prefix):
    i = 0
    name = prefix
    if existsDir(prefix):
        while existsDir(prefix + "_" + str(i)):
            i += 1
        name = prefix + "_" + str(i)
    os.makedirs(name)
    return name


def getSubfolderNames(folder):
    return [os.path.basename(name)
            for name in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, name))]


def getSubfolders(folder):
    return [os.path.join(folder, name)
            for name in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, name))]


def writeToFile(filename, content):
    ensureDir(filename)
    file = open(filename, 'w')
    file.truncate()
    file.write(content)
    file.close()


def readFromFile(filename):
    if not os.path.isfile(filename):
        return ''

    file = open(filename, 'r')
    result = file.read()
    file.close()
    return result


def create_dir(path):
    os.makedirs(path)


def delete_file(path):
    os.remove(path)


def serialize(filename: Union[Path, str], obj: Any, compressed: bool = False, use_json: bool = False,
              use_yaml: bool = False, use_msgpack: bool = False):
    # json only works for nested dicts
    ensureDir(filename)
    if compressed:
        file = gzip.open(filename, 'wt' if (use_json or use_yaml) else 'wb', compresslevel=5)
    else:
        file = open(filename, 'w' if (use_json or use_yaml) else 'wb')
    # dill can dump lambdas, and dill also dumps the class and not only the contents
    if use_json:
        json.dump(obj, file)
    elif use_yaml:
        yaml.dump(obj, file, Dumper=Dumper)
    elif use_msgpack:
        msgpack.dump(obj, file)
    else:
        dill.dump(obj, file)
    file.close()


def deserialize(filename: Union[Path, str], compressed: bool = False, use_json: bool = False, use_yaml: bool = False,
                use_msgpack: bool = False):
    # json only works for nested dicts
    if compressed:
        file = gzip.open(filename, 'rt' if (use_json or use_yaml) else 'rb')
    else:
        file = open(filename, 'r' if (use_json or use_yaml) else 'rb')
    if use_json:
        result = json.load(file)
    elif use_yaml:
        result = yaml.load(file, Loader=Loader)
    elif use_msgpack:
        result = msgpack.load(file)
    else:
        result = dill.load(file)
    file.close()
    return result


def copyFile(src, dst):
    ensureDir(dst)
    shutil.copyfile(src, dst)


def nsmallest(n, inputList):
    return heapq.nsmallest(n, inputList)[-1]


def identity(x):
    return x


def set_none_except(lst, idxs):
    for i in range(len(lst)):
        if i not in idxs:
            lst[i] = None


def argsort(lst):
    # from https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
    return sorted(range(len(lst)), key=lst.__getitem__)


def join_dicts(*dicts):
    # Attention: arguments do not commute since later dicts can override entries from earlier dicts!
    result = copy.copy(dicts[0])
    for d in dicts[1:]:
        result.update(d)
    return result


def update_dict(d: dict, update: Optional[dict] = None, remove_keys: Optional[Union[Any, List[Any]]] = None):
    d = copy.copy(d)
    if update is not None:
        d.update(update)
    if remove_keys is not None:
        if isinstance(remove_keys, List):
            for key in remove_keys:
                if key in d:
                    d.pop(key)
        else:
            if remove_keys in d:
                d.pop(remove_keys)
    return d


def map_nested(obj: Union[List, Dict, Any], f: Callable, dim: int):
    """
    dim=0 will apply f to obj directly, dim=1 to all elements in obj, etc.
    """
    if dim <= 0:
        return f(obj)
    elif isinstance(obj, dict):
        return {key: map_nested(value, f, dim-1) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [map_nested(value, f, dim-1) for value in obj]


def select_nested(obj: Union[List, Dict], idx: Any, dim: int):
    return map_nested(obj, lambda x: x[idx], dim)


def shift_dim_nested(obj: Union[List, Dict], dim1: int, dim2: int):
    # in a nested combination of lists and dicts, shift the indexing dimension dim1 to dim2
    # example: if d = {'a': [{'b': 1}, {'b': 2}]}, dim1 = 1, dim2 = 2, then the result should be
    # {'a': {'b': [1, 2]}}

    if dim1 < 0 or dim2 < 0:
        raise ValueError(f'expected dim1 >= 0 and dim2 >= 0, but got {dim1=} and {dim2=}')
    # if dim2 <= dim1:
    #     raise ValueError(f'expected dim2 > dim1, but got {dim1=} and {dim2=}')

    if dim1 > 0 and dim2 > 0:
        if isinstance(obj, dict):
            return {key: shift_dim_nested(value, dim1-1, dim2-1) for key, value in obj.items()}
        else:
            # assume that value is a list
            return [shift_dim_nested(value, dim1-1, dim2-1) for value in obj]
    elif dim1 > 1:
        # dim1 > dim2, shift backwards
        return shift_dim_nested(shift_dim_nested(obj, dim1, dim1 - 1), dim1 - 1, dim2)
    elif dim2 > 1:
        # dim2 > dim1, shift forwards
        return shift_dim_nested(shift_dim_nested(obj, dim1, dim1 + 1), dim1 + 1, dim2)
    else:
        # switch dimensions 0 and 1
        if isinstance(obj, dict):
            first = next(iter(obj.values()))
            if isinstance(first, dict):
                # swap two dicts
                return {key2: {key1: obj[key1][key2] for key1 in obj} for key2 in first}
            else:
                # assume it is a list
                return [{key1: obj[key1][i] for key1 in obj} for i in range(len(first))]
        else:
            first = obj[0]
            if isinstance(first, dict):
                return {key2: [obj[i][key2] for i in range(len(obj))] for key2 in first}
            else:
                # assume it is a list
                return [[obj[i][j] for i in range(len(obj))] for j in range(len(first))]
            pass
        pass


def pretty_table_str(str_table):
    if len(str_table) == 0:
        return ''
    max_lens = [np.max([len(row[i]) for row in str_table]) for i in range(len(str_table[0]))]
    whole_str = ''
    for row in str_table:
        for i, entry in enumerate(row):
            whole_str += entry + (' ' * (max_lens[i] - len(entry)))
        whole_str += '\n'
    return whole_str[:-1]  # remove last newline


def get_uuid_str():
    pid_str = str(multiprocessing.current_process().pid)
    time_str = str(time.time_ns())
    rand_str = str(uuid.UUID(bytes=os.urandom(16), version=4))
    return '_'.join([time_str, pid_str, rand_str])


def get_batch_intervals(n_total: int, batch_size: int) -> List[Tuple[int, int]]:
    boundaries = [i * batch_size for i in range(1 + n_total // batch_size)]
    if boundaries[-1] != n_total:
        boundaries.append(n_total)
    return [(start, stop) for start, stop in zip(boundaries[:-1], boundaries[1:])]


def all_equal(lst: List):
    # see https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    return not lst or [lst[0]]*len(lst) == lst


class Timer:
    def __init__(self):
        self.start_time_total = None
        self.start_time_process = None
        self.acc_time_total = 0.0
        self.acc_time_process = 0.0

    def start(self):
        if self.start_time_total is None or self.start_time_process is None:
            self.start_time_total = timeit.default_timer()
            self.start_time_process = time.process_time()

    def pause(self):
        if self.start_time_total is None or self.start_time_process is None:
            return  # has already been paused or not been started
        self.acc_time_total += timeit.default_timer() - self.start_time_total
        self.acc_time_process += time.process_time() - self.start_time_process
        self.start_time_total = None
        self.start_time_process = None

    def get_result_dict(self):
        return {'total': self.acc_time_total, 'process': self.acc_time_process}


class TimePrinter:
    def __init__(self, desc: str):
        self.desc = desc
        self.timer = Timer()

    def __enter__(self):
        self.timer.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.pause()
        print(f'Time for {self.desc}: {self.timer.get_result_dict()["total"]:g}s')


def extract_params(config: Dict[str, Any],
                   param_configs: List[Union[Tuple[str, Optional[Union[str, List[str]]]],
                                             Tuple[str, Optional[Union[str, List[str]]], Any]]]) -> Dict[str, Any]:
    """
    Convert parameters in config to correct parameter names for another method and (optionally) insert default values
    :param config: Dictionary with values for parameters
    :param param_configs: Tuples specifying parameter names, e.g.:
    ('eta', None) specifies that result['eta'] = config['eta'] should be set if 'eta' is in config
    ('eta', 'lr') specifies that result['eta'] = config['lr'] should be set if 'lr' is in config
    ('eta, ['eta', 'lr']) specifies that either config['eta'] or config['lr'] should be used, if available
    A third value in the tuple specifies a default value that should be used if no value is available in config.
    :return: A dictionary as specified above.
    """
    result = {}
    for param_config in param_configs:
        target_name = param_config[0]
        source_names = param_config[1]
        if source_names is None:
            source_names = [target_name]
        elif isinstance(source_names, str):
            source_names = [source_names]

        for source_name in source_names:
            if source_name in config:
                result[target_name] = config[source_name]
                break
        else:
            # if break is not used in the loop
            if len(param_config) >= 3:
                # default value specified
                result[target_name] = param_config[2]  # use the default value
    return result


def reverse_argmin(x: Union[List, np.ndarray]):
    """
    Does the same as np.argmin but in case of equality selects the last best one
    :param x: list or array of numbers
    :return: index of last minimum
    """
    if isinstance(x, list):
        x = np.asarray(x)
    assert(len(x.shape) == 1)
    return len(x) - 1 - int(np.argmin(x[::-1]))


def combine_seeds(seed_1: int, seed_2: int) -> int:
    """
    Combines two random seeds to a new seed in a hopefully "typically injective" way
    :param seed_1: First random seed.
    :param seed_2: Second random seed.
    :return: Another random seed
    """
    generator = np.random.default_rng(seed=seed_1)
    return int(generator.integers(low=0, high=2**24) + seed_2)


class ProcessPoolMapper:
    def __init__(self, n_processes: int, chunksize=1):
        self.n_processes = n_processes
        self.chunksize = chunksize
        pass

    def _apply(self, f_and_args_serialized: str) -> str:
        f, args = dill.loads(f_and_args_serialized)
        return dill.dumps(f(*args))

    def map(self, f, args_tuples: List[Tuple]) -> Any:
        if self.n_processes == 1:
            return [f(*args) for args in args_tuples]

        mp_ctx = mp.get_context('spawn')
        pool = mp_ctx.Pool(self.n_processes)
        serialized_args = [dill.dumps(args) for args in args_tuples]

        results = pool.map(self._apply, serialized_args, chunksize=self.chunksize)
        pool.terminate()

        return [dill.loads(s) for s in results]


# adapted from https://github.com/yandex-research/tabular-dl-tabr/blob/75105013189c76bc4f247633c2fb856bc948e579/lib/data.py#L262
class TabrQuantileTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, noise=1e-3, random_state=None, n_quantiles=1000, subsample=1_000_000_000,
                 output_distribution="normal"):
        self.noise = noise
        self.random_state = random_state
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution

    def fit(self, X, y=None):
        # Calculate the number of quantiles based on data size
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)

        # Initialize QuantileTransformer
        normalizer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state
        )

        # Add noise if required
        X_modified = self._add_noise(X) if self.noise > 0 else X

        # Fit the normalizer
        normalizer.fit(X_modified)
        # show that it's fitted
        self.normalizer_ = normalizer

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return self.normalizer_.transform(X)

    def _add_noise(self, X):
        stds = np.std(X, axis=0, keepdims=True)
        noise_std = self.noise / np.maximum(stds, self.noise)
        rng = np.random.default_rng(self.random_state)
        return X + noise_std * rng.standard_normal(X.shape)
