import dataclasses as dc
import typing as ty
from copy import deepcopy
from pathlib import Path
import os
import json
import numpy as np
import sklearn.preprocessing
import torch
from sklearn.impute import SimpleImputer
from model.lib.TData import TData
from torch.utils.data import DataLoader
import torch.nn.functional as F
import category_encoders

BINCLASS = 'binclass'
MULTICLASS = 'multiclass'
REGRESSION = 'regression'

ArrayDict = ty.Dict[str, np.ndarray]

def raise_unknown(unknown_what: str, unknown_value: ty.Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')

def load_json(path):
    return json.loads(Path(path).read_text())

@dc.dataclass
class Dataset:
    N: ty.Optional[ArrayDict]
    C: ty.Optional[ArrayDict]
    y: ArrayDict
    info: ty.Dict[str, ty.Any]

    @property
    def is_binclass(self) -> bool:
        return self.info['task_type'] == BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.info['task_type'] == MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.info['task_type'] == REGRESSION

    @property
    def n_num_features(self) -> int:
        return self.info['n_num_features']

    @property
    def n_cat_features(self) -> int:
        return self.info['n_cat_features']

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: str) -> int:
        """
        Return the size of the dataset partition.

        Args:

        - part: str

        Returns: int
        """
        X = self.N if self.N is not None else self.C
        assert(X is not None)
        return len(X[part])

THIS_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(THIS_PATH, '..', '..', '..'))

def dataname_to_numpy(dataset_name, dataset_path):

    """
    Load the dataset from the numpy files.

    :param dataset_name: str
    :param dataset_path: str
    :return: Tuple[ArrayDict, ArrayDict, ArrayDict, Dict[str, Any]]
    """
    dir_ = Path(os.path.join(DATA_PATH, dataset_path, dataset_name))

    def load(item) -> ArrayDict:
        return {
            x: ty.cast(np.ndarray, np.load(dir_ / f'{item}_{x}.npy', allow_pickle = True))  
            for x in ['train', 'val', 'test']
        }

    return (
        load('N') if dir_.joinpath('N_train.npy').exists() else None,
        load('C') if dir_.joinpath('C_train.npy').exists() else None,
        load('y'),
        load_json(dir_ / 'info.json'),
    )

def get_dataset(dataset_name, dataset_path):
    """
    Load the dataset from the numpy files.

    :param dataset_name: str
    :param dataset_path: str
    :return: Tuple[ArrayDict, ArrayDict, ArrayDict, Dict[str, Any]]
    """
    N, C, y, info = dataname_to_numpy(dataset_name, dataset_path)
    N_trainval = None if N is None else {key: N[key] for key in ["train", "val"]} if "train" in N and "val" in N else None
    N_test = None if N is None else {key: N[key] for key in ["test"]} if "test" in N else None

    C_trainval = None if C is None else {key: C[key] for key in ["train", "val"]} if "train" in C and "val" in C else None
    C_test = None if C is None else {key: C[key] for key in ["test"]} if "test" in C else None

    y_trainval = {key: y[key] for key in ["train", "val"]}
    y_test = {key: y[key] for key in ["test"]} 
    
    # tune hyper-parameters
    train_val_data = (N_trainval,C_trainval,y_trainval)
    test_data = (N_test,C_test,y_test)
    return train_val_data,test_data,info

def data_nan_process(N_data, C_data, num_nan_policy, cat_nan_policy, num_new_value = None, imputer = None, cat_new_value = None):
    """
    Process the NaN values in the dataset.

    :param N_data: ArrayDict
    :param C_data: ArrayDict
    :param num_nan_policy: str
    :param cat_nan_policy: str
    :param num_new_value: Optional[np.ndarray]
    :param imputer: Optional[SimpleImputer]
    :param cat_new_value: Optional[str]
    :return: Tuple[ArrayDict, ArrayDict, Optional[np.ndarray], Optional[SimpleImputer], Optional[str]]
    """
    if N_data is None:
        N = None
    else:
        N = deepcopy(N_data)
        if 'train' in N_data.keys():
            if N['train'].ndim == 1:
                N = {k: v.reshape(-1, 1) for k, v in N.items()}
        else:
            if N['test'].ndim == 1:
                N = {k: v.reshape(-1, 1) for k, v in N.items()}
        N = {k: v.astype(float) for k,v in N.items()}
        num_nan_masks = {k: np.isnan(v) for k, v in N.items()}
        if any(x.any() for x in num_nan_masks.values()):
            if num_new_value is None:
                if num_nan_policy == 'mean':
                    num_new_value = np.nanmean(N_data['train'], axis=0)
                elif num_nan_policy == 'median':
                    num_new_value = np.nanmedian(N_data['train'], axis=0)
                else:
                    raise_unknown('numerical NaN policy', num_nan_policy)
            for k, v in N.items():
                num_nan_indices = np.where(num_nan_masks[k])
                v[num_nan_indices] = np.take(num_new_value, num_nan_indices[1])
    if C_data is None:
        C = None
    else:
        assert(cat_nan_policy == 'new')
        C = deepcopy(C_data)
        if 'train' in C_data.keys():
            if C['train'].ndim == 1:
                C = {k: v.reshape(-1, 1) for k, v in C.items()}
        else:
            if C['test'].ndim == 1:
                C = {k: v.reshape(-1, 1) for k, v in C.items()}
        C = {k: v.astype(str) for k,v in C.items()}
        # assume the cat nan condition
        cat_nan_masks = {k: np.isnan(v) if np.issubdtype(v.dtype, np.number) else np.isin(v, ['nan', 'NaN', '', None]) for k, v in C.items()}
        if any(x.any() for x in cat_nan_masks.values()):  
            if cat_nan_policy == 'new':
                if cat_new_value is None:
                    cat_new_value = '___null___'
                    imputer = None
            elif cat_nan_policy == 'most_frequent':
                if imputer is None:
                    cat_new_value = None
                    imputer = SimpleImputer(strategy='most_frequent') 
                    imputer.fit(C['train'])
            else:
                raise_unknown('categorical NaN policy', cat_nan_policy)
            if imputer:
                C = {k: imputer.transform(v) for k, v in C.items()}
            else:
                for k, v in C.items():
                    cat_nan_indices = np.where(cat_nan_masks[k])
                    v[cat_nan_indices] = cat_new_value
        
    result = (N, C, num_new_value, imputer, cat_new_value)
    return result

def num_enc_process(N_data,num_policy,n_bins=2,y_train=None,is_regression=False,encoder=None):
    """
    Process the numerical features in the dataset.

    :param N_data: ArrayDict
    :param num_policy: str
    :param n_bins: int
    :param y_train: Optional[np.ndarray]
    :param is_regression: bool
    :param encoder: Optional[PiecewiseLinearEncoding]
    :return: Tuple[ArrayDict, Optional[PiecewiseLinearEncoding]]
    """
    from model.lib.num_embeddings import compute_bins,PiecewiseLinearEncoding,UnaryEncoding,JohnsonEncoding,BinsEncoding
    if N_data is not None:
        if num_policy == 'none':
            return N_data,None
        
        elif num_policy == 'Q_PLE':
            for item in N_data:
                N_data[item] = torch.from_numpy(N_data[item])
            if encoder is None:
                bins = compute_bins(N_data['train'],n_bins = n_bins,tree_kwargs = None,y=None,regression=None)
                encoder = PiecewiseLinearEncoding(bins)
            for item in N_data:
                N_data[item] = encoder(N_data[item]).cpu().numpy()

        elif num_policy == 'T_PLE':
            for item in N_data:
                N_data[item] = torch.from_numpy(N_data[item])
            if encoder is None:
                tree_kwargs = {'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4}
                bins = compute_bins(N_data['train'],n_bins = n_bins,tree_kwargs = tree_kwargs,y=torch.from_numpy(y_train),regression=is_regression)
                encoder = PiecewiseLinearEncoding(bins)
            for item in N_data:
                N_data[item] = encoder(N_data[item]).cpu().numpy()
        elif num_policy == 'Q_Unary':
            for item in N_data:
                N_data[item] = torch.from_numpy(N_data[item])
            if encoder is None:
                bins = compute_bins(N_data['train'],n_bins = n_bins,tree_kwargs = None,y=None,regression=None)
                encoder = UnaryEncoding(bins)
            for item in N_data:
                N_data[item] = encoder(N_data[item]).cpu().numpy()
        elif num_policy == 'T_Unary':
            for item in N_data:
                N_data[item] = torch.from_numpy(N_data[item])
            if encoder is None:
                tree_kwargs = {'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4}
                bins = compute_bins(N_data['train'],n_bins = n_bins,tree_kwargs = tree_kwargs,y=torch.from_numpy(y_train),regression=is_regression)
                encoder = UnaryEncoding(bins)
            for item in N_data:
                N_data[item] = encoder(N_data[item]).cpu().numpy()    
        elif num_policy == 'Q_bins':
            for item in N_data:
                N_data[item] = torch.from_numpy(N_data[item])
            if encoder is None:
                bins = compute_bins(N_data['train'],n_bins = n_bins,tree_kwargs = None,y=None,regression=None)
                encoder = BinsEncoding(bins)
            for item in N_data:
                N_data[item] = encoder(N_data[item]).cpu().numpy()
        elif num_policy == 'T_bins':
            for item in N_data:
                N_data[item] = torch.from_numpy(N_data[item])
            if encoder is None:
                tree_kwargs = {'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4}
                bins = compute_bins(N_data['train'],n_bins = n_bins,tree_kwargs = tree_kwargs,y=torch.from_numpy(y_train),regression=is_regression)
                encoder = BinsEncoding(bins)
            for item in N_data:
                N_data[item] = encoder(N_data[item]).cpu().numpy()  
        elif num_policy == 'Q_Johnson':
            for item in N_data:
                N_data[item] = torch.from_numpy(N_data[item])
            if encoder is None:
                bins = compute_bins(N_data['train'],n_bins = n_bins,tree_kwargs = None,y=None,regression=None)
                encoder = JohnsonEncoding(bins)
            for item in N_data:
                N_data[item] = encoder(N_data[item]).cpu().numpy()
        elif num_policy == 'T_Johnson':
            for item in N_data:
                N_data[item] = torch.from_numpy(N_data[item])
            if encoder is None:
                tree_kwargs = {'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4}
                bins = compute_bins(N_data['train'],n_bins = n_bins,tree_kwargs = tree_kwargs,y=torch.from_numpy(y_train),regression=is_regression)
                encoder = JohnsonEncoding(bins)
            for item in N_data:
                N_data[item] = encoder(N_data[item]).cpu().numpy()            
        
        return N_data,encoder
    else:
        return N_data,None


def data_enc_process(N_data, C_data, cat_policy, y_train = None, ord_encoder = None, mode_values = None, cat_encoder = None):
    """
    Process the categorical features in the dataset.

    :param N_data: ArrayDict
    :param C_data: ArrayDict
    :param cat_policy: str
    :param y_train: Optional[np.ndarray]
    :param ord_encoder: Optional[OrdinalEncoder]
    :param mode_values: Optional[List[int]]
    :param cat_encoder: Optional[OneHotEncoder]
    :return: Tuple[ArrayDict, ArrayDict, Optional[OrdinalEncoder], Optional[List[int]], Optional[OneHotEncoder]]
    """

    if C_data is not None:
        unknown_value = np.iinfo('int64').max - 3
        if ord_encoder is None:
            ord_encoder = sklearn.preprocessing.OrdinalEncoder(
                handle_unknown='use_encoded_value',  
                unknown_value=unknown_value, 
                dtype='int64', 
            ).fit(C_data['train'])
        C_data = {k: ord_encoder.transform(v) for k, v in C_data.items()}

        # for valset and testset, the unknown value is replaced by the mode value of the column
        if mode_values is not None:
            assert('test' in C_data.keys())
            for column_idx in range(C_data['test'].shape[1]):
                C_data['test'][:, column_idx][C_data['test'][:, column_idx] == unknown_value] = mode_values[column_idx]
        elif 'val' in C_data.keys():
            mode_values = [np.argmax(np.bincount(column[column != unknown_value]))
                        if np.any(column == unknown_value) else column[0]
                        for column in C_data['train'].T]
            for column_idx in range(C_data['val'].shape[1]):
                C_data['val'][:, column_idx][C_data['val'][:, column_idx] == unknown_value] = mode_values[column_idx]

        if cat_policy == 'indices':
            result = (N_data, C_data)
            return result[0], result[1], ord_encoder, mode_values, cat_encoder
        # use other encoding if we will treat categorical features as numerical
        elif cat_policy == 'ordinal':
            cat_encoder = ord_encoder
        elif cat_policy == 'ohe':
            if cat_encoder is None:
                cat_encoder = sklearn.preprocessing.OneHotEncoder(
                    handle_unknown='ignore', sparse_output=False, dtype='float64'
                )
                cat_encoder.fit(C_data['train'])
            C_data = {k: cat_encoder.transform(v) for k, v in C_data.items()}
        elif cat_policy == 'binary':
            if cat_encoder is None:
                cat_encoder = category_encoders.BinaryEncoder()
                cat_encoder.fit(C_data['train'].astype(str))
            C_data = {k: cat_encoder.transform(v.astype(str)).values for k, v in C_data.items()}
        elif cat_policy == 'hash':
            if cat_encoder is None:
                cat_encoder = category_encoders.HashingEncoder()
                cat_encoder.fit(C_data['train'].astype(str))
            C_data = {k: cat_encoder.transform(v.astype(str)).values for k, v in C_data.items()}
        elif cat_policy == 'loo':
            if cat_encoder is None:
                cat_encoder = category_encoders.LeaveOneOutEncoder()
                cat_encoder.fit(C_data['train'].astype(str), y_train)
            C_data = {k: cat_encoder.transform(v.astype(str)).values for k, v in C_data.items()}
        elif cat_policy == 'target':
            if cat_encoder is None:
                cat_encoder = category_encoders.TargetEncoder()
                cat_encoder.fit(C_data['train'].astype(str), y_train)
            C_data = {k: cat_encoder.transform(v.astype(str)).values for k, v in C_data.items()}
        elif cat_policy == 'catboost':
            if cat_encoder is None:
                cat_encoder = category_encoders.CatBoostEncoder()
                cat_encoder.fit(C_data['train'].astype(str), y_train)
            C_data = {k: cat_encoder.transform(v.astype(str)).values for k, v in C_data.items()}
        elif cat_policy == 'tabr_ohe':
            if cat_encoder is None:
                cat_encoder = sklearn.preprocessing.OneHotEncoder(
                    handle_unknown='ignore', sparse_output=False, dtype='float64'
                )
                cat_encoder.fit(C_data['train'])
            C_data = {k: cat_encoder.transform(v) for k, v in C_data.items()}
            result = (N_data, C_data)
            return result[0], result[1], ord_encoder, mode_values, cat_encoder
        else:
            raise_unknown('categorical encoding policy', cat_policy)
        if N_data is None:
            result = (C_data, None)
        else:
            result = ({x: np.hstack((N_data[x], C_data[x])) for x in N_data}, None)
        return result[0], result[1], ord_encoder, mode_values, cat_encoder
    else:
        return N_data, C_data, None, None, None

def data_norm_process(N_data, normalization, seed, normalizer = None):
    """
    Process the normalization of the dataset.

    :param N_data: ArrayDict
    :param normalization: str
    :param seed: int
    :param normalizer: Optional[TransformerMixin]
    :return: Tuple[ArrayDict, Optional[TransformerMixin]]
    """
    if N_data is None or normalization == 'none':
        return N_data, None

    if normalizer is None:
        N_data_train = N_data['train'].copy()

        if normalization == 'standard':
            normalizer = sklearn.preprocessing.StandardScaler()
        elif normalization == 'minmax':
            normalizer = sklearn.preprocessing.MinMaxScaler()
        elif normalization == 'quantile':
            normalizer = sklearn.preprocessing.QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(N_data['train'].shape[0] // 30, 1000), 10),
                random_state=seed
            )
        elif normalization == 'maxabs':
            normalizer = sklearn.preprocessing.MaxAbsScaler()
        elif normalization == 'power':
            normalizer = sklearn.preprocessing.PowerTransformer(method='yeo-johnson')
        elif normalization == 'robust':
            normalizer = sklearn.preprocessing.RobustScaler()
        else:
            raise_unknown('normalization', normalization)
        normalizer.fit(N_data_train)
   
    result = {k: normalizer.transform(v) for k, v in N_data.items()} 
    return result, normalizer

def data_label_process(y_data, is_regression, info = None, encoder = None):
    """
    Process the labels in the dataset.

    :param y_data: ArrayDict
    :param is_regression: bool
    :param info: Optional[Dict[str, Any]]
    :param encoder: Optional[LabelEncoder]
    :return: Tuple[ArrayDict, Dict[str, Any], Optional[LabelEncoder]]
    """
    y = deepcopy(y_data)        
    if is_regression:
        y = {k: v.astype(float) for k,v in y.items()}
        if info is None:
            mean, std = y_data['train'].mean(), y_data['train'].std()
        else:
            mean, std = info['mean'], info['std']
        y = {k: (v - mean) / std for k, v in y.items()}
        info = {'policy': 'mean_std', 'mean': mean, 'std': std}
        return y, info, None
    else:
        # classification
        if encoder is None:
            encoder = sklearn.preprocessing.LabelEncoder().fit(y['train'])
        y = {k:encoder.transform(v) for k, v in y.items()}
        return y, {'policy': 'none'}, encoder

def data_loader_process(is_regression, X, Y, y_info, device, batch_size, is_train):
    """
    Process the data loader.

    :param is_regression: bool
    :param X: Tuple[ArrayDict, ArrayDict]
    :param Y: ArrayDict
    :param y_info: Dict[str, Any]
    :param device: torch.device
    :param batch_size: int
    :param is_train: bool
    :return: Tuple[ArrayDict, ArrayDict, ArrayDict, DataLoader, DataLoader, Callable]
    """
    X = tuple(None if x is None else to_tensors(x) for x in X)
    Y = to_tensors(Y)

    X = tuple(None if x is None else {k: v.to(device) for k, v in x.items()} for x in X)
    Y = {k: v.to(device) for k, v in Y.items()}

    if X[0] is not None:
        X = ({k: v.double() for k, v in X[0].items()}, X[1])

    if is_regression:
        Y = {k: v.double() for k, v in Y.items()}
    else:
        Y = {k: v.long() for k, v in Y.items()}
    
    loss_fn = (
        F.mse_loss
        if is_regression
        else F.cross_entropy
    )

    if is_train:
        trainset = TData(is_regression, X, Y, y_info, 'train')
        valset = TData(is_regression, X, Y, y_info, 'val')
        train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)        
        val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, num_workers=0) 
        return X[0], X[1], Y, train_loader, val_loader, loss_fn
    else:
        testset = TData(is_regression, X, Y, y_info, 'test')
        test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)        
        return X[0], X[1], Y, test_loader, loss_fn


    

def to_tensors(data: ArrayDict) -> ty.Dict[str, torch.Tensor]:
    """
    Convert the numpy arrays to torch tensors.

    :param data: ArrayDict
    :return: Dict[str, torch.Tensor]
    """
    return {k: torch.as_tensor(v) for k, v in data.items()}

def get_categories(
    X_cat: ty.Optional[ty.Dict[str, torch.Tensor]]
) -> ty.Optional[ty.List[int]]:
    """
    Get the categories for each categorical feature.

    :param X_cat: Optional[Dict[str, torch.Tensor]]
    :return: Optional[List[int]]
    """
    return (
        None
        if X_cat is None
        else [
            len(set(X_cat['train'][:, i].tolist()))
            for i in range(X_cat['train'].shape[1])
        ]
    )
