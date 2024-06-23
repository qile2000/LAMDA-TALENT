import numpy as np
import sklearn
import torch
from torch.utils.data import Dataset
import typing as ty
ArrayDict = ty.Dict[str, np.ndarray]
def prepare_meta_feature(X, Y, args):
    # get same-class mask
    X_train, Y_train =  X['train'], Y['train']
    num_class = np.unique(Y_train).shape[0]
    X_train_split = [X_train[Y_train == k, :] for k in range(num_class)]
    centers = [sklearn.utils.resample(X_c, replace=False, n_samples=np.minimum(args.centers_num, X_c.shape[0]), random_state=args.seed) for X_c in X_train_split]     
    return centers

def prepare_meta_feature_regression(X, Y, args, dataname=None, is_meta=False):
    X_train, Y_train =  X['train'], Y['train']
    XY_train = np.concatenate((X_train, Y_train.reshape(-1,1)), axis=1)
    centers = sklearn.utils.resample(XY_train, replace=False, n_samples=np.minimum(args.centers_num, XY_train.shape[0]), random_state=args.seed) 
    return centers

def to_tensors(data: ArrayDict) -> ty.Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v) for k, v in data.items()}

class TabPTMData(Dataset):

    def __init__(self, dataset, X, Y, y_info, part):
        assert(part in ['train', 'val', 'test'])
        self.X_num = X[part]
        self.Y, self.y_info = Y[part], y_info

        self.dataset = dataset        
        self.num_class = 1 if dataset.is_regression else torch.unique(Y['train']).shape[0]
        self.is_regression = dataset.is_regression

    def get_dim_in(self):
        return 0 if self.X_num is None else self.X_num.shape[1]

    def get_categories(self):
        return (
            None
        )

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, i):
        data, label = self.X_num[i], self.Y[i]

        return data, label
