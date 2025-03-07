import copy
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from warnings import warn

import numpy as np
import pandas as pd
import sklearn
import torch
import multiprocessing as mp
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics._dist_metrics import check_array
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y

from TALENT.model.lib.realmlp.alg_interfaces.alg_interfaces import AlgInterface
from TALENT.model.lib.realmlp.alg_interfaces.base import SplitIdxs, InterfaceResources
from TALENT.model.lib.realmlp.data.data import DictDataset, TensorInfo
from TALENT.model.lib.realmlp.data.splits import RandomSplitter, KFoldSplitter
from TALENT.model.lib.realmlp.data.conversion import ToDictDatasetConverter
from TALENT.model.lib.realmlp.torch_utils import get_available_device_names
from TALENT.model.lib.realmlp.training.logging import StdoutLogger


def to_df(x) -> pd.DataFrame:
    try:
        return pd.DataFrame(x)
    except:
        pass

    return pd.DataFrame(np.array(x))


def to_normal_type(x) -> Any:
    if isinstance(x, pd.DataFrame) or isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, pd.Series):
        return x
    return np.asarray(x)


class AlgInterfaceEstimator(BaseEstimator):
    """
    Base class for wrapping AlgInterface subclasses with a scikit-learn compatible interface.
    """
    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        # override this
        raise NotImplementedError()

    def _supports_multioutput(self) -> bool:
        # only relevant for regression, override this if multioutput is not supported
        return True

    def _supports_single_class(self) -> bool:
        # only relevant for classification,
        # override this if training with only a single class in the training set is not supported
        return True

    def _supports_single_sample(self) -> bool:
        return True

    def _is_classification(self) -> bool:
        raise NotImplementedError()

    def _get_default_params(self) -> Dict[str, Any]:
        # override this in subclasses to handle default parameters that should not be treated in the constructor
        # e.g. because their default values are mutable (list/dict/...)
        return dict()

    def _allowed_device_names(self) -> List[str]:
        # override in subclasses that allow to run on a GPU or mps
        return ['cpu']

    def get_config(self) -> Dict[str, Any]:
        """
        Augments the result from self.get_params() with the parameters from self._get_default_params().
        Default parameters are used if the value in get_params() is either None or not present.
        :return: Dictionary of parameters augmented with default parameters.
        """
        params = copy.copy(self.get_params(deep=False))
        default_params = self._get_default_params()
        for key, value in default_params.items():
            if key not in params or params[key] is None:
                params[key] = value

        # return params
        # remove None values
        return {key: value for key, value in params.items() if value is not None}

    def fit(self, X, y, val_idxs: Optional[np.ndarray] = None,
            cat_features: Optional[Union[List[bool], np.ndarray]] = None) -> BaseEstimator:
        """
        Fit the estimator.

        :param X: Inputs (covariates). pandas DataFrame, numpy array, or similar array-like.
        :param y: Labels (targets, variates). pandas DataFrame/Series, numpy array, or similar array-like.
        :param val_idxs: Indices of validation set elements within X and y (optional).
            Can be an array of shape (n_val_samples,) or (n_val_splits,n_val_samples_per_split).
            In the latter case, the results of the models on the validation splits will be ensembled.
        :param cat_features: Which features/columns are categorical, specified as a list or array of booleans.
            If this is not specified, all columns with category/string/object dtypes are interpreted as categorical
            and all others as numerical.
        :return: Returns self.
        """
        # Check that X and y have correct shape
        # if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        # we don't want to store the converted ones here

        # print(f'{X=}')
        # print(f'{y=}')
        check_X_y(X, y, force_all_finite='allow-nan', multi_output=True)
        X = to_normal_type(X)
        y = to_normal_type(y)  # need to convert array-like objects to arrays for self.is_y_1d_

        if self._is_classification():
            # classes_ is overridden later, but this raises an error when y is a regression target, so it is useful
            self.classes_ = unique_labels(y)

        self.is_y_1d_ = isinstance(y, pd.Series) or (isinstance(y, np.ndarray) and len(y.shape) == 1)
        if isinstance(y, list):
            if len(np.asarray(y).shape) == 1:
                self.is_y_1d_ = True

        self.x_converter_ = ToDictDatasetConverter(cat_features=cat_features)
        self.y_encoder_ = OrdinalEncoder(dtype=np.int64)
        # if not (isinstance(y, np.ndarray) or isinstance(y, list)
        #         or isinstance(y, pd.DataFrame) or isinstance(y, pd.Series)):
        #     raise ValueError(f'y has type {type(y)}, but should be one of np.ndarray, list, pd.DataFrame, or pd.Series')
        # y_df = pd.DataFrame(y)
        X_df = to_df(X).copy()
        y_df = to_df(y).copy()
        # self.y_encoder_.fit_transform(y)

        if not self._supports_single_sample() and len(X_df) == 1:
            raise ValueError('Training with one sample is not supported!')

        x_ds = self.x_converter_.fit_transform(X_df)
        if torch.any(torch.isnan(x_ds.tensors['x_cont'])):
            raise ValueError('NaN values in continuous columns are currently not allowed!')

        self.is_y_float64_ = False  # checked later in the regression case

        # convert y
        if self._is_classification():
            self.y_encoder_ = OrdinalEncoder(dtype=np.int64)
            y_tfmd = self.y_encoder_.fit_transform(y_df)
            if len(y_tfmd.shape) == 1:
                y_tfmd = y_tfmd[:, None]
            if len(y_tfmd.shape) != 2:
                raise ValueError('len(y.shape) != 2')
            if y_tfmd.shape[1] != 1:
                raise ValueError('Multilabel classification is not supported!')
            if not self.is_y_1d_:
                warn(
                    (
                        "A column-vector y was passed when a 1d array was"
                        " expected. Please change the shape of y to "
                        "(n_samples,), for example using ravel()."
                    ),
                    DataConversionWarning,
                    stacklevel=2,
                )
            y_ds = DictDataset(tensors={'y': torch.as_tensor(y_tfmd, dtype=torch.long)},
                               tensor_infos={'y': TensorInfo(cat_sizes=[int(np.max(y_tfmd) + 1)])})
            self.classes_ = self.y_encoder_.categories_[0]
            if not self._supports_single_class() and len(self.classes_) == 1:
                raise ValueError(f'Training with only one class in the training set is not supported!')
        else:
            # regression
            if y_df[y_df.columns[0]].dtype == np.float64:
                self.is_y_float64_ = True
            y_tfmd = y_df.to_numpy(dtype=np.float32)
            if len(y_tfmd.shape) == 1:
                y_tfmd = y_tfmd[:, None]
            if len(y_tfmd.shape) != 2:
                raise ValueError('len(y.shape) != 2')
            y_ds = DictDataset(tensors={'y': torch.as_tensor(y_tfmd, dtype=torch.float32)},
                               tensor_infos={'y': TensorInfo(feat_shape=[y_tfmd.shape[1]])})
            if not self._supports_multioutput() and not self.is_y_1d_:
                warn(
                    (
                        "A column-vector y was passed when a 1d array was"
                        " expected. Please change the shape of y to "
                        "(n_samples,), for example using ravel()."
                    ),
                    DataConversionWarning,
                    stacklevel=2,
                )
            if not self._supports_multioutput() and y_ds.tensor_infos['y'].get_n_features() > 1:
                raise ValueError('Multioutput regression is not supported, '
                                 'please wrap this estimator with the MultiOutputRegressor '
                                 'from scikit-learn.')

        ds = DictDataset.join(x_ds, y_ds)

        # set n_features_in_ as required by https://scikit-learn.org/stable/developers/develop.html
        self.n_features_in_ = ds.tensor_infos['x_cont'].get_n_features() + ds.tensor_infos['x_cat'].get_n_features()

        params = self.get_config()
        n_cv = params.get('n_cv', 1)
        # val_fraction is only relevant for n_cv == 1
        val_fraction = params.get('val_fraction', 0.2)
        n_refit = params.get('n_refit', 0)

        self.cv_alg_interface_ = self._create_alg_interface(n_cv=n_cv)

        # ----- get random seeds -----
        random_state = params.get('random_state', None)
        if isinstance(random_state, int):
            seed = random_state
        elif random_state is None:
            seed = int(np.random.randint(0, 2 ** 31 - 1))
        elif isinstance(random_state, np.random.RandomState):
            seed = int(random_state.randint(0, 2 ** 31 - 1))
        else:
            raise ValueError(f'random_state type {type(random_state)} '
                             f'is not one of [NoneType, int, np.random.RandomState]')

        split_seed = seed
        refit_split_seed = seed + 1
        sub_split_seeds = list(np.random.RandomState(split_seed).randint(0, 2 ** 31 - 1, size=n_cv))
        sub_split_seeds = [int(seed) for seed in sub_split_seeds]
        refit_sub_split_seeds = list(np.random.RandomState(refit_split_seed).randint(0, 2 ** 31 - 1, size=n_refit))
        refit_sub_split_seeds = [int(seed) for seed in refit_sub_split_seeds]

        # ----- get train/val split -----

        if not isinstance(n_cv, int) or n_cv <= 0:
            raise ValueError(f'Expected n_cv to be an int >= 1, but got {n_cv=}')
        if val_idxs is not None:
            val_idxs = torch.as_tensor(val_idxs, dtype=torch.long)
            if len(val_idxs.shape) == 1:
                val_idxs = val_idxs[None, :]

            train_idxs_list = []
            for i in range(val_idxs.shape[0]):
                is_val_idx = torch.zeros(ds.n_samples, dtype=torch.bool)
                is_val_idx[val_idxs[i]] = True
                train_idxs_list.append(torch.argwhere(~is_val_idx).squeeze(-1))

            train_idxs = torch.stack(train_idxs_list, dim=0)
            if val_idxs.shape[0] == 1 and n_cv > 1:
                # replicate according to n_cv, such that an ensemble can be created
                train_idxs = train_idxs.expand(n_cv, -1)
                val_idxs = val_idxs.expand(n_cv, -1)
            elif n_cv != val_idxs.shape[0]:
                raise ValueError(f'Value provided for {n_cv=} is not equal to {val_idxs.shape[0]=}')
        else:
            if n_cv == 1:
                # random split
                splitter = RandomSplitter(seed=split_seed, first_fraction=1.0 - val_fraction)
                train_idxs, val_idxs = splitter.get_idxs(ds)
                train_idxs = train_idxs[None, :]
                val_idxs = val_idxs[None, :]
            else:
                splitter = KFoldSplitter(k=n_cv, seed=split_seed, stratified=self._is_classification())
                idxs_tuples = splitter.get_idxs(ds)
                train_idxs = torch.stack([t[0] for t in idxs_tuples], dim=0)
                val_idxs = torch.stack([t[1] for t in idxs_tuples], dim=0)

        if val_idxs.shape[1] == 0:
            val_idxs = None  # no validation set

        idxs_list = [SplitIdxs(train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=None, split_seed=split_seed,
                               sub_split_seeds=sub_split_seeds, split_id=0)]

        # print(f'{ds.tensors["x_cont"]=}')

        # ----- resources -----
        n_logical_threads = mp.cpu_count()
        device = params.get('device', None)
        n_threads = params.get('n_threads', n_logical_threads)

        gpu_devices = []

        device_names = get_available_device_names()
        device = str(device) # added
        # print(f'{type(device_names[1])}, {type(device)}')
        if device is None:
            allowed_device_names = [name for name in device_names if name.split(':')[0] in self._allowed_device_names()]
            if 'cuda:0' in allowed_device_names:
                gpu_devices.append('cuda:0')
            elif 'mps' in allowed_device_names:
                gpu_devices.append('mps')
            # print(f'{gpu_devices=}')
            # print(f'{self._allowed_device_names()=}')
            # print(f'{allowed_device_names=}')
            # print(f'{device_names=}')
        elif device != 'cpu':
            if device not in device_names:
                raise ValueError(f'Unknown device name "{device}", known device names are {device_names}')
            gpu_devices.append(device)

        tmp_folder: Optional[str] = params.get('tmp_folder', None)
        if tmp_folder is None:
            tmp_folders = [None]
            refit_tmp_folders = [None]
        else:
            tmp_path = Path(tmp_folder)
            # make sure that the refit stage doesn't load the models from the cv stage
            tmp_folders = [tmp_path / 'cv']
            refit_tmp_folders = [tmp_path / 'refit']

        logger = StdoutLogger(verbosity_level=params.get('verbosity', 0))

        interface_resources = InterfaceResources(n_threads=n_threads, gpu_devices=gpu_devices)
        self.cv_alg_interface_.fit(ds=ds, idxs_list=idxs_list, interface_resources=interface_resources,
                                   logger=logger, tmp_folders=tmp_folders, name=self.__class__.__name__)

        # todo: put alg_interface on the CPU after fit() (for saving)? How to do it?

        # todo: currently, there is only one alg_interface which may fit in parallel (for the NNs),
        #  but we could add an option to make them fit sequentially for RAM reasons or so
        #  (maybe this is best done via a MultiSplitWrapper or so)

        if n_refit > 0:
            self.refit_alg_interface_ = self.cv_alg_interface_.get_refit_interface(n_refit=n_refit)
            train_idxs = torch.arange(ds.n_samples, dtype=torch.long)[None, :].expand(n_refit, -1)
            refit_idxs_list = [SplitIdxs(train_idxs=train_idxs,
                                         val_idxs=None, test_idxs=None, split_seed=refit_split_seed,
                                         sub_split_seeds=refit_sub_split_seeds, split_id=0)]
            self.refit_alg_interface_.fit(ds=ds, idxs_list=refit_idxs_list, interface_resources=interface_resources,
                                          logger=logger, tmp_folders=refit_tmp_folders,
                                          name=self.__class__.__name__ + ' [refit]')
            self.alg_interface_ = self.refit_alg_interface_
        else:
            self.alg_interface_ = self.cv_alg_interface_

        return self

    def _predict_raw(self, X) -> torch.Tensor:
        """
        Predicts logits (for classification) or mean outputs (for regression)
        :param X: Input data.
        :return: Returns a tensor of shape [n_ensemble, n_samples, output_dim].
        """
        # Check is fit had been called
        check_is_fitted(self, ['alg_interface_', 'x_converter_'])

        # Input validation
        # if isinstance(X, np.ndarray):
        check_array(X, force_all_finite='allow-nan')

        x_ds = self.x_converter_.transform(to_df(X))
        if torch.any(torch.isnan(x_ds.tensors['x_cont'])):
            raise ValueError('NaN values in continuous columns are currently not allowed!')
        y_preds = self.alg_interface_.predict(x_ds).detach().cpu()
        return y_preds


class AlgInterfaceClassifier(AlgInterfaceEstimator, ClassifierMixin):
    def _is_classification(self) -> bool:
        return True

    def predict_proba(self, X) -> np.ndarray:
        y_preds = self._predict_raw(X)
        # y_preds are logits, so take the softmax and the mean over the ensemble dimension afterward
        y_probs = torch.softmax(y_preds, dim=-1).mean(dim=0)
        return y_probs.numpy()

    def predict_proba_ensemble(self, X) -> np.ndarray:
        # same as predict_proba but does not average over ensemble members
        y_preds = self._predict_raw(X)
        # y_preds are logits, so take the softmax and the mean over the ensemble dimension afterward
        y_probs = torch.softmax(y_preds, dim=-1)
        return y_probs.numpy()

    def get_validation_predictions_and_labels(self):
        pass

    def predict(self, X):
        """ Predict labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        y_probs = self.predict_proba(X)
        # class_idxs = np.argmax(y_probs, axis=-1)
        # ret = np.asarray(self.classes_)[class_idxs]  # modified to return probs
        return y_probs

    def predict_ensemble(self, X):
        y_probs = self.predict_proba_ensemble(X)
        class_idxs = np.argmax(y_probs, axis=-1)
        return np.asarray(self.classes_)[class_idxs]


class AlgInterfaceRegressor(AlgInterfaceEstimator, RegressorMixin):
    def _is_classification(self) -> bool:
        return False

    def _more_tags(self):
        return {'multioutput': self._supports_multioutput()}

    def predict(self, X):
        y_preds = self._predict_raw(X)
        y_np = y_preds.mean(dim=0).numpy()
        if self.is_y_1d_:
            y_np = y_np[:, 0]
        if self.is_y_float64_:
            y_np = y_np.astype(np.float64)

        return y_np

    def predict_ensemble(self, X):
        y_preds = self._predict_raw(X)
        y_np = y_preds.numpy()
        if self.is_y_1d_:
            y_np = y_np[:, :, 0]
        if self.is_y_float64_:
            y_np = y_np.astype(np.float64)

        return y_np
