from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict

import torch

from TALENT.model.lib.realmlp.alg_interfaces.base import SplitIdxs, InterfaceResources, RequiredResources
from TALENT.model.lib.realmlp.data.nested_dict import NestedDict

from TALENT.model.lib.realmlp.data.data import DictDataset
from TALENT.model.lib.realmlp.training.logging import Logger
from TALENT.model.lib.realmlp.training.metrics import Metrics


class AlgInterface:
    """
    AlgInterface is an abstract base class for tabular ML methods
    with an interfaces that offers more possibilities than a standard scikit-learn interface.

    In particular, it allows for parallelized fitting of multiple models, bagging, and refitting.
    The idea is as follows:

    - The dataset can be split into a test set and the remaining data. (We call this a trainval-test split.)
        The fit() method allows to specify multiple such splits,
        and some AlgInterface implementations (NNAlgInterface) allow to vectorize computations across these splits.
        However, for vectorization, we may require that the test set sizes are identical in all splits.
    - The remaining data can further be split into training and validation data. (We call this a train-val split.)
        AlgInterface allows to fit with one or multiple train-val splits, which can also be vectorized in NNAlgInterface.
        Optionally, the function `get_refit_interface()` allows to extract an AlgInterface that can be used for
        fitting the model on training+validation set
        with the best settings found on the validation set in the cross-validation stage (represented by self.fit_params).
        These "best settings" could be an early stopping epoch or number of trees,
        or best hyperparameters found by hyperparameter optimization.
        We call this refitting.

    Another feature of AlgInterface is that it provides methods to get (an estimate of) required resources
    and to evaluate metrics on training, validation, and test set.
    """

    def __init__(self, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        """
        :param fit_params: This parameter can be used to store the best hyperparameters
            found during fit() in (cross-)validation mode. These can then be used for fit() in refitting mode.
            If fit_params is not None, it should be a list with one dictionary per trainval-test split.
            The dictionaries then contain the obtained hyperparameters for each of the trainval-test splits.
            Normally, there are no best parameters per train-val split
            as we might not have the same number of refitted models as train-val splits.
        :param config: Other parameters.
        """
        self.config = config
        self.fit_params = fit_params

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> Optional[
        List[List[List[Tuple[Dict, float]]]]]:
        """
        Fit the models on the given data and splits.
        Should be overridden by subclasses unless fit_and_eval() is overloaded.
        In the latter case, this method will by default use fit_and_eval() and discard the evaluation.

        :param ds: DictDataset representing the dataset. Should be on the CPU.
        :param idxs_list: List containing one SplitIdxs object per trainval-test split. Indices should be on the CPU.
        :param interface_resources: Resources assigned to fit().
        :param logger: Logger that can be used for logging.
        :param tmp_folders: List of paths that can be used for storing intermediate data.
            The paths can be None, in which case methods will try not to save intermediate results.
            There should be one folder per trainval-test-split (i.e. only one per k-fold CV).
        :param name: Name of the algorithm (for logging).
        :return: May return information about different possible fit_params settings that can be used.
            Say a variable `results` is returned that is not None.
            Then, results[tt_split_idx][tv_split_idx] should be a list of tuples (params, loss).
            This is useful for k-fold cross-validation,
            where the params with the best average loss (averaged over tv_split_idx) can be selected for fit_params.
        """
        if self.__class__.fit_and_eval == AlgInterface.fit_and_eval:
            raise NotImplementedError()  # avoid infinite recursion
        else:
            self.fit_and_eval(ds, idxs_list, interface_resources, logger, tmp_folders, name, metrics=None,
                              return_preds=False)
        return None

    def fit_and_eval(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
                     logger: Logger, tmp_folders: List[Optional[Path]], name: str, metrics: Optional[Metrics],
                     return_preds: bool) -> List[NestedDict]:
        """
        Run fit() with the given parameters and then return the result of eval() with the given metrics.
        This method can be overridden instead of fit() if it is more convenient.
        The idea is that for hyperparameter optimization,
        one has to evaluate each hyperparameter combination anyway after training it,
        so it is more efficient to implement fit_and_eval() and return the evaluation of the best method at the end.
        See the documentation of fit() and eval() for the meaning of the parameters and returned values.
        """
        if self.__class__.fit == AlgInterface.fit:
            raise NotImplementedError()  # avoid infinite recursion
        self.fit(ds=ds, idxs_list=idxs_list, interface_resources=interface_resources,
                 logger=logger, tmp_folders=tmp_folders, name=name)
        return self.eval(ds=ds, idxs_list=idxs_list, metrics=metrics, return_preds=return_preds)

    def eval(self, ds: DictDataset, idxs_list: List[SplitIdxs], metrics: Optional[Metrics],
             return_preds: bool) -> List[NestedDict]:
        """
        Evaluates the (already fitted) method using various metrics on training, validation, and test sets.
        The results will also contain the found fit_params and optionally the predictions on the dataset.
        This method should normally not be overridden in subclasses.

        :param ds: Dataset.
        :param idxs_list: List of indices for the training-validation-test splits,
            one per trainval-test split as in fit().
        :param metrics: Metrics object that defines which metrics should be evaluated.
            If metrics is None, an empty list will be returned
            (which might avoid unnecessary computation when implementing fit() through fit_and_eval()).
        :param return_preds: Whether the predictions on the dataset should be included in the returned results.
        :return: Returns a list with one NestedDict for every trainval-test split.
            Denote by `results` such a NestedDict object. Then, `results` will contain the following contents:
            results['metrics', 'train'/'val'/'test', str(n_models), str(start_idx), metric_name] = metric_value
            Here, an ensemble of the predictions of models [start_idx:start_idx+n_models] will be used.
            results['y_preds'] = a list (converted from a tensor) with predictions on the whole dataset,
            included only if return_preds==True.
            results['fit_params'] = self.fit_params
        """

        if metrics is None:
            results = []
            # for idxs in idxs_list:
            #     result = NestedDict()
            #     for split_name in ['train', 'val', 'test']:
            #         result['metrics'][split_name]['1']['0'] = dict()
            #     if return_preds:
            #         pass
            #     results.append(dict(metrics))
            return results
        X, y = ds.split_xy()
        y = y.tensors['y']
        y_pred_full = self.predict(X).detach().cpu()
        # print(f'{y=}')
        # print(f'{y_pred_full=}')
        # print(f'{y.shape=}')
        # print(f'{y_pred_full.shape=}')
        idx = 0
        results_list = []
        for split_idx, idxs in enumerate(idxs_list):
            results = NestedDict()

            y_preds = y_pred_full[idx:idx + idxs.n_trainval_splits]
            if return_preds:
                results['y_preds'] = y_preds.numpy().tolist()
            idx += idxs.n_trainval_splits

            if idxs.test_idxs is not None:
                # print(f'{y_preds.shape=}')
                # print(f'{y.shape=}')
                results['metrics', 'test'] = metrics.compute_metrics_dict(
                    y_preds=[y_preds[i, idxs.test_idxs] for i in range(y_preds.shape[0])],
                    y=y[idxs.test_idxs],
                    use_ens=True)
            train_metrics = NestedDict()
            val_metrics = NestedDict()
            for i in range(idxs.n_trainval_splits):
                train_dict = metrics.compute_metrics_dict([y_preds[i, idxs.train_idxs[i]]], y[idxs.train_idxs[i]],
                                                          use_ens=False)
                train_metrics['1', str(i)] = train_dict['1', '0']

                if idxs.val_idxs is not None and idxs.val_idxs.shape[-1] > 0:
                    val_dict = metrics.compute_metrics_dict([y_preds[i, idxs.val_idxs[i]]], y[idxs.val_idxs[i]],
                                                            use_ens=False)
                    val_metrics['1', str(i)] = val_dict['1', '0']

            results['metrics', 'train'] = train_metrics
            if idxs.val_idxs is not None:
                results['metrics', 'val'] = val_metrics
            if self.fit_params is not None:
                results['fit_params'] = self.fit_params[split_idx]
            results_list.append(results)

        return results_list

    def predict(self, ds: DictDataset) -> torch.Tensor:
        """
        Method to predict labels on the given dataset. Override in subclasses.

        :param ds: Dataset on which to predict labels
        :return: Returns a tensor of shape [n_trainval_splits * n_splits, ds.n_samples, output_shape]
            In the classification case, output_shape will be the number of classes (even in the binary case)
            and the outputs will be logits (i.e., softmax should be applied to get probabilities)
            In the regression case, output_shape will be the target dimension (often 1).
        """
        raise NotImplementedError()

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        """
        Returns another AlgInterface that is configured for refitting on the training and validation data.
        Override in subclasses.

        :param n_refit: Number of models that should be refitted (with different seeds) per trainval-test split.
        :param fit_params: Fit parameters (see the constructor) that should be used for refitting.
            If fit_params is None, self.fit_params will be used instead.
        :return: Returns the AlgInterface object for refitting.
        """
        raise NotImplementedError()

    def get_fit_params(self) -> Optional[List[Dict]]:
        """
        :return: Return self.fit_params.
        """
        return self.fit_params

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int]) -> RequiredResources:
        """
        Estimate the required resources for fit().

        :param ds: Dataset. Does not have to contain tensors.
        :param n_cv: Number of train-val splits per trainval-test split.
        :param n_refit: Number of refitted models per trainval-test split.
        :param n_splits: Number of trainval-test splits.
        :param split_seeds: Seeds for every trainval-test split.
        :return: Returns estimated required resources.
        """
        raise NotImplementedError()