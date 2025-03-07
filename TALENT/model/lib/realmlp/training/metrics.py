from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np
import torchmetrics
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
import torch.nn.functional as F
import torch
import copy

from TALENT.model.lib.realmlp.data.data import DictDataset, TaskType
from TALENT.model.lib.realmlp.data.nested_dict import NestedDict
from TALENT.model.lib.realmlp.torch_utils import cat_if_necessary, torch_np_quantile
from TALENT.model.lib.realmlp.training.auc_mu import auc_mu_impl


# see also: https://scikit-learn.org/stable/modules/model_evaluation.html

# unify criterion and metric?
# class Metric
# then have something like SumMetric, ZeroMetric
# make metrics stateless
# how to associate them with LightGBM / XGBoost / CatBoost metrics?
# then allow to configure train_metric, val_metric and test_metric
# some metrics are non-differentiable and/or only have CPU computations
# in train loss, don't use .item() -> have two different methods
# some metrics are related, like auroc and 1-auc_ovo or mse and rmse, have conversion code?
# maybe distinguish between metric that outputs a dict and metric that outputs a number -
# Metrics could then just output a dict?
# have different reductions. But: auroc has pre-defined reduction. Maybe reductions only for tensor version?
# maybe have different subclasses for metrics that don't support everything
# rename Metrics -> MetricsTracker
# when using dict-computations, do them on floats or also on torch Tensors?
# todo: have different computation paths similar to old code?
# todo: how should evaluation code know which granularity of results needs to be loaded?
# todo: allow vectorization in apply_torch()
# confusion_matrix would not be a float, but maybe not necessary
# multi-output / combinations should also work in torch, for cross_entropy / ls combination
# kldiv+ls from ce and ce_unif needs n_classes. Maybe have metric n_classes from y_pred?
# also use the Factory/Fitter/Layer structure for metrics? Problem: auroc, which also does batch aggregation
# allow construction from metric name (for evaluation), and allow multiple names for the same metric?
# (e.g. cross_entropy, kldiv and logloss or auc and auroc)
# if we have class_error plus zero, then the name of that should still be class_error?
# conversion to LGBM/... names not unique since class_error is called differently depending on bin/multiclass
# problem: ls converts labels to soft labels


class Metric:
    def try_append(self, metrics_dict: Dict[str, Any]):
        pass

    def apply_dict(self, metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
        result_dict = copy.copy(metrics_dict)
        self.try_append(result_dict)
        # todo: this does not work properly if the metric of interest is already contained in metrics_dict
        # maybe need get_names() function?
        return {key: value for key, value in result_dict.items() if key not in metrics_dict}


class MultiMetric(Metric):
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics

    def try_append(self, metrics_dict: Dict[str, Any]):
        pass  # todo: call try_append of self.metrics


class SingleMetric(Metric):
    def get_name(self) -> str:
        raise NotImplementedError()

    def apply(self, y_pred: torch.Tensor, y: torch.Tensor) -> Any:
        metrics_dict = {'y_pred': y_pred, 'y': y}
        self.try_append(metrics_dict)
        return metrics_dict[self.get_name()]


class TorchMetric(Metric):
    def apply_torch(self, y_pred: torch.Tensor, y: torch.Tensor, reduction: Optional[str] = 'mean') -> torch.Tensor:
        raise NotImplementedError()

    def apply(self, y_pred: torch.Tensor, y: torch.Tensor) -> Any:
        return self.apply_torch(y_pred, y).item()


def zip_dicts(list_of_dicts):
    return {key: [list_of_dicts[i][key] for i in range(len(list_of_dicts))] for key in list_of_dicts[0].keys()}


def unzip_dict(dict_of_lists: dict):
    list_length = len(next(iter(dict_of_lists.items())))
    return [{key: dict_of_lists[key][i] for i in range(list_length)} for key in dict_of_lists.keys()]


def to_one_hot(y, num_classes, label_smoothing_eps=0.0):
    one_hot = F.one_hot(y, num_classes).float()
    if label_smoothing_eps > 0.0:
        low = label_smoothing_eps / num_classes
        high = 1.0 - label_smoothing_eps + low
        return low + (high - low) * one_hot
    else:
        return one_hot


def apply_reduction(res, reduction):
    if reduction == 'mean':
        return res.mean(dim=-1)
    elif reduction is None:
        return res
    elif reduction == 'sum':
        return res.sum(dim=-1)

    return None


def cross_entropy(y_pred: torch.Tensor, y: torch.Tensor, reduction='mean'):
    if torch.is_floating_point(y):
        res = (-F.log_softmax(y_pred, dim=-1) * y).sum(dim=-1)
    else:
        res = -F.log_softmax(y_pred, dim=-1).gather(-1, y).squeeze(-1)
    return apply_reduction(res, reduction)


def softmax_kldiv(y_pred: torch.Tensor, y: torch.Tensor, reduction='mean'):
    if torch.is_floating_point(y):
        # add 1e-30 to prevent taking the log of 0 -> it gets then multiplied by 0 anyway
        res = (((y + 1e-30).log() - F.log_softmax(y_pred, dim=-1)) * y).sum(dim=-1)
    else:
        res = -F.log_softmax(y_pred, dim=-1).gather(-1, y).squeeze(-1)
    return apply_reduction(res, reduction)


def brier_loss(y_pred: torch.Tensor, y: torch.Tensor, reduction='mean'):
    if not torch.is_floating_point(y):
        y = F.one_hot(y.squeeze(-1), num_classes=y_pred.shape[-1])
    res = (F.softmax(y_pred, dim=-1) - y).square().sum(dim=-1)
    return apply_reduction(res, reduction)


def cos_loss(y_pred, y, reduction='mean'):
    if not torch.is_floating_point(y):
        y = F.one_hot(y.squeeze(-1), num_classes=y_pred.shape[-1])
    res = 1.0 - (y_pred * y).sum(dim=-1) / (y_pred.norm(dim=-1) + 1e-3)
    return apply_reduction(res, reduction)


def mse(y_pred, y, reduction='mean'):
    if not torch.is_floating_point(y):
        # in case mse should be used for classification
        y = F.one_hot(y.squeeze(-1), num_classes=y_pred.shape[-1])
    if y_pred.dim() != y.dim():
        raise RuntimeError('MSE: y_pred.dim() != y.dim(): could lead to broadcasting errors')
    res = ((y_pred - y) ** 2).mean(dim=-1)
    return apply_reduction(res, reduction)


def pinball_loss(y_pred: torch.Tensor, y: torch.Tensor, quantile: float, reduction='mean'):
    if y_pred.dim() != y.dim():
        raise RuntimeError('Pinball loss: y_pred.dim() != y.dim(): could lead to broadcasting errors')
    err = y_pred - y
    # print(f'{quantile*err=}')
    res = torch.maximum((1-quantile) * err, -quantile * err).mean(dim=-1)
    return apply_reduction(res, reduction)


def mean_interleave(input, repeats, dim):
    assert input.shape[dim] % repeats == 0
    new_shape = input.shape[:dim] + [input.shape[dim] // repeats, repeats] + input.shape[dim+1:]
    return input.view(new_shape).mean(dim=dim+1)


def get_y_probs(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Returns the empirical probabilities of all classes in y.
    :param y: Tensor of shape [..., n_batch, 1] and dtype torch.long or another integer dtype,
    containing class labels in {0, 1, ..., n_classes-1}
    :param n_classes: Total number of classes
    :return: returns a tensor of shape [..., n_classes]
    """
    if y.shape[-1] != 1:
        raise ValueError(f'get_y_probs() only supports single-label classification')
    if torch.is_floating_point(y):
        raise ValueError(f'get_y_probs() expects y with non-floating dtype')
    if len(y.shape) > 2:
        # recursion
        return cat_if_necessary([get_y_probs(y[i], n_classes) for i in range(y.shape[0])], dim=0)

    return torch.bincount(y.squeeze(-1), minlength=n_classes).to(torch.float32) / y.shape[0]


def insert_missing_class_columns(y_pred: torch.Tensor, train_ds: DictDataset) -> torch.Tensor:
    """
    If train_ds.tensors['y'] does not contain some of the classes specified in train_ds.tensor_infos['y']
    and if y_pred does not contain columns for these missing classes,
    add columns for the missing classes to y_pred, with small probabilities.
    :param y_pred: Tensor of logits, shape [n_batch, n_classes]
    :param train_ds: Dataset used for training the model that produced y_pred.
    :return: Returns y_pred with possibly some columns inserted.
    """
    n_classes = train_ds.tensor_infos['y'].get_cat_sizes()[0].item()
    if y_pred.shape[-1] >= n_classes:
        return y_pred  # already all columns

    # assume that the missing classes/columns in y_pred are exactly those that are not represented in the training set
    train_class_counts = torch.bincount(train_ds.tensors['y'].squeeze(-1), minlength=n_classes).cpu()
    n_missing = n_classes - y_pred.shape[-1]
    pred_col_idx = 0
    new_cols = []
    logsumexp = torch.logsumexp(y_pred, dim=-1)
    # expected posterior probability of the class under uniform prior
    # (expected value of corresponding Dirichlet distribution, which is conjugate prior to "multinoulli" distribution)
    posterior_prob = 1 / (train_ds.n_samples + n_classes)
    # ensure that the probability of missing classes is posterior_prob if y_pred are the logits
    missing_values = logsumexp + np.log(posterior_prob / (1 - posterior_prob * n_missing))
    for i in range(n_classes):
        if train_class_counts[i] > 0:
            # this column should be represented
            new_cols.append(y_pred[:, pred_col_idx])
            pred_col_idx += 1
        else:
            new_cols.append(missing_values)

    return torch.stack(new_cols, dim=-1)


def remove_missing_classes(y_pred: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Removes missing classes from y_pred and y.
    For example, if y_pred.shape[-1] == 4 but y only contains the values 0 and 2,
    the columns y_pred[..., 1] and y_pred[..., 3] will be removed and the values (0, 2) will be mapped to (0, 1).
    :param y_pred: Predictions of shape (n_samples, n_classes) (should be logits
    because probabilities will not be normalized anymore after removing columns).
    :param y: classes of shape (n_samples,)
    :return: y_pred and y with missing classes removed
    """
    # shapes: y_pred should be n_samples x n_classes, y should be n_samples
    n_classes = y_pred.shape[-1]
    counts = torch.bincount(y, minlength=n_classes)
    is_present = counts > 0
    if torch.all(is_present).item():
        # all classes are present, nothing needs to be removed
        return y_pred, y

    num_present = is_present.sum().item()
    reduced_y_pred = y_pred[..., is_present]
    class_mapping = torch.zeros(n_classes, dtype=torch.long, device=y.device)
    class_mapping[is_present] = torch.arange(num_present, dtype=torch.long, device=y.device)
    reduced_y = class_mapping[y]
    # print(f'{is_present=}, {reduced_y_pred.shape=}, {torch.unique(reduced_y)=}')
    return reduced_y_pred, reduced_y


def expected_calibration_error(y_pred: torch.Tensor, y: torch.Tensor):
    if y.is_floating_point():
        y = y.argmax(dim=-1)
    else:
        y = y.squeeze(-1)

    if len(y_pred.shape) == 3:
        # contains a n_models dimension
        y_pred_models = [y_pred[i] for i in range(y_pred.shape[0])]
        y_models = [y[i] for i in range(y.shape[0])]
    else:
        y_pred_models = [y_pred]
        y_models = [y]

    model_scores = []
    # evaluate separately for each model
    for y_pred_indiv, y_indiv in zip(y_pred_models, y_models):
        # handle classes that don't occur in the test set
        y_pred_indiv, y_indiv = remove_missing_classes(y_pred_indiv, y_indiv)

        # convert logits to probabilities
        y_pred_indiv_probs = F.softmax(y_pred_indiv, dim=-1)

        # ensure that no probabilities are zero or one to circumvent some problems
        # https://github.com/Lightning-AI/torchmetrics/issues/1646
        y_pred_indiv_probs = y_pred_indiv_probs.clamp(1e-7, 1 - 1e-7)
        y_pred_indiv_probs = y_pred_indiv_probs / y_pred_indiv_probs.sum(dim=-1, keepdim=True)

        num_classes = y_pred_indiv_probs.shape[-1]
        is_binary = num_classes == 2
        if is_binary:
            # binary classification, torchmetrics expects only probabilities of the positive class
            y_pred_indiv_probs = y_pred_indiv_probs[..., 1]

        # print(f'{torch.unique(y_indiv)=}')
        # print(f'{torch.unique(y_pred_indiv_probs)=}')
        # print(f'{y_indiv.shape=}, {y_pred_indiv_probs.shape=}')
        # print(f'{torch.min(y_pred_indiv_probs)=}')
        # print(f'{torch.max(y_pred_indiv_probs)=}')

        metric = torchmetrics.CalibrationError(task='binary' if is_binary else 'multiclass', num_classes=num_classes)
        model_scores.append(metric.forward(y_pred_indiv_probs, y_indiv))

    if len(y_pred.shape) == 3:
        # input had n_models dimension, so output should have it, too
        return torch.as_tensor(model_scores, dtype=torch.float32)
    else:
        return torch.as_tensor(model_scores[0], dtype=torch.float32)


def auc_ovr_torchmetrics(y_pred: torch.Tensor, y: torch.Tensor):
    if y.is_floating_point():
        y = y.argmax(dim=-1)
    else:
        y = y.squeeze(-1)

    if len(y_pred.shape) == 3:
        # contains a n_models dimension
        y_pred_models = [y_pred[i] for i in range(y_pred.shape[0])]
        y_models = [y[i] for i in range(y.shape[0])]
    else:
        y_pred_models = [y_pred]
        y_models = [y]

    model_scores = []
    # evaluate separately for each model
    for y_pred_indiv, y_indiv in zip(y_pred_models, y_models):
        # handle classes that don't occur in the test set
        y_pred_indiv, y_indiv = remove_missing_classes(y_pred_indiv, y_indiv)

        # convert logits to probabilities
        y_pred_indiv_probs = F.softmax(y_pred_indiv, dim=-1)

        # ensure that no probabilities are zero or one to circumvent some problems
        # https://github.com/Lightning-AI/torchmetrics/issues/1646
        y_pred_indiv_probs = y_pred_indiv_probs.clamp(1e-7, 1 - 1e-7)
        y_pred_indiv_probs = y_pred_indiv_probs / y_pred_indiv_probs.sum(dim=-1, keepdim=True)

        num_classes = y_pred_indiv_probs.shape[-1]
        is_binary = num_classes == 2
        if is_binary:
            # binary classification, torchmetrics expects only probabilities of the positive class
            y_pred_indiv_probs = y_pred_indiv_probs[..., 1]

        # print(f'{torch.unique(y_indiv)=}')
        # print(f'{torch.unique(y_pred_indiv_probs)=}')
        # print(f'{y_indiv.shape=}, {y_pred_indiv_probs.shape=}')
        # print(f'{torch.min(y_pred_indiv_probs)=}')
        # print(f'{torch.max(y_pred_indiv_probs)=}')

        metric = torchmetrics.AUROC(task='binary' if is_binary else 'multiclass', num_classes=num_classes)
        model_scores.append(metric.forward(y_pred_indiv_probs, y_indiv))

    if len(y_pred.shape) == 3:
        # input had n_models dimension, so output should have it, too
        return torch.as_tensor(model_scores, dtype=torch.float32)
    else:
        return torch.as_tensor(model_scores[0], dtype=torch.float32)


class Metrics:
    def __init__(self, metric_names, val_metric_name, task_type):
        self.metric_names = metric_names
        self.val_metric_name = val_metric_name
        self.task_type = task_type
        if val_metric_name not in metric_names:
            self.metric_names.append(val_metric_name)

    def compute_metrics_dict(self, y_preds: List[torch.Tensor], y: torch.Tensor, use_ens: bool) -> NestedDict:
        """
        :param y_preds: y predictions by (possibly multiple) ensemble members
        :param y: actual labels (one-hot encoded in case of classification)
        :param use_ens: Whether to also compute metrics for ensembled predictions
        :return: Returns a NestedDict indexed by [str(n_models), str(start_idx), metric_name]
        containing the respective metric values (float) for an ensemble using y_preds[start_idx:start_idx+n_models]
        In the ensembling case, n_models > 1 is also used, but only with start_idx = 0
        """
        if np.any([y_pred.dim() != 2 for y_pred in y_preds]):
            raise RuntimeError('Not all y_preds have dim 2')
        if y.dim() != 2:
            raise RuntimeError('y.dim() != 2')

        results_dict = NestedDict()

        # individual results
        for start_idx, y_pred in enumerate(y_preds):
            for metric_name in self.metric_names:
                result = Metrics.apply(y_pred, y, metric_name).item()
                results_dict[str(1), str(start_idx), metric_name] = float(result)

        # ensemble results
        if len(y_preds) > 1 and use_ens:
            for n_models in range(2, len(y_preds) + 1):
                y_pred = Metrics.avg_preds(y_preds[:n_models], self.task_type)
                for metric_name in self.metric_names:
                    result = Metrics.apply(y_pred, y, metric_name).cpu().numpy()
                    results_dict[str(n_models), str(0), metric_name] = float(result)

        return results_dict

    def compute_val_score(self, val_metrics_dict: NestedDict) -> float:
        # ['1'] refers to ensemble with 1 member
        # values() contains the results for the different individual models
        individual_val_scores = [indiv_dict[self.val_metric_name] for indiv_dict in val_metrics_dict['1'].values()]
        return float(np.mean(individual_val_scores))

    @staticmethod
    def apply(y_pred: torch.Tensor, y: torch.Tensor, metric_name: str) -> torch.Tensor:
        # shapes in general: n_models x n_samples x output_dim
        # for some classification metrics, y should contain the class numbers,
        # be of type torch.long and have output_dim = 1
        # for other classification metrics like cross_entropy, y can also be soft labels with output_dim = n_classes
        # in the classification case, y_pred are assumed to be logits
        invalid = torch.logical_or(torch.isnan(y_pred), torch.isinf(y_pred))
        if torch.any(invalid):
            if y.is_floating_point():
                # regression
                y_pred = torch.clone(y_pred)
                y_pred[torch.any(invalid, dim=-1), :] = 0.0
            else:
                # classification
                # y_pred[invalid] = -np.Inf  # leads to NaN after softmax()
                y_pred = torch.clone(y_pred)
                not_invalid = y_pred[~invalid]
                if len(not_invalid) == 0:
                    y_pred[invalid] = 0.0
                else:
                    y_pred[invalid] = torch.min(not_invalid) - 100   # a very small value, basically zero probability
                y_pred_probs = torch.softmax(y_pred, dim=-1)
                y_pred = torch.log(y_pred_probs + 1e-30)

        def get_y_categorical():
                if y.is_floating_point():
                    return y.argmax(dim=-1)
                return y.squeeze(-1)

        if metric_name == 'class_error':
            return torch.count_nonzero(y_pred.argmax(dim=-1) != get_y_categorical(), dim=-1) / y_pred.shape[-2]
        elif metric_name == 'cos_loss':
            return cos_loss(y_pred, y)
        elif metric_name == 'cross_entropy':
            return cross_entropy(y_pred, y)
        elif metric_name == 'n_cross_entropy':
            n_classes = y_pred.shape[-1]
            y_avg_log = torch.log(get_y_probs(y, n_classes) + 1e-30)
            # insert batch dimension and expand along batch dimension
            y_avg_log = y_avg_log.unsqueeze(-2).expand(*y_pred.shape)
            return cross_entropy(y_pred, y) / cross_entropy(y_avg_log, y)
        elif metric_name == 'ce_unif':
            return (-F.softmax(y_pred, dim=-1).log()).mean(dim=-1).mean(dim=-1)
        elif metric_name == '1-auc_ovo':
            return 1.0 - Metrics.apply_sklearn_classification_metric(
                y_pred, y, lambda y1, y2: roc_auc_score(y1, y2, multi_class='ovo'), needs_pred_probs=True)
        elif metric_name == '1-auc_ovr':
            return 1.0 - Metrics.apply_sklearn_classification_metric(
                y_pred, y, lambda y1, y2: roc_auc_score(y1, y2, multi_class='ovr'), needs_pred_probs=True)
        elif metric_name == '1-auc_ovr_alt':
            return 1.0 - auc_ovr_torchmetrics(y_pred, y)
        elif metric_name == '1-auc_mu':
            return 1.0 - Metrics.apply_sklearn_classification_metric(
                y_pred, y, auc_mu_impl, needs_pred_probs=True, two_class_single_column=False)
        elif metric_name == 'brier':
            return brier_loss(y_pred, y)
        elif metric_name == 'n_brier':
            n_classes = y_pred.shape[-1]
            y_avg_log = torch.log(get_y_probs(y, n_classes) + 1e-30)
            # insert batch dimension and expand along batch dimension
            y_avg_log = y_avg_log.unsqueeze(-2).expand(*y_pred.shape)
            return brier_loss(y_pred, y) / brier_loss(y_avg_log, y)
        elif metric_name == '1-balanced_accuracy':
            return 1.0 - Metrics.apply_sklearn_classification_metric(y_pred, y, balanced_accuracy_score,
                                                                     needs_pred_probs=False)
        elif metric_name == '1-mcc':
            return 1.0 - Metrics.apply_sklearn_classification_metric(y_pred, y, matthews_corrcoef,
                                                                     needs_pred_probs=False)
        elif metric_name == 'ece':
            return expected_calibration_error(y_pred, y)
        elif metric_name == 'rmse':
            return mse(y_pred, y).sqrt()
        elif metric_name == 'nrmse':
            # rmse relative to rmse of the best constant predictor
            rmse = mse(y_pred, y).sqrt()
            den = y.std(correction=0)
            return rmse/den
        elif metric_name == 'mae':
            return (y_pred - y).abs().mean(dim=-1).mean(dim=-1)
        elif metric_name == 'nmae':
            # mae relative to mae of the best constant predictor
            median = torch.median(y)
            mae = (y_pred - y).abs().mean(dim=-1).mean(dim=-1)
            den = (median - y).abs().mean(dim=-1).mean(dim=-1)
            return mae/den
        elif metric_name == 'max_error':
            return (y_pred - y).abs().max(dim=-1)[0].max(dim=-1)[0]
        elif metric_name == 'n_max_error':
            # max error relative to the max error of the best constant predictor
            max_error = (y_pred - y).abs().max(dim=-1)[0].max(dim=-1)[0]
            max = y.max(dim=-1)[0].max(dim=-1)[0]
            min = y.min(dim=-1)[0].min(dim=-1)[0]
            ref_error = (0.5 * (max-min))
            return max_error / (ref_error + 1e-30)
        elif metric_name.startswith('pinball('):
            # expected format: pinball(number), e.g. pinball(0.95)
            quantile = float(metric_name[len('pinball('):-1])
            return pinball_loss(y_pred, y, quantile)
        elif metric_name.startswith('n_pinball('):
            # expected format: n_pinball(number), e.g. n_pinball(0.95)
            # compute loss divided by loss of the best constant predictor
            quantile = float(metric_name[len('n_pinball('):-1])
            raw_loss = pinball_loss(y_pred, y, quantile)
            best_constant_y_pred = torch_np_quantile(y, quantile, dim=-2, keepdim=True).expand(*y_pred.shape)
            best_constant_loss = pinball_loss(best_constant_y_pred, y, quantile)
            return raw_loss / (best_constant_loss + 1e-30)
        else:
            raise ValueError(f'Unknown metric {metric_name}')

    @staticmethod
    def apply_sklearn_classification_metric(y_pred: torch.Tensor, y: torch.Tensor, metric_function: Callable,
                                            needs_pred_probs: bool, two_class_single_column: bool = True):
        if y.is_floating_point():
            y = y.argmax(dim=-1)
        else:
            y = y.squeeze(-1)

        if len(y_pred.shape) == 3:
            # contains a n_models dimension
            y_pred_models = [y_pred[i] for i in range(y_pred.shape[0])]
            y_models = [y[i] for i in range(y.shape[0])]
        else:
            y_pred_models = [y_pred]
            y_models = [y]

        model_scores = []
        # evaluate separately for each model
        for y_pred_indiv, y_indiv in zip(y_pred_models, y_models):
            # handle classes that don't occur in the test set
            y_pred_indiv, y_indiv = remove_missing_classes(y_pred_indiv, y_indiv)

            if needs_pred_probs:
                # convert logits to probabilities
                y_pred_np = F.softmax(y_pred_indiv, dim=-1).cpu().numpy()
                if y_pred_np.shape[-1] == 2 and two_class_single_column:
                    # binary classification, scikit-learn expects only probabilities of the positive class
                    y_pred_np = y_pred_np[..., 1]
            else:
                # convert logits to predicted class
                y_pred_np = torch.argmax(y_pred_indiv, dim=-1).cpu().numpy()

            y_np = y_indiv.cpu().numpy()
            model_scores.append(metric_function(y_np, y_pred_np))

        if len(y_pred.shape) == 3:
            # input had n_models dimension, so output should have it, too
            return torch.as_tensor(model_scores, dtype=torch.float32)
        else:
            return torch.as_tensor(model_scores[0], dtype=torch.float32)


    @staticmethod
    def avg_preds(y_preds: List[torch.Tensor], task_type):
        if task_type == TaskType.CLASSIFICATION:
            # it should be logmeanexp, but doesn't matter because it is normalized by softmax
            # y_pred = torch.logsumexp(torch.stack(y_preds, dim=0), dim=0)
            probs = [F.softmax(y_pred, dim=-1) for y_pred in y_preds]
            avg_probs = sum(probs) / len(probs)
            y_pred = torch.log(avg_probs + 1e-30)
        else:
            y_pred = sum(y_preds) / len(y_preds)
        return y_pred

    @staticmethod
    def defaults(y_cat_sizes, val_metric_name: Optional[str] = None) -> 'Metrics':
        if val_metric_name is None:
            val_metric_name = 'class_error' if y_cat_sizes[0] > 0 else 'rmse'

        # removed cos_loss
        default_class_metrics = ['class_error', 'cross_entropy', 'ce_unif', 'brier',
                                 'n_cross_entropy', 'n_brier',
                                 '1-balanced_accuracy', '1-mcc', 'ece', '1-auc_ovo', '1-auc_ovr']

        if len(y_cat_sizes) == 1 and y_cat_sizes[0] == 2:
            # bin class
            return Metrics(default_class_metrics, val_metric_name, TaskType.CLASSIFICATION)
        elif y_cat_sizes[0] > 0:
            if y_cat_sizes[0] > 100:
                default_class_metrics = [m for m in default_class_metrics if m != '1-auc_ovo']
            # multi-class (or multi-label classification)
            return Metrics(default_class_metrics, val_metric_name, TaskType.CLASSIFICATION)
        else:  # regression
            return Metrics(['rmse', 'mae', 'max_error', 'nrmse', 'nmae', 'n_max_error'],
                           val_metric_name, TaskType.REGRESSION)

    @staticmethod
    def default_metric_name(task_type):
        if task_type == TaskType.CLASSIFICATION:
            return 'class_error'
        elif task_type == TaskType.REGRESSION:
            return 'rmse'
        else:
            raise ValueError(f'Unknown task type {task_type}')


